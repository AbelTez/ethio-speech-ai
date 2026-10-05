"""Whole-answer integration with tested OmniVoice mixed precision and stage timings."""
import argparse
import csv
from pathlib import Path
from uuid import uuid4

from .artifacts import read_json,write_json,now,safe_name,export_run
from .pipeline_benchmark import prepare,read_audio,run_turn,verify_result,review_for
from .pipeline_core import accepted,metrics
from .tts_profile import ProfiledTTS,HOOKS
from .tts_precision import PrecisionTTS

VERSION='pipeline-v3-omni-mixed-precision-profiled'


class IntegratedMixedPrecisionTTS(PrecisionTTS):
    def __init__(self,backend,**kwargs):
        super().__init__(backend,**kwargs)
        self.settings={**backend.settings,'precision':'FP32 weights; FP16 autocast speech generation',
                       'codec_precision':'float32','logits_scoring_precision':'float32',
                       'numerical_checks':'same as accepted standalone precision comparison'}

    def generate(self,text):
        return super().generate(text,'amp_fp16')


def prepare_v3(repo,output,owner,input_dir):
    return prepare(repo,output,owner,input_dir,
        dataset_file=repo/'data/pipeline_public/questions_v2.json',version=VERSION,
        settings_extra={'tts':'FP32 weights / FP16 autocast generation / FP32 codec / 32 steps / auto voice',
            'tts_precision_mode':'amp_fp16','numerical_checks':True,'mode':'sequential','external_chunking':False,'max_words_per_clause':None,
            'everyday_instruction':'accepted direct v1','grounded_instruction':'unchanged v1',
            'tts_profiling':'synchronized inclusive/exclusive stage wall times','profile_hooks':list(HOOKS)})


def run(repo,output,owner,input_dir):
    root,data=prepare_v3(repo,output,owner,input_dir)
    todo=[]
    for sample in data['samples']:
        path=root/'results'/f'{sample["id"]}__sequential.json'
        if path.exists() and read_json(path)['status']=='success':
            verify_result(root,read_json(path));print('Reusing:',sample['id'],flush=True)
        else:todo.append(sample)
    if not todo:print('All three turns already saved; no model loaded.',flush=True);return
    import torch
    from .pipeline_backend import PipelineModels
    session=uuid4().hex[:12];capacity_rows=[]
    def capacity(stage):
        torch.cuda.synchronize();free,total=torch.cuda.mem_get_info()
        row={'stage':stage,'free_gib':free/2**30,'total_gib':total/2**30,
             'allocated_gib':torch.cuda.memory_allocated()/2**30,
             'reserved_gib':torch.cuda.memory_reserved()/2**30,'time':now()}
        capacity_rows.append(row);write_json(root/'sessions'/f'{session}.json',{'capacity':capacity_rows})
        print('CAPACITY',row,flush=True)
    try:
        models=PipelineModels(repo,capacity)
        models.tts=ProfiledTTS(IntegratedMixedPrecisionTTS(models.tts))
        write_json(root/'tts_settings.json',{'settings':models.tts.settings,
            'observed_hooks':models.tts.profiler.observed,'missing_hooks':models.tts.profiler.missing})
        print('TTS profiling hooks:',models.tts.profiler.observed,flush=True)
        if models.tts.profiler.missing:print('Unavailable hooks (will not be inferred):',models.tts.profiler.missing,flush=True)
        audio,sr=read_audio(root/'inputs'/data['samples'][0]['audio_file'])
        models.warmup(audio,sr,capacity)
        for sample in todo:
            print('Running whole answer:',sample['id'],flush=True)
            try:run_turn(models,sample,'sequential',root,session)
            except Exception as exc:
                path=root/'results'/f'{sample["id"]}__sequential.json'
                r=read_json(path) if path.exists() else {}
                if isinstance(exc,torch.OutOfMemoryError) or 'CUDA' in str(exc) or r.get('worker_still_running'):raise
                print('Saved failed turn:',str(exc),flush=True)
    except BaseException as exc:
        import traceback
        write_json(root/'sessions'/f'{session}-error.json',{'time':now(),'error':str(exc),
                  'traceback':traceback.format_exc(),'capacity':capacity_rows});raise


def report(output,owner):
    base=output/safe_name(owner)
    root=base/'runs'/safe_name(read_json(base/'latest.json')['run_id'])
    rows=[]
    for path in sorted((root/'results').glob('*.json')):
        r=read_json(path);verify_result(root,r);v=review_for(root,path)
        m=metrics(r)
        ok=accepted(r,v) and v.get('complete_ending') is True
        row={'sample_id':r['sample_id'],'status':r['status'],'transcript':r.get('transcript',''),
            'answer':r.get('answer',''),'stop_reason':r.get('stop_reason'),
            'reviewed':bool(v.get('reviewed')),'quality_accepted':ok,
            'accepted_target_under_3s':bool(ok and m.get('candidate_tts_start_from_asr_s',999)<3),
            **m,'notes':v.get('notes',''),'error':r.get('error','')}
        if r['chunks']:
            if len(r['chunks'])!=1:raise ValueError('Expected one whole-answer TTS call.')
            details=r['chunks'][0]['details'];p=details.get('stage_profile',{})
            row.update(tts_precision_mode=details.get('precision_mode'),
                       tts_forward_logits_dtypes=details.get('forward_logits_dtypes'),
                       tts_codec_output_dtype=details.get('codec_output_dtype'),
                       tts_logits_finite=details.get('logits_finite'),
                       native_chunked=details.get('native_chunked'),
                       native_synthesis_calls=details.get('native_synthesis_calls'),
                       tts_profile_total_s=p.get('total_s'),tts_unattributed_s=p.get('unattributed_s'))
            for stage,timing in p.get('stages',{}).items():
                row['tts_'+stage+'_exclusive_s']=timing['exclusive_s']
        rows.append(row)
    dest=base/'reports';dest.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with (dest/'pipeline.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write_json(dest/'results.json',rows)
    lines=['# Whole-answer integrated pipeline v3','',
        'Hohe FP16 → Gemma 4 E4B NF4 → OmniVoice FP16 autocast generation / FP32 weights and codec / 32 steps / auto voice.',
        'Same questions, instructions and whole-answer synthesis as pipeline v2; tested mixed precision added.',
        'Precision and finite-logit checks are required. No silent FP32 fallback. Timings include numerical checks.',
        'TTS-only precision timings are not integrated timings; compare this run with its actual transcripts, answers, hardware and capacity.',
        'Three recorded questions, one sequential pass. One TTS call per complete answer, punctuation retained.',
        'All models resident; loading and warmup excluded. Instrumented timings include synchronization overhead.',
        'Target = first ASR model inference invocation → TTS generate invocation; waveform availability is separate.',
        'No browser playback, VAD, speculative execution or production latency claim.',
        'Quality acceptance additionally requires a complete spoken ending. Missing reviews are not passes.',
        'TTS exclusive stage times avoid counting nested work twice. Inclusive method times remain in result JSON.',
        'Decode-dispatch/transfer is residual work inside the decode wrapper, not pure GPU copy time; unattributed includes wrapper/dispatch/instrumentation overhead.',
        'Missing profiling hooks are reported explicitly, never treated as evidence of zero cost.', '',
        '|Question|Status|ASR → TTS start s|ASR → audio ready s|Quality|',
        '|---|---|---|---|---|']
    for r in rows:
        fmt=lambda key: f'{r[key]:.3f}' if r.get(key) is not None else '—'
        quality=('pass' if r['quality_accepted'] else 'fail') if r['reviewed'] else 'unreviewed'
        lines.append(f'|{r["sample_id"]}|{r["status"]}|{fmt("candidate_tts_start_from_asr_s")}|{fmt("first_audio_ready_from_asr_s")}|{quality}|')
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','export'])
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--owner',default='abel')
    p.add_argument('--inputs',type=Path)
    a=p.parse_args();safe_name(a.owner)
    if a.action=='run':
        if a.inputs is None:p.error('--inputs required')
        try:run(a.repo,a.output,a.owner,a.inputs)
        finally:
            if (a.output/a.owner/'latest.json').exists():report(a.output,a.owner)
    else:
        report(a.output,a.owner)
        print('RESULT ZIP:',export_run(a.output/a.owner,a.output/f'pipeline-v3-{a.owner}-{uuid4().hex[:10]}.zip'),flush=True)

if __name__=='__main__':main()

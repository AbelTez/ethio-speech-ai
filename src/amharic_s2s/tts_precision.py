"""Paired OmniVoice FP32 / FP16 autocast screening on reviewed fixed answers."""
import argparse
import csv
from functools import wraps
from pathlib import Path
import time
import traceback
from uuid import uuid4

from .artifacts import (read_json,write_json,digest,object_digest,source_snapshot,
                        environment,safe_name,now,export_run)
from .tts_benchmark import validate_waveform
from .tts_sentences import validate_dataset,verify_audio,accepted,RATINGS

MODES=('fp32','amp_fp16')
PROTOCOL={'version':'tts-precision-v1','steps':32,'seed':42,'guidance_scale':2.0,
          'voice':'auto','external_chunking':False,'punctuation':'retained verbatim',
          'weights':'float32','codec':'float32; autocast disabled',
          'amp_scope':'native iterative speech-token generation; logits and scoring kept FP32',
          'timing':'synchronized generate through CPU waveform; includes numerical checks; excludes load, warmup and saving',
          'order':'alternate mode order by sentence; one measured call per sentence and mode',
          'deployment':'TTS alone; compare only fresh pairs in the same session, not prior integrated timings'}


class PrecisionTTS:
    """Autocast iterative generation only; never convert the codec or model weights."""
    def __init__(self,backend,device_type='cuda',amp_dtype=None):
        self.backend=backend;self.torch=backend.torch;self.device_type=device_type
        self.amp_dtype=amp_dtype or self.torch.float16
        self.mode='fp32';self.running=False
        model=backend.model
        if getattr(model,'_precision_comparison_installed',False):raise ValueError('Precision wrapper already installed.')
        for p in model.parameters():
            if p.is_floating_point() and p.dtype!=self.torch.float32:
                raise ValueError('Expected FP32 master weights, including the codec.')
        iterative=model._generate_iterative
        forward=model.forward
        scoring=model._predict_tokens_with_scoring
        decode=model.audio_tokenizer.decode
        @wraps(iterative)
        def generate_tokens(*args,**kwargs):
            with self.torch.autocast(self.device_type,dtype=self.amp_dtype,enabled=self.mode=='amp_fp16'):
                return iterative(*args,**kwargs)
        @wraps(forward)
        def checked_forward(*args,**kwargs):
            out=forward(*args,**kwargs)
            if self.running:
                self.forward_dtypes.add(str(out.logits.dtype))
                # Defer host synchronization until generation ends. Do not repair invalid logits.
                self.finite.append(self.torch.isfinite(out.logits).all())
            out.logits=out.logits.float()
            return out
        @wraps(scoring)
        def score(*args,**kwargs):
            with self.torch.autocast(self.device_type,enabled=False):
                return scoring(*args,**kwargs)
        @wraps(decode)
        def decode_fp32(*args,**kwargs):
            with self.torch.autocast(self.device_type,enabled=False):
                result=decode(*args,**kwargs)
                if self.running:
                    self.codec_calls+=1
                    if result.audio_values.dtype!=self.torch.float32:
                        raise RuntimeError('Codec returned non-FP32 audio.')
                return result
        model._generate_iterative=generate_tokens;model.forward=checked_forward
        model._predict_tokens_with_scoring=score;model.audio_tokenizer.decode=decode_fp32
        model._precision_comparison_installed=True

    def __getattr__(self,name):return getattr(self.backend,name)

    def generate(self,text,mode):
        if mode not in MODES:raise ValueError('Unknown precision mode.')
        if self.running:raise RuntimeError('Concurrent calls are not supported.')
        self.mode=mode;self.forward_dtypes=set();self.finite=[];self.codec_calls=0;self.running=True
        try:
            with self.torch.autocast(self.device_type,enabled=False):
                wave,sr,details=self.backend.generate(text)
            if not self.finite or not self.codec_calls:raise RuntimeError('Expected generation/codec hooks were not called.')
            if not self.torch.stack(self.finite).all().item():raise FloatingPointError('Nonfinite speech logits; no fallback applied.')
            expected=str(self.amp_dtype if mode=='amp_fp16' else self.torch.float32)
            if self.forward_dtypes!={expected}:raise RuntimeError(f'Unexpected logits precision: {self.forward_dtypes}; expected {expected}')
            return wave,sr,{**details,'precision_mode':mode,'forward_calls':len(self.finite),
                'forward_logits_dtypes':sorted(self.forward_dtypes),'codec_decode_calls':self.codec_calls,
                'codec_output_dtype':'torch.float32','logits_finite':True}
        finally:self.running=False


def prepare(repo,output,owner,env=None):
    data=read_json(repo/'data/tts_public/precision_answers.json');validate_dataset(data)
    cat=read_json(repo/'configs/tts_candidates.json')
    cfg={'dataset':data,'candidate':cat['candidates']['omni_amharic'],'codec':cat['codec'],
         'protocol':PROTOCOL,'environment':environment() if env is None else env,'source':source_snapshot(repo)}
    name='tts-precision-'+object_digest(cfg)[:20]
    root=output/safe_name(owner)/'runs'/name
    if not (root/'run.json').exists():write_json(root/'run.json',{'run_id':name,'created_at':now(),'owner':owner,**cfg})
    write_json(output/owner/'active.json',{'run_id':name})
    print('RUN_ID:',name,flush=True)
    return root,cfg


def schedule(samples):
    return [(sample,mode) for i,sample in enumerate(samples) for mode in (MODES if i%2==0 else MODES[::-1])]


def generate(root,cfg,backend_factory=None):
    import soundfile as sf
    todo=[]
    for sample,mode in schedule(cfg['dataset']['samples']):
        p=root/'results'/f'{sample["id"]}__{mode}.json'
        if p.exists() and read_json(p)['status']=='success':verify_audio(root,read_json(p))
        else:todo.append((sample,mode))
    if not todo:print('All six clips saved; no model loaded.',flush=True);return
    session=uuid4().hex;stage='loading'
    try:
        start=time.perf_counter()
        if backend_factory is None:
            from .tts_backend import LocalTTS
            backend=PrecisionTTS(LocalTTS(cfg['candidate'],cfg['codec'],device='cuda',steps=32,reference=None))
        else:backend=backend_factory(cfg)
        backend.sync();loaded=time.perf_counter()
        session_info={'load_s':loaded-start,'settings':backend.settings,'loading_info':backend.loading_info,
                      'warmups':{},'environment':cfg['environment']}
        write_json(root/'sessions'/f'{session}.json',session_info)
        for mode in MODES:
            stage='warmup_'+mode;backend.seed(41);backend.sync();start=time.perf_counter()
            wave,sr,details=backend.generate('ሰላም፣ እንዴት ነዎት?',mode);backend.sync()
            validate_waveform(wave,sr)
            session_info['warmups'][mode]={'seconds':time.perf_counter()-start,'details':details}
            write_json(root/'sessions'/f'{session}.json',session_info)
        for sample,mode in todo:
            stage=sample['id']+'__'+mode
            r={'sample_id':sample['id'],'mode':mode,'text':sample['text'],'expected':sample['expected'],
               'status':'running','session':session}
            try:
                backend.seed(PROTOCOL['seed']);backend.sync();start=time.perf_counter()
                wave,sr,details=backend.generate(sample['text'],mode);backend.sync()
                elapsed=time.perf_counter()-start
                wave,metrics=validate_waveform(wave,sr)
                relative='audio/'+stage+'.wav';path=root/relative;path.parent.mkdir(exist_ok=True)
                pending=path.with_name('.pending-'+uuid4().hex+'.wav')
                sf.write(pending,wave,sr,subtype='FLOAT');pending.replace(path)
                r.update(status='success',metrics={**metrics,'generation_s':elapsed,'rtf':elapsed/metrics['audio_duration_s']},
                         details=details,audio_file=relative,audio_sha256=digest(path))
                print(stage,'| generation',round(elapsed,3),'s | audio',round(metrics['audio_duration_s'],3),'s',flush=True)
            except BaseException as exc:
                r.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
            finally:write_json(root/'results'/f'{stage}.json',r)
    except BaseException as exc:
        write_json(root/'failures'/f'{session}.json',{'stage':stage,'error':str(exc),'traceback':traceback.format_exc()})
        raise


def report(output,owner):
    base=output/safe_name(owner);root=base/'runs'/safe_name(read_json(base/'active.json')['run_id'])
    rows=[]
    for p in sorted((root/'results').glob('*.json')):
        r=read_json(p);verify_audio(root,r)
        path=root/'reviews'/p.name;v=read_json(path) if path.exists() else {}
        if v.get('result_sha256')!=digest(p):v={}
        d=r.get('details',{});m=r.get('metrics',{})
        rows.append({'sample_id':r['sample_id'],'mode':r['mode'],'session':r['session'],
            'text':r['text'],'status':r['status'],'generation_s':m.get('generation_s'),
            'audio_duration_s':m.get('audio_duration_s'),'rtf':m.get('rtf'),
            'reviewed':all(type(v.get(k)) is bool for k in RATINGS) and type(v.get('naturalness')) is int,
            'quality_pass':accepted(r,v),**{k:v.get(k) for k in RATINGS},'naturalness':v.get('naturalness'),
            'notes':v.get('notes',''),'native_chunked':d.get('native_chunked'),
            'forward_logits_dtypes':d.get('forward_logits_dtypes'),'codec_output_dtype':d.get('codec_output_dtype'),
            'error':r.get('error','')})
    pairs=[]
    for s in read_json(root/'run.json')['dataset']['samples']:
        modes={r['mode']:r for r in rows if r['sample_id']==s['id']}
        a,b=modes.get('fp32',{}),modes.get('amp_fp16',{})
        comparable=all(r.get('status')=='success' for r in (a,b)) and a['session']==b['session']
        quality=comparable and a['quality_pass'] and b['quality_pass']
        pairs.append({'sample_id':s['id'],'same_session_complete_pair':comparable,'both_quality_pass':quality,
            'fp32_over_amp_speedup':a['generation_s']/b['generation_s'] if comparable else None,
            'amp_naturalness_at_least_control':bool(quality and b['naturalness']>=a['naturalness'])})
    dest=base/'reports';dest.mkdir(parents=True,exist_ok=True)
    for name,records in [('clips',rows),('pairs',pairs)]:
        write_json(dest/(name+'.json'),records)
        with (dest/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]) if records else ['status']);w.writeheader();w.writerows(records)
    lines=['# OmniVoice precision comparison','',
        'TTS only; exact accepted answers; 32 steps; auto voice; punctuation retained; one complete answer per call.',
        'FP32 master weights and codec. Candidate uses FP16 autocast in iterative generation; logits/scoring stay FP32.',
        'One warmup per mode, then one measured call per answer/mode. Times include numerical checks; exclude load/warmup/saving/playback.',
        'Compare fresh same-session pairs only. Different-session resumed pairs are not ranked. No comparison to earlier integrated timings or end-to-end claim.',
        'Quality requires intelligibility, faithful words, complete ending and naturalness >=3. Check naturalness against the control too.', '',
        '|Answer|Mode|Status|Generation s|Quality|Naturalness|','|---|---|---|---|---|---|']
    for r in rows:
        q=('pass' if r['quality_pass'] else 'fail') if r['reviewed'] else 'unreviewed'
        t=f'{r["generation_s"]:.3f}' if r['generation_s'] is not None else '—'
        lines.append(f'|{r["sample_id"]}|{r["mode"]}|{r["status"]}|{t}|{q}|{r["naturalness"]}|')
    failures=list((root/'failures').glob('*.json'))
    if failures:lines+=['',f'{len(failures)} failure log(s) saved under runs/{root.name}/failures. Inspect them before retrying.']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','export'])
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--owner',default='abel')
    a=p.parse_args();safe_name(a.owner)
    if a.action=='run':
        root,cfg=prepare(a.repo,a.output,a.owner)
        try:generate(root,cfg)
        finally:report(a.output,a.owner)
    else:
        report(a.output,a.owner)
        print('RESULT ZIP:',export_run(a.output/a.owner,a.output/f'tts-precision-{a.owner}-{uuid4().hex[:10]}.zip'),flush=True)

if __name__=='__main__':main()

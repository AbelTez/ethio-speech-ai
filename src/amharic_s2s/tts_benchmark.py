"""Resumable local TTS listening comparison. No ASR judge or hosted synthesis."""
import argparse,csv,io,json,os,time,traceback
from pathlib import Path
from uuid import uuid4
from .artifacts import (read_json,write_json,digest,object_digest,environment,now,
    safe_name,save_result,completed_keys,export_run)

PROTOCOL={'version':'tts-listen-v3','seed':42,'planned_repeats':1,'precision':'float32',
 'language':'Amharic','batch_size':1,'guidance_scale':2.0,
 'timing':'synchronized call start through CPU waveform availability; excludes load, warmup, WAV write and playback',
 'audio':'native model sample rate; FLOAT WAV; no external normalization or resampling',
 'first_audio':'not measured; these generation APIs return completed waveforms',
 'normalization':'OmniVoice optional normalization disabled; MMS uses its required native tokenizer/uroman'}
# Synthesis includes decoding/postprocessing and dispatch, not GPU kernels alone.
PROTOCOL['timing_breakdown']='preparation_s: MMS cached romanization/tokenization/device input, or OmniVoice native _preprocess_all + _prepare_inference_inputs; synthesis_s: total minus preparation, including codec, postprocessing and dispatch'
PROTOCOL['mms_romanizer']='one instance per loaded backend; token IDs must match Transformers 5.17.0 native preparation'
PROTOCOL['omni_splitter']='ethiopic-punctuation-with-word-fallback-v3; exact character preservation; native duration threshold/target unchanged'
REVIEW=['intelligible','faithful','naturalness_1_to_5','notes']


def source(repo):
    files=['src/amharic_s2s/artifacts.py','src/amharic_s2s/tts_backend.py',
        'src/amharic_s2s/tts_benchmark.py','src/amharic_s2s/tts_text.py','src/amharic_s2s/tts_chunking.py','scripts/tts_compare.py','requirements/tts-colab.txt']
    return {f:digest(repo/f) for f in files}


def prepare(repo,output,owner,key,device,steps,reference=None,env=None,suite='short'):
    safe_name(owner);safe_name(key)
    catalog=read_json(repo/'configs/tts_candidates.json');candidate=catalog['candidates'][key]
    from .tts_backend import dataset_path
    dataset=read_json(dataset_path(repo,suite))
    ids=[safe_name(s['id']) for s in dataset['samples']]
    if len(ids)!=len(set(ids)) or any(not s['text'].strip() for s in dataset['samples']):raise ValueError('Invalid prompts')
    if candidate['loader']=='mms' and (reference or steps!=32):raise ValueError('MMS has no diffusion steps or reference voice: use defaults.')
    if candidate['loader']=='omni' and device!='cuda':raise ValueError('Run OmniVoice on GPU.')
    ref=None
    if reference:
        if not reference['text'].strip():raise ValueError('Exact reference transcript is required.')
        ref={'audio_sha256':digest(reference['path']),'text':reference['text']}
    config={'stage':'tts','planned_repeats':1,'suite':suite,'model_key':key,'candidate':candidate,
        'codec':catalog['codec'] if candidate['loader']=='omni' else None,
        'device':device,'steps':steps if candidate['loader']=='omni' else None,
        'reference':ref,'voice_mode':'reference' if ref else ('auto' if candidate['loader']=='omni' else 'fixed'),
        'protocol':PROTOCOL}
    if env is None:
        import importlib.metadata
        env=environment()
        for name in ('uroman','pydub','librosa','tensorboardX','webdataset'):
            try:env['packages'][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:env['packages'][name]=None
    if device=='cuda' and not env.get('gpu'):raise RuntimeError('GPU not available; select a GPU runtime.')
    hashes=source(repo)
    identity={'config':object_digest(config),'dataset':object_digest(dataset),'source':object_digest(hashes),'environment':object_digest(env)}
    name=f'tts-{key}-{object_digest(identity)[:20]}'
    directory=output/owner/'runs'/name
    if (directory/'run.json').exists():
        if read_json(directory/'run.json')['identity']!=identity or read_json(directory/'config.json')!=config or read_json(directory/'dataset.json')!=dataset:
            raise ValueError('Run metadata changed; refusing overwrite.')
    else:
        if any((directory/'results').glob('*.json')):raise ValueError('Results exist without metadata.')
        write_json(directory/'config.json',config);write_json(directory/'dataset.json',dataset)
        write_json(directory/'run.json',{'run_id':name,'owner':owner,'created_at':now(),'identity':identity,'environment':env,'source':hashes})
    write_json(output/owner/'active'/f'{key}-{suite}-{steps}.json',{'run_name':name})
    return directory,config,dataset


def verified_done(directory):
    done=completed_keys(directory)
    for sid,repeat in done:
        r=read_json(directory/'results'/f'{sid}__{repeat}.json');audio=directory/r['details']['audio_file']
        if not audio.is_file() or digest(audio)!=r['details']['audio_sha256']:
            raise RuntimeError(f'Saved audio missing or modified: {audio}. Preserve this run and restore the file.')
    return done


def validate_waveform(wav,sr):
    import numpy as np
    wav=np.asarray(wav,dtype=np.float32)
    if wav.ndim==2 and 1 in wav.shape:wav=wav.reshape(-1)
    if wav.ndim!=1 or not len(wav) or not np.isfinite(wav).all():raise ValueError('Empty, non-mono or non-finite generated audio.')
    if type(sr) is not int or sr<=0:raise ValueError('Invalid sample rate.')
    peak=float(np.max(np.abs(wav)))
    if peak<1e-7:raise ValueError('Generated waveform is silent.')
    return wav,{'audio_duration_s':len(wav)/sr,'sample_rate_hz':sr,'peak_abs':peak,
        'rms':float(np.sqrt(np.mean(wav.astype('float64')**2))),
        'out_of_range_fraction':float(np.mean(np.abs(wav)>1))}


def measure(directory,dataset,backend,limit):
    import soundfile as sf
    selected=dataset['samples'][:limit] if limit else dataset['samples']
    done=verified_done(directory)
    for sample in selected:
        if (sample['id'],1) in done:continue
        temp=None
        try:
            seed=PROTOCOL['seed']+int(object_digest(sample['id'])[:6],16)
            backend.seed(seed);backend.sync()
            t=time.perf_counter();wav,sr,extra=backend.generate(sample['text']);backend.sync()
            elapsed=time.perf_counter()-t
            wav,metrics=validate_waveform(wav,sr)
            metrics.update(generation_s=elapsed,rtf=elapsed/metrics['audio_duration_s'])
            prep=extra.pop('preparation_s',None)
            if prep is not None:
                if not 0<=prep<=elapsed:raise ValueError('Invalid preparation timing boundaries.')
                metrics.update(preparation_s=prep,synthesis_s=elapsed-prep)
            metrics.update(input_words=len(sample['text'].split()),input_characters=len(sample['text']))
            target=directory/'audio'/f"{sample['id']}.wav";target.parent.mkdir(exist_ok=True)
            temp=target.with_name('.pending-'+uuid4().hex+'.wav')
            sf.write(temp,wav,sr,subtype='FLOAT');os.replace(temp,target)
            save_result(directory,sample['id'],1,metrics,details={'text':sample['text'],'expected':sample['expected'],
                'slice':sample['slice'],'seed':seed,'audio_file':str(target.relative_to(directory)),
                'audio_sha256':digest(target),**extra})
            print(f"SAVED {sample['id']} | {elapsed:.3f}s | {metrics['audio_duration_s']:.2f}s audio | RTF {metrics['rtf']:.3f}",flush=True)
        except Exception as exc:
            if temp and temp.exists():temp.unlink()
            save_result(directory,sample['id'],1,{},'failed',f'{type(exc).__name__}: {exc}',details={'text':sample['text'],'expected':sample['expected']})
            raise


def run(a):
    if os.environ.get('CUDA_LAUNCH_BLOCKING')=='1':raise ValueError('Remove CUDA_LAUNCH_BLOCKING before timing.')
    from .tts_backend import LocalTTS
    import torch
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Select a CUDA GPU runtime.')
    reference={'path':a.reference_wav,'text':a.reference_text} if a.reference_wav else None
    directory,config,dataset=prepare(a.repo,a.output,a.owner,a.model,a.device,a.steps,reference,suite=a.suite)
    print('Run:',directory.name,'\nOutput:',directory,flush=True)
    done=verified_done(directory)
    smoke_limit=1 if a.suite=='long' else 2
    if a.limit==0:
        if a.reviewed_run!=directory.name or not all((s['id'],1) in done for s in dataset['samples'][:smoke_limit]):
            raise ValueError('Listen to the saved smoke WAV(s) first, then pass the current run as --reviewed-run.')
    selected=dataset['samples'][:a.limit] if a.limit else dataset['samples']
    if all((s['id'],1) in done for s in selected):print('Already saved; no model loaded.');return
    stage='loading';started=time.perf_counter()
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.manual_seed(PROTOCOL['seed'])
        backend=LocalTTS(config['candidate'],config['codec'],a.device,a.steps,reference)
        backend.sync();load_s=time.perf_counter()-started
        stage='warmup';start=time.perf_counter();backend.seed(41)
        warm,sr,_=backend.generate('ሰላም፣ እንዴት ነዎት?');backend.sync();validate_waveform(warm,sr)
        write_json(directory/'loads'/f'{uuid4().hex}.json',{'load_including_download_s':load_s,'warmup_s':time.perf_counter()-start,'settings':backend.settings,'loading_info':backend.loading_info})
        stage='measured_generation';measure(directory,dataset,backend,a.limit)
    except Exception as exc:
        write_json(directory/'session_errors'/f'{uuid4().hex}.json',{'at':now(),'stage':stage,'error':str(exc),'traceback':traceback.format_exc()})
        raise


def csv_write(path,rows,fields):
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    path.write_text(buf.getvalue(),encoding='utf-8-sig')


def report(output,owner):
    import statistics
    safe_name(owner);home=output/owner;dest=home/'reports';dest.mkdir(parents=True,exist_ok=True)
    review_path=home/'listening_review.csv';reviews={}
    if review_path.exists():
        with review_path.open(encoding='utf-8-sig',newline='') as f:
            reviews={(r['run_id'],r['sample_id']):r for r in csv.DictReader(f)}
    rows=[];summaries=[]
    for p in sorted((home/'runs').glob('*/run.json')):
        directory=p.parent;m=read_json(p);c=read_json(directory/'config.json');d=read_json(directory/'dataset.json')
        group=object_digest({k:v for k,v in m['identity'].items() if k!='config'}|{'protocol':c['protocol'],'device':c['device'],'reference':c['reference']})[:12]
        base={'run_id':directory.name,'suite':c.get('suite','short'),'model_key':c['model_key'],'model_id':c['candidate']['model_id'],'revision':c['candidate']['revision'],'comparison_group':group,'device':c['device'],'precision':'float32','steps':c['steps'],'voice_mode':c['voice_mode'],'license':c['candidate']['license']}
        rr=[]
        for record in sorted((directory/'results').glob('*.json')):
            r=read_json(record);details=r.get('details',{})
            row={**base,'sample_id':r['sample_id'],'status':r['status'],'error':r['error'],'text':details.get('text',''),'expected':details.get('expected',''),**r['metrics'],
                'audio_file':str((directory/details['audio_file']).relative_to(home)) if details.get('audio_file') else '',
                'audio_sha256':details.get('audio_sha256',''),'native_chunked':details.get('native_chunked',''),'native_synthesis_calls':details.get('native_synthesis_calls',''),
                'chunk_characters':json.dumps(details.get('chunk_characters',[])),
                'chunk_words':json.dumps(details.get('chunk_words',[]))}
            review=reviews.get((directory.name,r['sample_id']),{})
            row.update({f:review.get(f,'') for f in REVIEW})
            row['reviewed']=row['intelligible'] in ('pass','fail') and row['faithful'] in ('pass','fail') and row['naturalness_1_to_5'] in ('1','2','3','4','5')
            row['quality_pass']=row['reviewed'] and row['intelligible']=='pass' and row['faithful']=='pass' and int(row['naturalness_1_to_5'])>=3
            rr.append(row)
            reviews[(directory.name,r['sample_id'])]={k:row.get(k,'') for k in ['run_id','sample_id','model_key','text','audio_file']+REVIEW}
        rows+=rr;good=[r for r in rr if r['status']=='success'];valid=[r for r in good if r['quality_pass']]
        summary={**base,'saved':len(good),'expected':len(d['samples']),'complete':len(good)==len(d['samples']),'reviewed':sum(r['reviewed'] for r in good),'quality_pass':len(valid),'failed':sum(r['status']=='failed' for r in rr),'session_errors':len(list((directory/'session_errors').glob('*.json')))}
        for label,items in [('all_completed',good),('quality_passed',valid)]:
            if items:
                summary[label+'_generation_p50_s']=statistics.median(r['generation_s'] for r in items)
                summary[label+'_generation_max_s']=max(r['generation_s'] for r in items)
                summary[label+'_rtf_p50']=statistics.median(r['rtf'] for r in items)
                summary[label+'_under_2_s']=sum(r['generation_s']<2 for r in items)
        for field in ('preparation_s','synthesis_s'):
            values=[r[field] for r in good if field in r]
            if values:summary[field+'_p50']=statistics.median(values)
        summaries.append(summary)
    fields=list(dict.fromkeys(['run_id','sample_id','model_key','status','text','audio_file']+[k for r in rows for k in r]))
    csv_write(dest/'clips.csv',rows,fields)
    csv_write(dest/'comparison.csv',summaries,list(dict.fromkeys(['model_key','saved','expected']+[k for r in summaries for k in r])))
    csv_write(review_path,list(reviews.values()),['run_id','sample_id','model_key','text','audio_file']+REVIEW)
    write_json(dest/'summary.json',summaries)
    lines=['# TTS listening and timing report','','Separate short (six texts) and long (three texts) suites, one pass. Compare complete runs in the same comparison group and suite; voice modes and diffusion steps remain explicit.',
        'Success = finite non-silent waveform saved. Listening determines intelligibility, faithfulness and naturalness.',
        'Quality pass = intelligible and faithful both pass, naturalness >=3/5. This is one listener judgment, not a MOS study.',
        'Generation time includes text preparation, model synthesis/postprocessing and CPU waveform availability. Load, warmup, WAV writes and browser playback are excluded.',
        'Preparation and synthesis timings are available in clips.csv. Synthesis includes codec/postprocessing/dispatch. OmniVoice native chunking remains enabled; it does not stream audio to the player.',
        'No first-audio or end-to-end latency claim. No automatic ASR CER is calculated. The under-2-second goal is aspirational.',
        'MMS weights: CC-BY-NC-4.0, evaluation comparison only; not the default organizational deployment choice.','',
        '|Model|Suite|Device|Steps|Saved/expected|Reviewed/pass|Total median s|Preparation median s|Synthesis median s|',
        '|---|---|---|---|---|---|---|---|---|']
    for s in summaries:
        fmt=lambda k:f'{s[k]:.3f}' if k in s else '—'
        lines.append(f"|{s['model_key']}|{s['suite']}|{s['device']}|{s['steps']}|{s['saved']}/{s['expected']}|{s['reviewed']}/{s['quality_pass']}|{fmt('all_completed_generation_p50_s')}|{fmt('preparation_s_p50')}|{fmt('synthesis_s_p50')}|")
    lines+=['','## Long texts: inspect each length separately','',
        '|Model|Steps|Sample|Words|Total s|Preparation s|Synthesis s|RTF|Quality|',
        '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        if r['suite']!='long' or r['status']!='success':continue
        fmt=lambda k:f'{r[k]:.3f}' if k in r else '—'
        quality=('pass' if r['quality_pass'] else 'fail') if r['reviewed'] else 'unrated'
        lines.append(f"|{r['model_key']}|{r['steps']}|{r['sample_id']}|{r.get('input_words','—')}|{fmt('generation_s')}|{fmt('preparation_s')}|{fmt('synthesis_s')}|{fmt('rtf')}|{quality}|")
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preflight','run','report','export'])
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--output',type=Path)
    p.add_argument('--owner',default='abel');p.add_argument('--model',choices=['omni_amharic','omni_base','mms_amh'],default='omni_amharic')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--steps',type=int,choices=[16,32],default=32)
    p.add_argument('--suite',choices=['short','long'],default='short')
    p.add_argument('--limit',type=int,choices=[0,1,2],default=2);p.add_argument('--reviewed-run')
    p.add_argument('--reference-wav',type=Path);p.add_argument('--reference-text',default='')
    a=p.parse_args()
    if a.action=='preflight':
        from .tts_backend import preflight
        preflight(a.repo,a.model,a.suite);return
    if a.output is None:p.error('--output required')
    if bool(a.reference_wav)!=bool(a.reference_text.strip()):p.error('Provide both reference WAV and its exact transcript, or neither.')
    try:
        if a.action=='run':run(a)
    finally:report(a.output,a.owner)
    if a.action=='export':
        path=a.output/f'tts-v3-results-{a.owner}-{now().replace(":","").replace(".","-")}-{uuid4().hex[:6]}.zip'
        print('Result ZIP:',export_run(a.output/a.owner,path))

if __name__=='__main__':main()

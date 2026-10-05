"""Three reviewed answers, one complete OmniVoice call per sentence."""
import argparse
import csv
from pathlib import Path
import time
import traceback
from uuid import uuid4

from .artifacts import (read_json,write_json,digest,object_digest,source_snapshot,
                        environment,safe_name,now,export_run)
from .tts_benchmark import validate_waveform

PROTOCOL={'version':'tts-accepted-sentences-v1','steps':32,'precision':'float32',
          'guidance_scale':2.0,'seed':42,'voice':'auto','reference':None,
          'external_chunking':False,'punctuation':'retained verbatim',
          'normalize_text':False,'planned_repeats':1,
          'timing':'synchronized generate invocation through complete CPU waveform availability; excludes load, warmup, WAV write and playback',
          'native_chunking':'observed and reported; native implementation otherwise unchanged'}
RATINGS=('intelligible','faithful','complete_ending')


def validate_dataset(data):
    seen=set()
    if len(data['samples'])!=3:raise ValueError('Expected the three accepted sentences.')
    for s in data['samples']:
        safe_name(s['id'])
        if s['id'] in seen or not s['text'].strip():raise ValueError('Duplicate ID or empty text.')
        if s['provenance']['human_decision']!='pass':raise ValueError('Use reviewed accepted text.')
        seen.add(s['id'])


def prepare(repo,output,owner,env=None):
    data=read_json(repo/'data/tts_public/accepted_sentences.json');validate_dataset(data)
    cat=read_json(repo/'configs/tts_candidates.json')
    config={'dataset':data,'candidate':cat['candidates']['omni_amharic'],'codec':cat['codec'],
            'protocol':PROTOCOL,'environment':environment() if env is None else env,'source':source_snapshot(repo)}
    name='tts-sentences-'+object_digest(config)[:20]
    root=output/safe_name(owner)/'runs'/name
    if not (root/'run.json').exists():write_json(root/'run.json',{'run_id':name,'created_at':now(),'owner':owner,**config})
    write_json(output/owner/'active.json',{'run_id':name})
    print('RUN_ID:',name,flush=True)
    return root,config


def verify_audio(root,r):
    if r.get('audio_file'):
        if digest(root/r['audio_file'])!=r['audio_sha256']:raise ValueError('Saved audio checksum changed.')


def generate(root,config,backend_class=None):
    import soundfile as sf
    todo=[]
    for sample in config['dataset']['samples']:
        p=root/'results'/f'{sample["id"]}.json'
        if p.exists() and read_json(p)['status']=='success':verify_audio(root,read_json(p))
        else:todo.append(sample)
    if not todo:print('All three WAVs already saved; no model loaded.',flush=True);return
    if backend_class is None:
        from .tts_backend import LocalTTS
        backend_class=LocalTTS
    session=uuid4().hex;stage='loading'
    try:
        start=time.perf_counter()
        backend=backend_class(config['candidate'],config['codec'],device='cuda',steps=32,reference=None)
        backend.sync();loaded=time.perf_counter()
        stage='warmup';backend.seed(41)
        wave,sr,_=backend.generate('ሰላም፣ እንዴት ነዎት?');backend.sync();validate_waveform(wave,sr)
        write_json(root/'sessions'/f'{session}.json',{'load_s':loaded-start,
            'warmup_s':time.perf_counter()-loaded,'settings':backend.settings,'loading_info':backend.loading_info})
        stage='sentences'
        for sample in todo:
            r={'sample_id':sample['id'],'text':sample['text'],'expected':sample['expected'],
               'status':'running','session':session}
            try:
                backend.seed(PROTOCOL['seed']);backend.sync()
                start=time.perf_counter()
                # One full sentence, verbatim. Never split, trim punctuation, crop or concatenate audio.
                wave,sr,details=backend.generate(sample['text']);backend.sync()
                elapsed=time.perf_counter()-start
                wave,metrics=validate_waveform(wave,sr)
                relative='audio/'+sample['id']+'.wav';path=root/relative
                path.parent.mkdir(exist_ok=True)
                pending=path.with_name('.pending-'+uuid4().hex+'.wav')
                sf.write(pending,wave,sr,subtype='FLOAT');pending.replace(path)
                r.update(status='success',metrics={**metrics,'generation_s':elapsed,
                    'rtf':elapsed/metrics['audio_duration_s'],'input_words':len(sample['text'].split())},
                    details=details,audio_file=relative,audio_sha256=digest(path))
                print(sample['id'],'| generation',round(elapsed,3),'s | audio',round(metrics['audio_duration_s'],3),
                      's | native chunked:',details.get('native_chunked'),flush=True)
            except BaseException as exc:
                r.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
            finally:write_json(root/'results'/f'{sample["id"]}.json',r)
    except BaseException as exc:
        write_json(root/'failures'/f'{session}.json',{'stage':stage,'error':str(exc),'traceback':traceback.format_exc()})
        raise


def accepted(record,review):
    return (record['status']=='success' and all(review.get(k) is True for k in RATINGS)
            and type(review.get('naturalness')) is int and 3<=review['naturalness']<=5)


def report(output,owner):
    base=output/safe_name(owner)
    root=base/'runs'/safe_name(read_json(base/'active.json')['run_id'])
    rows=[]
    for path in sorted((root/'results').glob('*.json')):
        r=read_json(path);verify_audio(root,r)
        p=root/'reviews'/path.name;v=read_json(p) if p.exists() else {}
        if v.get('result_sha256')!=digest(path):v={}
        m=r.get('metrics',{});d=r.get('details',{})
        ok=accepted(r,v)
        rows.append({'sample_id':r['sample_id'],'text':r['text'],'status':r['status'],
            'generation_s':m.get('generation_s'),'audio_duration_s':m.get('audio_duration_s'),
            'rtf':m.get('rtf'),'preparation_s':d.get('preparation_s'),
            'native_chunked':d.get('native_chunked'),'native_synthesis_calls':d.get('native_synthesis_calls'),
            'reviewed':all(type(v.get(k)) is bool for k in RATINGS) and type(v.get('naturalness')) is int,
            'quality_pass':ok,'quality_pass_under_2s':bool(ok and m.get('generation_s',999)<2),
            **{k:v.get(k) for k in RATINGS},'naturalness':v.get('naturalness'),'notes':v.get('notes',''),
            'error':r.get('error','')})
    dest=base/'reports';dest.mkdir(parents=True,exist_ok=True)
    with (dest/'clips.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['status']);writer.writeheader();writer.writerows(rows)
    write_json(dest/'clips.json',rows)
    text=['# OmniVoice: three accepted sentences','',
          'One complete sentence per public generate call, 32 steps, FP32, automatic voice, no reference.',
          'Punctuation retained; no external chunking, waveform cropping or normalization. Native internal chunking is logged.',
          'Time measures complete waveform availability. No pipeline or browser latency claim. One pass per sentence.',
          'Quality requires intelligible, faithful speech, a complete ending and naturalness at least 3/5.', '',
          '|Sentence|Status|Generation s|Audio s|Quality|Native chunking|','|---|---|---|---|---|---|']
    for r in rows:
        quality=('pass' if r['quality_pass'] else 'fail') if r['reviewed'] else 'unreviewed'
        text.append(f'|{r["sample_id"]}|{r["status"]}|{r["generation_s"]}|{r["audio_duration_s"]}|{quality}|{r["native_chunked"]}|')
    (dest/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('\n'.join(text),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','export'])
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--owner',default='abel')
    a=p.parse_args();safe_name(a.owner)
    if a.action=='run':
        root,config=prepare(a.repo,a.output,a.owner)
        try:generate(root,config)
        finally:report(a.output,a.owner)
    else:
        report(a.output,a.owner)
        print('RESULT ZIP:',export_run(a.output/a.owner,a.output/f'tts-sentences-{a.owner}-{uuid4().hex[:10]}.zip'),flush=True)

if __name__=='__main__':main()

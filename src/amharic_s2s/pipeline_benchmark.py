"""Recorded-question integration: selected models resident; sequential vs streamed clauses."""
import argparse
import csv
import json
from pathlib import Path
import queue
import shutil
import threading
import time
import traceback
from uuid import uuid4

from .artifacts import (read_json, write_json, digest, object_digest, environment,
                        source_snapshot, safe_name, now, export_run)
from .pipeline_core import VERSION, MODES, ClauseBuffer, quality_flags, metrics, accepted


def read_audio(path):
    import numpy as np
    import soundfile as sf
    audio,sr = sf.read(path,dtype='float32',always_2d=True)
    if not 1 <= len(audio)/sr <= 30: raise ValueError(f'{path.name}: use 1–30 seconds of speech.')
    if not np.isfinite(audio).all(): raise ValueError(f'{path.name}: invalid audio.')
    audio = audio.mean(axis=1)
    if np.max(np.abs(audio)) < 1e-5: raise ValueError(f'{path.name}: silent recording.')
    return audio,sr


def inputs(repo, input_dir, dataset_file=None):
    data = read_json(dataset_file or repo/'data/pipeline_public/questions.json')
    for s in data['samples']:
        p = input_dir/s['audio_file']
        if not p.is_file(): raise FileNotFoundError(f'Record or upload {p}')
        audio,sr = read_audio(p)
        s.update(audio_sha256=digest(p),duration_s=len(audio)/sr,sampling_rate=sr)
    return data


def prepare(repo, output, owner, input_dir, dataset_file=None, version=VERSION, settings_extra=None):
    data = inputs(repo,input_dir,dataset_file)
    selected = {
        'asr':read_json(repo/'configs/asr_candidates.json')['candidates']['hohe'],
        'llm':read_json(repo/'configs/llm_next_candidates.json')['candidates']['gemma4_e4b'],
        'tts':read_json(repo/'configs/tts_candidates.json')['candidates']['omni_amharic'],
        'codec':read_json(repo/'configs/tts_candidates.json')['codec'],
        'settings':{'asr':'float16','llm':'nf4 / FP16 compute','tts':'float32 / 32 steps / auto voice',
                    'seed':42,'max_new_tokens':128,'max_words_per_clause':24},
    }
    selected['settings'].update(settings_extra or {})
    identity = {'version':version,'dataset':data,'models':selected,
                'source':source_snapshot(repo),'environment':environment()}
    run_id = 'pipeline-'+object_digest(identity)[:20]
    root = output/safe_name(owner)/'runs'/run_id
    root.mkdir(parents=True,exist_ok=True)
    if not (root/'run.json').exists():
        write_json(root/'run.json',{'run_id':run_id,'created_at':now(),'owner':owner,**identity})
        (root/'inputs').mkdir(exist_ok=True)
        for s in data['samples']: shutil.copyfile(input_dir/s['audio_file'],root/'inputs'/s['audio_file'])
    for s in data['samples']:
        if digest(root/'inputs'/s['audio_file']) != s['audio_sha256']:
            raise RuntimeError('Saved input changed; restore original recording before resuming.')
    write_json(output/owner/'latest.json',{'run_id':run_id})
    print('RUN_ID:',run_id,flush=True)
    return root,data


def save_audio(root, relative, wav, sr):
    import numpy as np
    import soundfile as sf
    wav = np.asarray(wav,dtype=np.float32).squeeze()
    if wav.ndim != 1 or not len(wav) or not np.isfinite(wav).all() or np.max(np.abs(wav)) < 1e-7:
        raise ValueError('Invalid, empty or silent TTS waveform.')
    path = root/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    sf.write(path,wav,sr,subtype='FLOAT')
    return wav,digest(path)


def run_turn(models, sample, mode, root, session, queue_timeout_s=180):
    import numpy as np
    from .llm_backend import prompt_messages
    attempt = uuid4().hex[:10]
    key = sample['id']+'__'+mode
    record = {'sample_id':sample['id'],'mode':mode,'session':session,'attempt':attempt,
              'status':'running','events':{},'chunks':[],'quality_flags':[],
              'reference':sample['reference'],'context':sample['context'],'expected':sample['expected']}
    start = time.perf_counter()
    event_lock = threading.RLock()
    def mark(name):
        with event_lock: record['events'][name] = time.perf_counter()-start
    audio,sr = read_audio(root/'inputs'/sample['audio_file'])
    pending_path = root/'attempts'/f'{key}-{attempt}.json'
    cancel, q, worker = threading.Event(),queue.Queue(),None
    waves = []
    def synthesize(text, boundary):
        if cancel.is_set(): raise RuntimeError('Generation cancelled.')
        flags = quality_flags(text)
        if flags: raise ValueError(f'Speech text failed basic checks: {flags}')
        i = len(record['chunks'])
        tts_start = time.perf_counter()-start
        wav,rate,details = models.tts.generate(text.strip())
        ready = time.perf_counter()-start
        relative = f'audio/{key}-{attempt}/chunk_{i+1:02d}.wav'
        wav,sha = save_audio(root,relative,wav,rate)
        chunk = {'index':i+1,'text':text,'boundary':boundary,'tts_start':tts_start,
                 'audio_ready':ready,'audio_duration_s':len(wav)/rate,'sampling_rate':rate,
                 'path':relative,'sha256':sha,'details':details,
                 'saved_at_s':time.perf_counter()-start}
        record['chunks'].append(chunk)
        waves.append((wav,rate))
        with event_lock: write_json(pending_path,record)
        print('AUDIO_READY '+json.dumps({'path':str(root/relative),'sample_id':sample['id'],
                                         'mode':mode,'chunk':i+1},ensure_ascii=False),flush=True)
    try:
        models.torch.cuda.synchronize()
        models.torch.cuda.reset_peak_memory_stats()
        models.tts.seed(42)
        transcript,raw = models.transcribe(audio,sr,mark)
        record.update(transcript=transcript,raw_transcript=raw)
        print(f'ASR [{sample["id"]}/{mode}]: {transcript}',flush=True)
        messages = prompt_messages({'instruction':sample['instruction']},{**sample,'question':transcript})
        def produce():
            try:
                result = models.generate_text(messages,lambda text:q.put(('text',text)),cancel,mark)
                q.put(('done',result))
            except BaseException as exc:
                q.put(('error',f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'))
        worker = threading.Thread(target=produce,daemon=True)
        worker.start()
        buffer = ClauseBuffer()
        streamed_text = ''
        while True:
            kind,value = q.get(timeout=queue_timeout_s)
            if kind == 'error': raise RuntimeError(value)
            if kind == 'done':
                record.update(value)
                break
            streamed_text += value
            if mode == 'clauses':
                for text,boundary in buffer.feed(value): synthesize(text,boundary)
        worker.join()
        record['quality_flags'] = quality_flags(record['answer'])
        if streamed_text.strip() != record['answer'].strip():
            raise RuntimeError('Streamed text differs from final decoded answer; inspect tokenizer streaming.')
        if record['stop_reason'] != 'eos':
            raise RuntimeError(f'Incomplete LLM answer: {record["stop_reason"]}. Partial audio is not accepted.')
        if record['quality_flags']:
            raise RuntimeError(f'Answer failed basic checks: {record["quality_flags"]}')
        if mode == 'sequential': synthesize(record['answer'],'full_answer')
        else:
            for text,boundary in buffer.feed(final=True): synthesize(text,boundary)
        if not waves: raise RuntimeError('No audio produced.')
        if len({sr for _,sr in waves}) != 1: raise ValueError('Chunk sample rates differ.')
        relative = f'audio/{key}-{attempt}/combined.wav'
        _,sha = save_audio(root,relative,np.concatenate([w for w,_ in waves]),waves[0][1])
        record.update(combined_audio=relative,combined_sha256=sha,status='success')
        mark('complete')
        record['peak_allocated_gib'] = models.torch.cuda.max_memory_allocated()/2**30
        record['peak_reserved_gib'] = models.torch.cuda.max_memory_reserved()/2**30
    except BaseException as exc:
        cancel.set()
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        cancel.set()
        if worker and worker.is_alive(): worker.join(timeout=15)
        record['worker_still_running'] = bool(worker and worker.is_alive())
        with event_lock:
            record['metrics'] = metrics(record)
            write_json(pending_path,record)
            write_json(root/'results'/f'{key}.json',record)
        print('TURN_RESULT '+json.dumps({'sample_id':sample['id'],'mode':mode,
                  'status':record['status'],'metrics':record['metrics']},ensure_ascii=False),flush=True)


def verify_result(root, r):
    for c in r.get('chunks',[]):
        if digest(root/c['path']) != c['sha256']: raise RuntimeError('Saved audio checksum mismatch.')
    if r.get('combined_audio') and digest(root/r['combined_audio']) != r['combined_sha256']:
        raise RuntimeError('Saved combined audio checksum mismatch.')


def review_for(root,path):
    p = root/'reviews'/(path.stem+'.json')
    review = read_json(p) if p.exists() else {}
    return review if review.get('result_sha256') == digest(path) else {}


def run(repo, output, owner, input_dir, limit, reviewed):
    root,data = prepare(repo,output,owner,input_dir)
    if limit == 3:
        if reviewed != root.name: raise ValueError('First review the one-question run and provide its exact RUN_ID.')
        for mode in MODES:
            p = root/'results'/f'everyday__{mode}.json'
            if not p.exists() or not review_for(root,p).get('reviewed'):
                raise ValueError('Save listening reviews for both everyday modes first. A failure can be reviewed too.')
    todo = []
    for sample in data['samples'][:limit]:
        for mode in MODES:
            p = root/'results'/f'{sample["id"]}__{mode}.json'
            if p.exists() and read_json(p)['status']=='success':
                verify_result(root,read_json(p)); print('Reusing:',p.name,flush=True)
            else: todo.append((sample,mode))
    if not todo: print('Selected turns already saved.'); return
    import torch
    session = uuid4().hex[:12]
    capacity_rows = []
    def capacity(stage):
        torch.cuda.synchronize()
        free,total = torch.cuda.mem_get_info()
        row = {'stage':stage,'free_gib':free/2**30,'total_gib':total/2**30,
               'allocated_gib':torch.cuda.memory_allocated()/2**30,
               'reserved_gib':torch.cuda.memory_reserved()/2**30,'time':now()}
        capacity_rows.append(row)
        write_json(root/'sessions'/f'{session}.json',{'capacity':capacity_rows})
        print('CAPACITY '+json.dumps(row),flush=True)
    try:
        from .pipeline_backend import PipelineModels
        models = PipelineModels(repo,capacity)
        audio,sr = read_audio(root/'inputs'/data['samples'][0]['audio_file'])
        models.warmup(audio,sr,capacity)
        for sample,mode in todo:
            print('Running:',sample['id'],mode,flush=True)
            try:
                run_turn(models,sample,mode,root,session)
            except Exception as exc:
                saved = read_json(root/'results'/f'{sample["id"]}__{mode}.json')
                if isinstance(exc,torch.OutOfMemoryError) or 'CUDA' in str(exc) or saved.get('worker_still_running'):
                    raise
                print('Turn failed; saved for review:',exc,flush=True)
    except BaseException as exc:
        write_json(root/'sessions'/f'{session}-error.json',{'error':str(exc),'traceback':traceback.format_exc(),
                                                         'capacity':capacity_rows,'time':now()})
        raise


def report(output, owner):
    base = output/owner
    latest = read_json(base/'latest.json')['run_id']
    root = base/'runs'/safe_name(latest)
    rows = []
    for path in sorted((root/'results').glob('*.json')):
        r = read_json(path)
        verify_result(root,r)
        review = review_for(root,path)
        ok = accepted(r,review)
        m = metrics(r)
        rows.append({'run_id':latest,'sample_id':r['sample_id'],'mode':r['mode'],'status':r['status'],
                     'reviewed':bool(review.get('reviewed')),'quality_accepted':ok,
                     'accepted_target_under_3s': bool(ok and m.get('candidate_tts_start_from_asr_s',999)<3),
                     **m,'transcript':r.get('transcript',''),'answer':r.get('answer',''),
                     'stop_reason':r.get('stop_reason',''),'error':r.get('error','')})
    dest = base/'reports'; dest.mkdir(exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r)) or ['run_id','status']
    with (dest/'pipeline.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write_json(dest/'latest_results.json',{'run_id':latest,'results':rows})
    (dest/'README.txt').write_text(
        'One pass per question/mode. Only compare these paired integration results.\n'
        'Target: ASR inference invocation to first TTS generate invocation, after text checks.\n'
        'Candidate timing is not acceptance: review ASR, answer, meaningful first chunk, audio faithfulness and naturalness.\n'
        'First audio is a complete clause waveform ready in the server process, not browser playback.\n'
        'Playback gaps are simulated from waveform readiness/durations; no transport or player measured.\n'
        'Models stay resident. Loading/warmup excluded. GPU contention is included in wall times.\n'
        'Chunking can affect pronunciation, omissions, repetition and voice continuity. Listen to every chunk.\n'
        'No VAD, speculative ASR/LLM, retrieval, production safety guarantee or live microphone latency claim.\n')
    print('Reports:',dest,flush=True)
    for row in rows: print(row['sample_id'],row['mode'],row['status'],
                           'TTS start:',row.get('candidate_tts_start_from_asr_s'),
                           'audio ready:',row.get('first_audio_ready_from_asr_s'),
                           'accepted:',row['quality_accepted'])
    return root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check-inputs','run','report','export'])
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--owner',default='abel')
    p.add_argument('--inputs',type=Path)
    p.add_argument('--sample-limit',type=int,choices=[1,3],default=1)
    p.add_argument('--reviewed-run')
    a=p.parse_args();safe_name(a.owner)
    if a.action in ('run','check-inputs') and a.inputs is None: p.error('--inputs is required')
    if a.action=='check-inputs':
        for s in inputs(a.repo,a.inputs)['samples']: print(s['id'],round(s['duration_s'],2),'seconds',s['reference'])
    elif a.action=='run':
        try: run(a.repo,a.output,a.owner,a.inputs,a.sample_limit,a.reviewed_run)
        finally:
            if (a.output/a.owner/'latest.json').exists(): report(a.output,a.owner)
    else:
        root=report(a.output,a.owner)
        if a.action=='export':
            stamp=now().replace(':','').replace('.','-')
            print('RESULT ZIP:',export_run(a.output/a.owner,a.output/f'pipeline-results-{a.owner}-{stamp}.zip'))

if __name__=='__main__': main()

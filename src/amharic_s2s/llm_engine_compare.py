"""Gemma deployment-configuration screen: compiled NF4 versus llama.cpp Q4_K_M.

Separate subprocesses own GPU memory. Both receive identical pretokenized prompts.
No ASR, retrieval, TTS, production changes or automatic winner selection.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import shutil
import socket
import statistics
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request
from uuid import uuid4

from .artifacts import read_json, write_json, digest, object_digest, environment, now
from .llm_backend import prompt_messages, diagnostics

ENGINES=('transformers_compiled','llama_cpp')


def fetch(url,path,sha):
    if path.exists() and digest(path)==sha:return
    part=path.with_suffix(path.suffix+'.part')
    print('Downloading',path.name,flush=True)
    request=urllib.request.Request(url,headers={'User-Agent':'amharic-s2s-engine-screen'})
    with urllib.request.urlopen(request,timeout=120) as response, part.open('wb') as out:
        total=int(response.headers.get('Content-Length',0));done=0;progress=-1
        while block:=response.read(4*1024*1024):
            out.write(block);done+=len(block)
            if total and done*10//total!=progress:
                progress=done*10//total;print(path.name, min(100,done*100//total),'%',flush=True)
    if digest(part)!=sha:raise RuntimeError('Download checksum mismatch: '+path.name)
    part.replace(path)


def tool_environment(directory):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
    dirs=sorted({str(p.parent) for p in directory.rglob('*.so*')})
    env['LD_LIBRARY_PATH']=':'.join(dirs+[env.get('LD_LIBRARY_PATH','')])
    return env


def setup(repo,work):
    cfg=read_json(repo/'configs/llm_engine_comparison.json')
    libc,version=platform.libc_ver()
    if platform.machine()!='x86_64' or libc!='glibc' or tuple(map(int,version.split('.')[:2]))<(2,39):
        raise RuntimeError('The pinned Ubuntu 24.04 binary needs x86_64 and glibc >= 2.39. Your saved Colab runs used glibc 2.39. Use the current Colab runtime; no download/build was started.')
    work.mkdir(parents=True,exist_ok=True)
    destination=work/cfg['llama_release']
    marker=destination/'verified.json'
    if marker.exists() and read_json(marker).get('config_sha256')==object_digest(cfg) and Path(read_json(marker)['server']).is_file():
        print('Reusing verified llama.cpp installation.',flush=True);return
    if shutil.disk_usage(work).free<30*2**30:
        raise RuntimeError('Need at least 30 GiB free before setup/model downloads. Use a fresh runtime; no files were deleted.')
    destination.mkdir(exist_ok=True)
    for asset in cfg['assets']:
        archive=work/asset['name'];fetch(asset['url'],archive,asset['sha256'])
        with tarfile.open(archive,'r:gz') as tar:
            # Python 3.12+ data filter rejects traversal, devices and escaping links.
            tar.extractall(destination,filter='data')
    binaries=list(destination.rglob('llama-server'))
    if len(binaries)!=1:raise RuntimeError('Expected one llama-server binary; inspect extracted release.')
    server=binaries[0];env=tool_environment(destination)
    check=subprocess.run([str(server),'--list-devices'],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
    print(check.stdout,flush=True)
    if check.returncode or not re.search(r'CUDA\d+:',check.stdout):
        raise RuntimeError('Pinned CUDA binary did not detect a GPU. Export the setup log; no CPU fallback or source build was attempted.')
    version=subprocess.check_output([str(server),'--version'],env=env,text=True,stderr=subprocess.STDOUT,timeout=30)
    write_json(marker,{'server':str(server),'version':version,'devices':check.stdout,'config_sha256':object_digest(cfg)})
    print('Pinned CUDA server ready; no model loaded.',flush=True)


def prepare_prompts(repo):
    from .llm_next import imports
    tf=imports()
    candidate=read_json(repo/'configs/llm_next_candidates.json')['candidates']['gemma4_e4b']
    tokenizer=tf.AutoTokenizer.from_pretrained(candidate['model_id'],revision=candidate['revision'])
    instructions=read_json(repo/'configs/runtime_instructions.json')
    data=read_json(repo/'configs/llm_engine_cases.json')
    samples=[]
    for sample in data['samples']:
        instruction=instructions['grounded' if sample['context'] else 'direct']
        messages=prompt_messages({'instruction':instruction},sample)
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer(text,add_special_tokens=False)['input_ids']
        if len(ids)+128>512:raise ValueError(sample['id']+' exceeds the existing 384-token prompt budget; no truncation.')
        samples.append({**sample,'messages':messages,'rendered_prompt':text,'input_ids':ids,'input_tokens':len(ids),
                        'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()})
    return {'samples':samples,'instructions':instructions,'dataset':data,'candidate':candidate}


class MemorySampler:
    """Same sampled nvidia-smi process-memory definition for both engines."""
    def __init__(self,pid):self.pid=pid;self.peak=None;self.stop=threading.Event()
    def sample(self):
        try:
            text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,stderr=subprocess.DEVNULL,timeout=4)
            for line in text.splitlines():
                pid,mem=[x.strip() for x in line.split(',')]
                if int(pid)==self.pid:self.peak=max(self.peak or 0,int(mem))
        except (OSError,ValueError,subprocess.SubprocessError):pass
    def _loop(self):
        while not self.stop.wait(.5):self.sample()
    def __enter__(self):
        self.sample();self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join(timeout=5);self.sample()


def record(root,engine,case,repeat,result,warm=False):
    answer=result.get('answer','')
    flags=diagnostics(answer)
    ok=result.get('stop_reason')=='eos' and bool(answer.strip())
    row={'engine':engine,'case_id':case['id'],'repeat':repeat,'warmup':warm,
         'status':'success' if ok else 'incomplete','prompt_sha256':case['prompt_sha256'],
         'input_tokens':case['input_tokens'],'diagnostics':flags,**result}
    write_json(root/engine/('warmup' if warm else 'results')/f'{case["id"]}__{repeat}.json',row)
    return row


def run_hf(root):
    from .llm_next import NextLLM, NEXT_PROTOCOL
    from .llm_compile import DecodeCompiler, policy, graphs, deadline
    import torch
    import torch._dynamo
    prompts=read_json(root/'prompts.json');status={'status':'loading','started_at':now()}
    write_json(root/ENGINES[0]/'session.json',status)
    torch._dynamo.config.suppress_errors=False
    started=time.perf_counter();backend=NextLLM(prompts['candidate'],'nf4',NEXT_PROTOCOL)
    status['download_and_load_s']=time.perf_counter()-started
    status['loading_info']=backend.loading_info
    def generate(case,dispatch,warm):
        torch.manual_seed(42);torch.cuda.synchronize()
        before=graphs();old=dispatch.calls;seen=[];pieces=[];first=None
        class Stream(backend.TextStreamer):
            def put(self,value):
                nonlocal first
                if not (self.skip_prompt and self.next_tokens_are_prompt):
                    if first is None:first=time.perf_counter()
                    seen.extend(value.reshape(-1).tolist())
                super().put(value)
            def on_finalized_text(self,text,stream_end=False):pieces.append(text)
        with policy(backend,'compiled',warm=warm):
            t=time.perf_counter()
            ids=torch.tensor([case['input_ids']],device='cuda',dtype=torch.long)
            with torch.inference_mode():
                output=backend.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                    generation_config=backend.generation,logits_processor=backend.score_checks,
                    streamer=Stream(backend.tokenizer,skip_prompt=True,skip_special_tokens=True))
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
        tokens=output[0,len(case['input_ids']):].tolist()
        answer=backend.tokenizer.decode(tokens,skip_special_tokens=True).strip()
        if seen!=tokens or ''.join(pieces).strip()!=answer:raise RuntimeError('HF stream/output mismatch')
        created=graphs()-before;calls=dispatch.calls-old
        if len(tokens)>1 and calls<=0:raise RuntimeError('Compiled decode path was not used.')
        result={'answer':answer,'stop_reason':'eos' if tokens and tokens[-1] in prompts['candidate']['eos_token_ids'] else 'limit_or_timeout',
                'full_answer_s':elapsed,'first_observed_token_s':first-t if first is not None else None,
                'engine_generated_tokens':len(tokens),'text_tokens':len(backend.tokenizer(answer,add_special_tokens=False)['input_ids']),
                'generated_token_ids':tokens,'compiled_decode_calls':calls,'new_graphs':created,
                'timing_status':'warmup' if warm else 'includes_compilation' if created else 'warmed'}
        if created and not warm:result['status']='invalid_timing'
        return result
    with MemorySampler(os.getpid()) as memory, DecodeCompiler(backend).installed() as dispatch:
        dispatch.variant='compiled';start_graphs=graphs();t=time.perf_counter()
        with deadline(480):
            for repeat in (1,2):
                for case in prompts['samples']:
                    print('HF warmup',repeat,case['id'],flush=True)
                    record(root,ENGINES[0],case,repeat,generate(case,dispatch,True),True)
        if graphs()<=start_graphs or dispatch.calls<=0:raise RuntimeError('Compilation warmup not verified')
        status['warmup_s']=time.perf_counter()-t;write_json(root/ENGINES[0]/'session.json',status)
        for repeat in (1,2,3):
            cases=prompts['samples'];cases=cases[repeat-1:]+cases[:repeat-1]
            for case in cases:
                print('HF measured',repeat,case['id'],flush=True)
                r=record(root,ENGINES[0],case,repeat,generate(case,dispatch,False))
                if r['status']=='invalid_timing':raise RuntimeError('New graph during measurement; saved invalid timing and stopped this engine.')
    status.update(status='complete',process_vram_peak_mib_sampled=memory.peak)
    write_json(root/ENGINES[0]/'session.json',status)


class LocalServer:
    def __init__(self,port):
        self.base=f'http://127.0.0.1:{port}'
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(self,path,data=None,timeout=120):
        req=urllib.request.Request(self.base+path,data=None if data is None else json.dumps(data,ensure_ascii=False).encode(),
            headers={'Content-Type':'application/json'})
        return self.opener.open(req,timeout=timeout)
    def json(self,path,data=None,timeout=120):
        with self.request(path,data,timeout) as response:return json.load(response)


def completion_payload(case):
    return {'prompt':case['input_ids'],'n_predict':128,'temperature':0,'top_k':1,'top_p':1,'min_p':0,
            'repeat_penalty':1,'presence_penalty':0,'frequency_penalty':0,'seed':42,
            'cache_prompt':False,'stream':True,'return_tokens':True,'id_slot':0}


def parse_sse(lines,started,clock=time.perf_counter):
    parts=[];tokens=[];first=None;final=None
    for raw in lines:
        line=raw.decode('utf-8').strip() if isinstance(raw,bytes) else raw.strip()
        if not line.startswith('data:'):continue
        body=line[5:].strip()
        if body=='[DONE]':break
        item=json.loads(body)
        if 'error' in item:raise RuntimeError('llama.cpp stream error: '+str(item['error']))
        if item.get('tokens'):
            if first is None:first=clock()-started
            tokens.extend(item['tokens'])
        parts.append(item.get('content',''))
        if item.get('stop'):
            final=item;break
    if final is None:raise RuntimeError('Stream ended without a terminal completion record')
    if final.get('truncated'):raise RuntimeError('llama.cpp truncated the prompt/context')
    return {'answer':''.join(parts).strip(),'stop_reason':final.get('stop_type','unknown'),
            'full_answer_s':clock()-started,'first_observed_token_s':first,
            'engine_generated_tokens':final.get('tokens_predicted',final.get('timings',{}).get('predicted_n')),
            'generated_token_ids':tokens,'server_timings':final.get('timings'),
            'tokens_evaluated':final.get('tokens_evaluated'),'generation_settings':final.get('generation_settings'),
            'tokens_cached':final.get('tokens_cached'),
            'timing_status':'warmed','new_graphs':None}


def stop_server(process):
    if process.poll() is None:
        process.terminate()
        try:process.wait(timeout=20)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)


def verify_layer_offload(log):
    """Require placement evidence; server readiness alone does not prove GPU use."""
    counts=re.findall(r'offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU',log)
    if not counts:
        raise RuntimeError('Full layer offload was not verified: no offload summary in server.log. '
                           'Pinned llama.cpp needs --log-verbosity 4 for library INFO messages.')
    loaded,total=map(int,counts[-1])
    if loaded<=0 or loaded!=total:
        raise RuntimeError(f'Full layer offload was not verified: {loaded}/{total} layers reported. '
                           'Inspect server.log; no CPU fallback accepted.')
    return [loaded,total]


def run_cpp(root,work):
    from huggingface_hub import hf_hub_download
    from .llm_next import imports
    cfg=read_json(root/'config.json');prompts=read_json(root/'prompts.json')
    directory=work/cfg['llama_release'];installed=read_json(directory/'verified.json')
    if installed['config_sha256']!=object_digest(cfg):raise RuntimeError('Installed binary/config mismatch')
    target=root/ENGINES[1];target.mkdir(exist_ok=True)
    status={'status':'downloading','started_at':now(),'binary':installed}
    write_json(target/'session.json',status)
    print('Downloading/reusing the one pinned Q4_K_M GGUF file...',flush=True)
    t=time.perf_counter();g=cfg['gguf']
    model=Path(hf_hub_download(g['repo'],g['filename'],revision=g['revision']))
    if model.stat().st_size!=g['bytes'] or digest(model)!=g['sha256']:raise RuntimeError('GGUF checksum/size mismatch')
    status['download_and_verify_s']=time.perf_counter()-t
    tf=imports();tok=tf.AutoTokenizer.from_pretrained(prompts['candidate']['model_id'],revision=prompts['candidate']['revision'])
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    client=LocalServer(port)
    args=[installed['server'],'--model',str(model),'--host','127.0.0.1','--port',str(port),
          '--n-gpu-layers','99','--split-mode','none','--ctx-size','512','--parallel','1',
          # At default verbosity 3, b11146 hides library INFO (including offload).
          # Level 4 includes it without enabling level-5 debug logging.
          '--cache-ram','0','--no-context-shift','--fit','off','--perf',
          '--log-verbosity','4','--log-colors','off']
    status['command']=args;write_json(target/'session.json',status)
    logpath=target/'server.log';t=time.perf_counter()
    with logpath.open('w') as log:
        server=subprocess.Popen(args,env=tool_environment(directory),stdout=log,stderr=subprocess.STDOUT)
        # Save PID immediately so the notebook can clean up on interruption.
        write_json(root/'server_pid.json',{'pid':server.pid,'binary':installed['server']})
        try:
            until=time.monotonic()+300;notice=0
            while True:
                if server.poll() is not None:raise RuntimeError('llama-server exited: '+logpath.read_text(errors='replace')[-5000:])
                try:
                    if client.json('/health',timeout=2).get('status')=='ok':break
                except (OSError,ValueError):pass
                if time.monotonic()>until:raise TimeoutError('llama-server not ready after five minutes; see server.log')
                if time.monotonic()>notice:print('Waiting for GGUF model loading...',flush=True);notice=time.monotonic()+15
                time.sleep(1)
            status['load_s']=time.perf_counter()-t
            lines=logpath.read_text(errors='replace')
            write_json(target/'session.json',status)
            status['offloaded_layers']=verify_layer_offload(lines)
            print('Verified GPU layer offload:',status['offloaded_layers'],flush=True)
            status['props']=client.json('/props')
            for case in prompts['samples']:
                ids=client.json('/tokenize',{'content':case['rendered_prompt'],'add_special':False,'parse_special':True})['tokens']
                if ids!=case['input_ids']:raise RuntimeError('Tokenizer IDs differ for '+case['id']+'; comparisons stopped before generation.')
            status['tokenizer_ids_match']=True
            def generate(case):
                t=time.perf_counter()
                with client.request('/completion',completion_payload(case)) as response:r=parse_sse(response,t)
                r['text_tokens']=len(tok(r['answer'],add_special_tokens=False)['input_ids'])
                if r['tokens_evaluated']!=case['input_tokens']:raise RuntimeError('Server prompt length differs; no comparison allowed.')
                return r
            with MemorySampler(server.pid) as memory:
                t=time.perf_counter()
                for repeat in (1,2):
                    for case in prompts['samples']:
                        print('llama.cpp warmup',repeat,case['id'],flush=True)
                        r=generate(case);r['timing_status']='warmup';record(root,ENGINES[1],case,repeat,r,True)
                status['warmup_s']=time.perf_counter()-t;write_json(target/'session.json',status)
                for repeat in (1,2,3):
                    cases=prompts['samples'];cases=cases[repeat-1:]+cases[:repeat-1]
                    for case in cases:
                        print('llama.cpp measured',repeat,case['id'],flush=True)
                        record(root,ENGINES[1],case,repeat,generate(case))
            status.update(status='complete',process_vram_peak_mib_sampled=memory.peak)
            write_json(target/'session.json',status)
        finally:stop_server(server)


def report(root):
    data=read_json(root/'prompts.json') if (root/'prompts.json').exists() else {'samples':[]}
    groups=[]
    for engine in ENGINES:
        for case in data['samples']:
            files=sorted((root/engine/'results').glob(case['id']+'__*.json'))
            rows=[read_json(p) for p in files]
            ok=(len(rows)==3 and {r['repeat'] for r in rows}=={1,2,3} and
                all(r['status']=='success' and r['stop_reason']=='eos' and r['timing_status']=='warmed'
                    and r['prompt_sha256']==case['prompt_sha256'] for r in rows))
            m=statistics.median([r['full_answer_s'] for r in rows]) if ok else None
            tokens=[r.get('first_observed_token_s') for r in rows]
            group={'case':case,'engine':engine,'complete':ok,'saved':len(rows),'results':rows,
                'evidence_sha256':object_digest({p.name:digest(p) for p in files}),
                'median_full_answer_s':m,'median_first_observed_token_s':statistics.median(tokens) if ok and all(x is not None for x in tokens) else None,
                'min_full_answer_s':min(r['full_answer_s'] for r in rows) if ok else None,
                'max_full_answer_s':max(r['full_answer_s'] for r in rows) if ok else None,
                'unique_answers':list(dict.fromkeys(r['answer'] for r in rows))}
            path=root/'groups'/f'{case["id"]}__{engine}.json'
            write_json(path,group)
            reviewed=read_json(root/'reviews'/path.name) if (root/'reviews'/path.name).exists() else {}
            group['quality']=reviewed.get('decision','not reviewed') if reviewed.get('result_sha256')==digest(path) else 'not reviewed'
            if not ok and group['quality']=='pass':group['quality']='invalid review'
            group['median_text_tokens']=statistics.median([r['text_tokens'] for r in rows]) if ok and all('text_tokens' in r for r in rows) else None
            groups.append(group)
    with (root/'comparison.csv').open('w',newline='') as f:
        fields=['case_id','engine','complete','saved','median_full_answer_s','median_first_observed_token_s','min_full_answer_s','max_full_answer_s','input_tokens','median_text_tokens','unique_answers','quality']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for g in groups:w.writerow({**{k:g[k] for k in fields if k in g},'case_id':g['case']['id'],'input_tokens':g['case']['input_tokens'],'unique_answers':len(g['unique_answers'])})
    lines=['# Gemma engine comparison','',
        'Same prompt tokens and GPU; NF4/compiled Transformers versus Q4_K_M/llama.cpp. Three warmed repeats per question. Quantization and conversion lineage differ. Only complete groups have timing summaries. Quality is reviewed separately; no automatic winner.',
        '', '|Question|Engine|Saved|Full answer median (s)|Common-tokenizer answer tokens|Quality|',
        '|---|---|---|---|---|---|']
    for g in groups:
        value=f'{g["median_full_answer_s"]:.3f}' if g['complete'] else 'incomplete'
        lines.append(f'|{g["case"]["id"]}|{g["engine"]}|{g["saved"]}/3|{value}|{g["median_text_tokens"]}|{g["quality"]}|')
    lines+=['','First observed token includes callback/local HTTP buffering. Full answer includes prefill and decoding but excludes prompt preparation, model loading and warmup. This is not ASR → TTS or first audio.',
            'Fixed engine order; small sample; no integrated VRAM/concurrency claim. Consult session.json for loading/warmup and sampled process VRAM (not an exact peak).']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report:',root/'comparison.csv',flush=True)


def run(repo,output,work):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'active.json').exists():
        root=output/read_json(output/'active.json')['root']
        print('Existing attempt:',root,'\nReview/export it; automatic reruns are disabled.',flush=True)
        report(root);return
    root=output/('engine-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid4().hex[:6])
    root.mkdir();write_json(output/'active.json',{'root':root.name})
    print('RESULT DIRECTORY:',root,flush=True)
    cfg=read_json(repo/'configs/llm_engine_comparison.json');write_json(root/'config.json',cfg)
    engine_order=list(reversed(ENGINES))  # Save candidate results before the slower HF compilation step.
    meta={'status':'starting','created_at':now(),'environment':environment(),'engine_order':engine_order,
          'source_sha256':digest(Path(__file__)),'model_catalog':read_json(repo/'configs/llm_next_candidates.json'),
          'timing':'Pre-rendered and pretokenized prompts; full answer includes GPU input creation (HF) or local HTTP/JSON/SSE (llama.cpp), prefill, decode and text streaming. No ASR/TTS/retrieval. First observed token includes transport/buffering and is not first audio.',
          'limitations':'Blocked engine order to avoid repeated loading; 3 repeats only. Different quantizers/conversion lineage and numerical guards. HF score checks retained; llama.cpp uses its own runtime. Output token counts can differ. No integrated memory-fit or production winner claim.'}
    write_json(root/'run.json',meta)
    try:
        prompts=prepare_prompts(repo);write_json(root/'prompts.json',prompts)
        print('Prompt lengths:',[(c['id'],c['input_tokens']) for c in prompts['samples']],flush=True)
        for engine in engine_order:
            print('Starting',engine,'alone on GPU 0...',flush=True)
            (root/engine).mkdir(exist_ok=True)
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['PYTHONUNBUFFERED']='1'
            args=[sys.executable,'-m','amharic_s2s.llm_engine_compare','worker','--engine',engine,'--root',str(root),'--work',str(work)]
            # Whole-process bound also protects against a stalled driver/compiler.
            with (root/engine/'console.log').open('w') as log:
                with subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True) as child:
                    def kill_owned_group():
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                    timer=threading.Timer(1500,kill_owned_group);timer.start()
                    try:
                        for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                        code=child.wait()
                    except BaseException:
                        kill_owned_group();raise
                    finally:timer.cancel();cleanup_server(root)
            report(root)
            if code:raise RuntimeError(engine+' failed. Export the partial result; do not repeat the same attempt.')
        meta['status']='complete'
    except BaseException as exc:
        meta.update(status='failed',error=type(exc).__name__+': '+str(exc))
        (root/'error.txt').write_text(traceback.format_exc());raise
    finally:write_json(root/'run.json',meta);report(root)


def cleanup_server(root):
    p=root/'server_pid.json'
    if not p.exists():return
    info=read_json(p);proc=Path('/proc')/str(info['pid'])
    try:
        # Only terminate the exact server recorded for this experiment; never kill
        # arbitrary model servers or another notebook's process.
        if proc.joinpath('exe').resolve()!=Path(info['binary']).resolve():return
        os.kill(info['pid'],15)
        for _ in range(20):
            if not proc.exists():return
            time.sleep(.1)
        if proc.joinpath('exe').resolve()==Path(info['binary']).resolve():os.kill(info['pid'],9)
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','run','worker','report'])
    p.add_argument('--repo',type=Path);p.add_argument('--output',type=Path);p.add_argument('--work',type=Path)
    p.add_argument('--root',type=Path);p.add_argument('--engine',choices=ENGINES)
    a=p.parse_args()
    if a.action=='setup':setup(a.repo,a.work)
    elif a.action=='run':run(a.repo,a.output,a.work)
    elif a.action=='report':report(a.root)
    else:
        try:
            if a.engine==ENGINES[0]:run_hf(a.root)
            else:run_cpp(a.root,a.work)
        except BaseException as exc:
            path=a.root/a.engine/'session.json';state=read_json(path) if path.exists() else {}
            state.update(status='failed',error=type(exc).__name__+': '+str(exc))
            write_json(path,state);raise


if __name__=='__main__':main()

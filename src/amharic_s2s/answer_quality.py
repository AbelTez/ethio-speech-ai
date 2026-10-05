"""Five paired everyday questions: instruction comparison for the selected Gemma model."""
import argparse
import csv
from pathlib import Path
import time
import traceback
from uuid import uuid4

from .artifacts import (read_json,write_json,digest,object_digest,source_snapshot,
                        environment,safe_name,now,export_run)
from .llm_backend import prompt_messages
from .llm_next import NextLLM,NEXT_PROTOCOL

PROTOCOL={**NEXT_PROTOCOL,'version':'answer-quality-v1','dataset':'five everyday questions, paired instructions',
          'order':'paired; control first on odd questions, direct first on even questions',
          'quality_first':True,'planned_repeats':1,'seed':42}
# Human Pass requires relevance, clear Amharic, practical advice, completeness and no repetition.


def cases(dataset):
    if set(dataset['instructions']) != {'control','direct'}:raise ValueError('Need both instruction versions.')
    if len(dataset['samples'])!=5:raise ValueError('This diagnostic expects five questions.')
    seen=set()
    out=[]
    for i,s in enumerate(dataset['samples']):
        safe_name(s['id'])
        if s['id'] in seen or not s['question'].strip() or not s['expected'].strip():raise ValueError('Invalid questions.')
        seen.add(s['id'])
        for variant in (('control','direct') if i%2==0 else ('direct','control')):
            sample={**s,'instruction':dataset['instructions'][variant],'context':''}
            # Expected answers are review-only and never passed to the model.
            out.append({'id':s['id']+'__'+variant,'question_id':s['id'],'variant':variant,
                'question':s['question'],'expected':s['expected'],
                'messages':prompt_messages({'instruction':sample['instruction']},sample)})
    return out


def prepare(repo,output,owner,env=None):
    data=read_json(repo/'data/llm_public/answer_quality.json');cases(data)
    candidate=read_json(repo/'configs/llm_next_candidates.json')['candidates']['gemma4_e4b']
    config={'dataset':data,'candidate':candidate,'preset':'nf4','protocol':PROTOCOL,
            'environment':environment() if env is None else env,'source':source_snapshot(repo)}
    name='answer-quality-'+object_digest(config)[:20]
    root=output/safe_name(owner)/'runs'/name
    if not (root/'run.json').exists():write_json(root/'run.json',{'run_id':name,'created_at':now(),'owner':owner,**config})
    write_json(output/owner/'active.json',{'run_id':name})
    print('RUN_ID:',name,flush=True)
    return root,config


def completed(root,key):
    p=root/'results'/f'{key}.json'
    return p.exists() and read_json(p).get('status')=='success'


def generate_cases(root,config,backend_class=None):
    todo=[c for c in cases(config['dataset']) if not completed(root,c['id'])]
    if not todo:
        print('All ten answers already saved; no model loaded.',flush=True);return
    stage='model_load';session=uuid4().hex
    try:
        start=time.perf_counter()
        backend=(backend_class or NextLLM)(config['candidate'],'nf4',PROTOCOL)
        backend.torch.manual_seed(PROTOCOL['seed'])
        backend.torch.cuda.synchronize()
        loaded=time.perf_counter()
        stage='warmup'
        backend.generate([{'role':'user','content':'በአማርኛ ሰላም በል።'}],max_new_tokens=8)
        backend.torch.cuda.synchronize()
        write_json(root/'sessions'/f'{session}.json',{'load_s':loaded-start,
            'warmup_s':time.perf_counter()-loaded,'started_at':now(),
            'generation_config':backend.generation.to_dict(),'loading_info':backend.loading_info})
        stage='answers'
        for c in todo:
            r={**c,'status':'running','session':session}
            try:
                backend.torch.manual_seed(PROTOCOL['seed'])
                metrics,details=backend.generate(c['messages'])
                r.update(status='success',metrics=metrics,details=details)
                print('\n'+c['id']+' | '+details['stop_reason']+' | '+str(round(metrics['full_answer_s'],3))+' s',flush=True)
                print('QUESTION:',c['question'],'\nANSWER:',details['answer'],flush=True)
            except BaseException as exc:
                r.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
            finally:write_json(root/'results'/f'{c["id"]}.json',r)
    except BaseException as exc:
        write_json(root/'failures'/f'{session}.json',{'stage':stage,'error':str(exc),'traceback':traceback.format_exc()})
        raise


def review(root,path):
    p=root/'reviews'/path.name
    value=read_json(p) if p.exists() else {}
    return value if value.get('result_sha256')==digest(path) else {}


def quality_pass(record,rating):
    d=record.get('details',{})
    return (record['status']=='success' and d.get('stop_reason')=='eos'
            and bool(d.get('answer','').strip()) and rating.get('decision')=='pass')


def report(output,owner):
    base=output/safe_name(owner)
    root=base/'runs'/safe_name(read_json(base/'active.json')['run_id'])
    metadata=read_json(root/'run.json')
    rows=[]
    for c in cases(metadata['dataset']):
        path=root/'results'/f'{c["id"]}.json'
        if not path.exists():continue
        r=read_json(path);v=review(root,path);m=r.get('metrics',{});d=r.get('details',{})
        rows.append({'run_id':root.name,'question_id':c['question_id'],'variant':c['variant'],
            'status':r['status'],'question':r['question'],'answer':d.get('answer',''),
            'expected':r['expected'],'stop_reason':d.get('stop_reason',''),
            'full_answer_s':m.get('full_answer_s'),'first_clause_s':m.get('first_clause_s'),
            'generated_tokens':m.get('generated_tokens'),'reviewed':v.get('decision') in ('pass','fail'),
            'quality_pass':quality_pass(r,v),'decision':v.get('decision'),
            'one_sentence':v.get('one_sentence'),'notes':v.get('notes',''),'error':r.get('error','')})
    dest=base/'reports';dest.mkdir(parents=True,exist_ok=True)
    with (dest/'responses.csv').open('w',encoding='utf-8-sig',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]) if rows else ['status']);w.writeheader();w.writerows(rows)
    write_json(dest/'responses.json',rows)
    text=['# Gemma everyday answer quality','',
          'Five paired questions, one pass each. Compare quality before latency. No ASR, TTS or pipeline timing.',
          'Code success means generation returned; capped answers cannot pass quality. Human ratings are required.',
          'One-sentence adherence is recorded separately from correctness. No automatic model selection.', '',
          '|Instruction|Saved / expected|Reviewed|Quality passes|','|---|---|---|---|']
    for variant in ('control','direct'):
        subset=[r for r in rows if r['variant']==variant]
        text.append(f'|{variant}|{sum(r["status"]=="success" for r in subset)}/5|{sum(r["reviewed"] for r in subset)}|{sum(r["quality_pass"] for r in subset)}|')
    (dest/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('\n'.join(text),flush=True)
    return root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','export'])
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--owner',default='abel')
    a=p.parse_args();safe_name(a.owner)
    if a.action=='run':
        root,config=prepare(a.repo,a.output,a.owner)
        try:generate_cases(root,config)
        finally:report(a.output,a.owner)
    else:
        report(a.output,a.owner)
        destination=a.output/f'answer-quality-{a.owner}-{uuid4().hex[:10]}.zip'
        print('RESULT ZIP:',export_run(a.output/a.owner,destination),flush=True)

if __name__=='__main__':main()

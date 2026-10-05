"""CPU-only source retrieval report for the document-grounded demo."""
import argparse
import json
from pathlib import Path
from .knowledge import KnowledgeIndex,social_reply,BANK_INSTRUCTION
from .llm_backend import prompt_messages

def check(repo,output,tokenizer=None):
    index=KnowledgeIndex.load(repo/'knowledge/bank_v2/bank_index.json')
    rows=[]
    for case in json.loads((repo/'configs/bank_demo_cases.json').read_text())['cases']:
        social=social_reply(case['question'])
        found=index.search(case['question']) if not social else {'hits':[],'search_ms':0}
        top=found['hits'][0] if found['hits'] else None
        row={**case,'social_template':social,'retrieval':found,'prompt_tokens':None,'prompt_fits':None}
        if top and tokenizer:
            context='[1] '+top['title']+'\n'+top['content']
            messages=prompt_messages({'instruction':BANK_INSTRUCTION},{'question':case['question'],'context':context})
            rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            n=len(tokenizer(rendered,add_special_tokens=False)['input_ids'])
            row.update(prompt_tokens=n,prompt_fits=n+128<=512)
        rows.append(row)
        print(case['id'],'|', 'social' if social else 'no evidence' if not top else top['sources'][0]['source'],
              '| prompt tokens:',row['prompt_tokens'],'| fits:',row['prompt_fits'],flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'corpus_sha256':index.digest,'cases':rows,
        'note':'Retrieval/packing only. No LLM generation or audio inference. No correctness score inferred from similarity.'},ensure_ascii=False,indent=2)+'\n')
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokenizer',action='store_true');a=p.parse_args();tok=None
    if a.tokenizer:
        from transformers import AutoTokenizer
        c=json.loads((a.repo/'configs/llm_next_candidates.json').read_text())['candidates']['gemma4_e4b']
        tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'])
    check(a.repo,a.output,tok)

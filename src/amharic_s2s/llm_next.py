"""Gemma 4 / Qwen3.5 text-only prompt diagnostic using a separate v5 runtime."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import sys
from uuid import uuid4

from .artifacts import read_json, export_run
from .llm_backend import TransformersLLM, text_only_imports, prompt_messages
from .llm_benchmark import PROTOCOL, run, report, validate_dataset

NEXT_PROTOCOL = {**PROTOCOL, 'version':'llm-next-am-en-v2-context-scope', 'max_new_tokens':128,
    'use_model_defaults':'not an API in Transformers 5; explicit generation arguments',
    'dataset':'three identical questions, paired short Amharic/English instructions',
    'compute_dtype':'float16', 'quantization':'NF4 double quantization; lm_head and Qwen time gates unquantized',
    'backend':'Transformers 5.17.0 text submodels; native PyTorch without external recurrent kernels'}


def imports(preserve_audio=False):
    # Isolated subprocess; keep inherited optional integrations out of this text test.
    text_only_imports(preserve_audio=preserve_audio)
    for name in ('fla','causal_conv1d','mamba_ssm','peft'):
        if name not in sys.modules:
            sys.modules[name] = None
    import transformers
    if transformers.__version__ != '5.17.0':
        raise RuntimeError('Use the separate llm-next venv with Transformers 5.17.0.')
    return transformers


def text_config_and_class(candidate,preserve_audio=False):
    tf = imports(preserve_audio=preserve_audio)
    from huggingface_hub import hf_hub_download
    cfg = read_json(hf_hub_download(candidate['model_id'],'config.json',revision=candidate['revision']))
    if candidate['loader']=='gemma4_text':
        return tf.Gemma4TextConfig(**cfg['text_config']), tf.Gemma4ForCausalLM
    if candidate['loader']=='qwen35_text':
        return tf.Qwen3_5TextConfig(**cfg['text_config']), tf.Qwen3_5ForCausalLM
    raise ValueError('Unknown next-round architecture.')


class NextLLM(TransformersLLM):
    def __init__(self,candidate,preset,protocol,shared_audio=False):
        tf = imports(preserve_audio=shared_audio)
        import torch
        if preset != 'nf4':
            raise ValueError('This initial diagnostic has one recorded preset: nf4 with FP16 compute.')
        if not torch.cuda.is_available():
            raise RuntimeError('Select a CUDA GPU runtime.')
        self.torch, self.protocol, self.candidate, self.preset = torch,protocol,candidate,preset
        self.tokenizer=tf.AutoTokenizer.from_pretrained(candidate['model_id'],revision=candidate['revision'])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token
        config,loader=text_config_and_class(candidate,preserve_audio=shared_audio)
        skip=['lm_head']
        if candidate['loader']=='qwen35_text':
            skip+=['in_proj_a','in_proj_b']
        quant=tf.BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=skip)
        self.model, info=loader.from_pretrained(candidate['model_id'],revision=candidate['revision'],
            config=config,dtype=torch.float16,quantization_config=quant,
            device_map={'model':0,'lm_head':0},attn_implementation='sdpa',
            key_mapping={r'^model\.language_model\.':'model.'},output_loading_info=True)
        # V5 may return sets; normalize these for durable JSON metadata.
        def serial(v):
            if isinstance(v,dict): return {k:serial(x) for k,x in v.items()}
            if isinstance(v,(set,tuple,list)): return [serial(x) for x in v]
            return v if isinstance(v,(str,int,float,bool,type(None))) else str(v)
        self.loading_info=serial(info)
        if any(info.get(k) for k in ('missing_keys','mismatched_keys','error_msgs','conversion_errors')):
            raise RuntimeError(f'Incomplete text-model load: {self.loading_info}')
        allowed=('model.vision_tower.','model.audio_tower.','model.embed_vision.','model.embed_audio.',
                 'model.visual.','mtp.')
        unexpected=[k for k in info.get('unexpected_keys',[]) if not k.startswith(allowed)]
        if unexpected: raise RuntimeError(f'Unexpected text weights: {unexpected[:15]}')
        self.model.eval()
        if any(p.device.type!='cuda' for p in self.model.parameters()):
            raise RuntimeError('GPU-only inference required; CPU/disk offload was detected.')
        eos = candidate['eos_token_ids']
        self.generation=tf.GenerationConfig(do_sample=False,num_beams=1,num_return_sequences=1,
            max_new_tokens=protocol['max_new_tokens'],max_time=protocol['max_time_s'],
            use_cache=True,pad_token_id=self.tokenizer.pad_token_id,eos_token_id=eos,
            bos_token_id=self.tokenizer.bos_token_id,repetition_penalty=1.0)
        # V5 has removed use_model_defaults. Leave architecture-specific cache selection
        # to the model (Qwen3.5 uses a hybrid recurrent/attention cache).
        self.generation_overrides={}
        self.TextStreamer=tf.TextStreamer
        class CheckScores(tf.LogitsProcessor):
            def __call__(self,input_ids,scores):
                valid=(~torch.isnan(scores).any() & ~torch.isposinf(scores).any()
                       & torch.isfinite(scores).any(dim=-1).all())
                if not valid.item():
                    raise FloatingPointError('Invalid generation scores; do not treat this as model language quality.')
                return scores
        self.score_checks=tf.LogitsProcessorList([CheckScores()])


def preflight(repo,key):
    tf=imports()
    from huggingface_hub import hf_hub_download
    candidate=read_json(repo/'configs/llm_next_candidates.json')['candidates'][key]
    dataset=read_json(repo/'data/llm_public/next_prompts.json')
    validate_dataset(dataset)
    cfg,cls=text_config_and_class(candidate)
    tokenizer=tf.AutoTokenizer.from_pretrained(candidate['model_id'],revision=candidate['revision'])
    lengths=[]
    for sample in dataset['samples']:
        rendered=tokenizer.apply_chat_template(prompt_messages(dataset,sample),tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer(rendered,add_special_tokens=False)['input_ids']
        if min(ids)<0 or max(ids)>=cfg.vocab_size:
            raise ValueError('Tokenizer IDs outside model vocabulary.')
        lengths.append(len(ids))
    if max(lengths)>NEXT_PROTOCOL['max_input_tokens']:
        raise ValueError('Input exceeds protocol limit; no truncation allowed.')
    if any(i>=cfg.vocab_size for i in candidate['eos_token_ids']):
        raise ValueError('EOS token outside vocabulary.')
    # Qwen is sharded; Gemma is a single safetensors file with no index.
    if candidate['loader']=='qwen35_text':
        index=read_json(hf_hub_download(candidate['model_id'],'model.safetensors.index.json',revision=candidate['revision']))
        if not any(k.startswith('model.language_model.') for k in index['weight_map']):
            raise ValueError('Checkpoint text prefix differs from the verified loading map.')
    print('Selected:',candidate['model_id'],'revision:',candidate['revision'])
    print('Loader:',cls.__name__,'| NF4 linear weights / FP16 compute; text only')
    print('Six prompt lengths:',lengths)
    print('Preflight passed. No model weights loaded.')


def next_report(output,owner):
    path=report(output,owner)
    lines=['# New-model prompt diagnostic','',
        'Six answers per model: three questions repeated with Amharic (_am) and English (_en) instructions.',
        'Questions and context stay in Amharic in both variants. All answers must be Amharic.',
        'Prompt correction v2: everyday questions use general knowledge; supplied-context restrictions remain on grounded/guard cases.',
        'Compare with v1 by matching model, preset, sample ID and environment; changed prompts produce a new comparison group.',
        'Compare answer pairs before judging prompt sensitivity. Three cases cannot establish general quality.',
        'Preset: NF4 with FP16 compute, explicit greedy decoding, thinking disabled. Each configuration needs human review.',
        'Keep this round separate from the earlier five-model screening. No pipeline latency claim.', '',
        '|Model|Instruction|Prompt|Status|Stop|Full answer s|First clause s|',
        '|---|---|---|---|---|---|---|']
    import csv
    with (path/'responses.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        lines.append('|'+ '|'.join([r['model_key'],r['sample_id'].rsplit('_',1)[-1],r['sample_id'],
            r.get('status',''),r.get('stop_reason',''),r.get('full_answer_s',''),r.get('first_clause_s','')])+'|')
    (path/'NEXT_REPORT.md').write_text('\n'.join(lines)+'\n')
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preflight','run','report','export'])
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path)
    p.add_argument('--owner',default='abel')
    p.add_argument('--model',choices=['qwen35_4b','gemma4_e4b'],default='qwen35_4b')
    p.add_argument('--sample-limit',type=int,choices=[3,0],default=3)
    p.add_argument('--reviewed-run')
    a=p.parse_args()
    if a.action=='preflight': preflight(a.repo,a.model);return
    if a.output is None: p.error('--output is required')
    if a.action=='run':
        try:
            run(a.repo,a.output,a.owner,a.model,'nf4',a.sample_limit,a.reviewed_run,
                backend_class=NextLLM,catalog_path=a.repo/'configs/llm_next_candidates.json',
                dataset_path=a.repo/'data/llm_public/next_prompts.json',protocol=NEXT_PROTOCOL)
        finally: next_report(a.output,a.owner)
    else:
        next_report(a.output,a.owner)
        if a.action=='export':
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            print(export_run(a.output/a.owner,a.output/f'llm-next-{a.owner}-{stamp}-{uuid4().hex[:6]}.zip'))

if __name__=='__main__':main()

"""Observe the existing pipeline generator without changing its decoding policy."""
from contextlib import contextmanager,ExitStack
from functools import wraps
import threading
import time
from types import SimpleNamespace


class TokenizerTimer:
    def __init__(self,base,trace):self.base,self.trace=base,trace
    def __getattr__(self,name):return getattr(self.base,name)
    def __call__(self,*args,**kwargs):
        start=time.perf_counter();result=self.base(*args,**kwargs)
        self.trace.tokenization_s+=time.perf_counter()-start
        self.trace.input_tokens=int(result['input_ids'].shape[-1])
        return result
    def apply_chat_template(self,*args,**kwargs):
        start=time.perf_counter();result=self.base.apply_chat_template(*args,**kwargs)
        self.trace.render_s+=time.perf_counter()-start
        return result


class GenerationTrace:
    def __init__(self,torch):
        self.torch=torch;self.forwards=[];self.checks=[];self.tokens=[]
        self.render_s=0.;self.tokenization_s=0.;self.input_tokens=None

    @contextmanager
    def installed(self,backend):
        trace=self
        def replace(stack,obj,name,value):
            original=getattr(obj,name)
            stack.callback(setattr,obj,name,original);setattr(obj,name,value)
        with ExitStack() as stack:
            original_forward=backend.model.forward
            @wraps(original_forward)
            def forward(*args,**kwargs):
                start=self.torch.cuda.Event(enable_timing=True);end=self.torch.cuda.Event(enable_timing=True)
                host=time.perf_counter();start.record()
                output=original_forward(*args,**kwargs)
                end.record()
                self.forwards.append({'host_start':host,'host_return':time.perf_counter(),
                    'start_event':start,'end_event':end,
                    'cache_type':type(getattr(output,'past_key_values',None)).__name__})
                return output
            replace(stack,backend.model,'forward',forward)
            replace(stack,backend,'tokenizer',TokenizerTimer(backend.tokenizer,self))
            checks=[]
            for processor in backend.score_checks:
                class TimedCheck:
                    def __init__(self,original):self.original=original
                    def __call__(self,input_ids,scores):
                        start=time.perf_counter()
                        try:return self.original(input_ids,scores)
                        finally:trace.checks.append(time.perf_counter()-start)
                checks.append(TimedCheck(processor))
            replace(stack,backend,'score_checks',type(backend.score_checks)(checks))
            original_streamer=backend.TextStreamer
            class TimedStreamer(original_streamer):
                def put(self,value):
                    is_prompt=self.skip_prompt and self.next_tokens_are_prompt
                    arrived=time.perf_counter()
                    result=super().put(value)
                    if not is_prompt:trace.tokens.append({'time':arrived,'count':int(value.numel())})
                    return result
            replace(stack,backend,'TextStreamer',TimedStreamer)
            yield self

    def summarize(self,origin):
        # Called only after the generation stream has synchronized.
        rows=[{'index':i,'host_start_s':r['host_start']-origin,
               'host_dispatch_s':r['host_return']-r['host_start'],
               'gpu_stream_s':r['start_event'].elapsed_time(r['end_event'])/1000,
               'cache_type':r['cache_type']} for i,r in enumerate(self.forwards)]
        tokens=[{'ready_s':t['time']-origin,'count':t['count']} for t in self.tokens]
        metrics={'prompt_render_s':self.render_s,'tokenization_s':self.tokenization_s,
                 'input_tokens':self.input_tokens,'forward_calls':len(rows),
                 'score_check_calls':len(self.checks),'score_check_wall_s':sum(self.checks),
                 'score_check_mean_wall_s':sum(self.checks)/len(self.checks) if self.checks else None,
                 'first_token_s':tokens[0]['ready_s'] if tokens else None,
                 'streamed_token_count':sum(t['count'] for t in tokens),
                 'before_first_forward_s':rows[0]['host_start_s'] if rows else None,
                 'prefill_gpu_stream_s':rows[0]['gpu_stream_s'] if rows else None,
                 'decode_gpu_stream_s':sum(r['gpu_stream_s'] for r in rows[1:]) if rows else None}
        gaps=[b['ready_s']-a['ready_s'] for a,b in zip(tokens,tokens[1:])]
        metrics['mean_intertoken_s']=sum(gaps)/len(gaps) if gaps else None
        return metrics,{'forwards':rows,'token_arrivals':tokens,'score_check_wall_s_each':self.checks,
            'interpretation':'GPU stream and host intervals overlap; do not sum. Score-check wall time includes waiting for queued GPU work and is not pure validation cost.'}


def generate_answer(backend,stream,messages,profiled=False,cap=128):
    from .pipeline_backend import PipelineModels
    events={};texts=[]
    def mark(name):events[name]=time.perf_counter()
    def emit(text):
        if text and 'first_text' not in events:mark('first_text')
        texts.append(text)
    proxy=SimpleNamespace(llm=backend,torch=backend.torch,llm_stream=stream)
    backend.torch.cuda.synchronize()
    trace=GenerationTrace(backend.torch)
    from contextlib import nullcontext
    with trace.installed(backend) if profiled else nullcontext():
        result=PipelineModels.generate_text(proxy,messages,emit,threading.Event(),mark,cap=cap)
    if ''.join(texts).strip()!=result['answer'].strip():raise RuntimeError('Stream and answer differ.')
    metrics={'full_answer_s':events['llm_done']-events['llm_start'],
             'first_text_s':events.get('first_text',events['llm_done'])-events['llm_start'],
             'generated_tokens':result['generated_tokens']}
    details={}
    if profiled:
        extra,details=trace.summarize(events['llm_start']);metrics.update(extra)
        if metrics['forward_calls']!=result['generated_tokens'] or metrics['streamed_token_count']!=result['generated_tokens']:
            raise RuntimeError('Unexpected forward/token counts; inspect generation before interpreting the profile.')
    return metrics,{**result,'profile':details}

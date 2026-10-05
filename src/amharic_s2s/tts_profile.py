"""Observe nested OmniVoice stages without changing synthesis arguments or outputs.

Wall times include synchronization overhead. Use exclusive times for sums;
inclusive parent times contain their children and must not be added together.
"""
from functools import wraps
import time

HOOKS=(('_preprocess_all','preparation'),('_prepare_inference_inputs','preparation'),
       ('_generate_iterative','token_generation'),('_generate_chunked','chunk_orchestration'),
       ('_decode_and_post_process','decode_dispatch_and_transfer'),
       ('_post_process_audio','audio_postprocessing'))


class StageProfiler:
    def __init__(self,sync,clock=time.perf_counter):
        self.sync,self.clock=sync,clock
        self.active=False;self.stack=[];self.rows=[]
        self.observed=[];self.missing=[]

    def attach(self,obj,name,stage):
        method=getattr(obj,name,None)
        label=stage+':'+name
        if not callable(method):self.missing.append(label);return
        if getattr(method,'_tts_stage_profiled',False):raise ValueError('Profiler already attached.')
        @wraps(method)
        def wrapped(*args,**kwargs):
            if not self.active:return method(*args,**kwargs)
            self.sync();start=self.clock()
            frame={'children_s':0.0};self.stack.append(frame)
            try:return method(*args,**kwargs)
            finally:
                try:self.sync()
                finally:
                    elapsed=self.clock()-start
                    self.stack.pop()
                    exclusive=max(0.0,elapsed-frame['children_s'])
                    self.rows.append({'stage':stage,'method':name,'inclusive_s':elapsed,'exclusive_s':exclusive})
                    if self.stack:self.stack[-1]['children_s']+=elapsed
        wrapped._tts_stage_profiled=True
        setattr(obj,name,wrapped);self.observed.append(label)

    def measure(self,method,*args,**kwargs):
        if self.active:raise RuntimeError('One TTS call at a time is required.')
        self.rows=[];self.stack=[];self.sync();start=self.clock();self.active=True
        try:
            result=method(*args,**kwargs)
            self.sync();total=self.clock()-start
        finally:self.active=False
        stages={}
        for row in self.rows:
            s=stages.setdefault(row['stage'],{'calls':0,'inclusive_s':0.,'exclusive_s':0.})
            s['calls']+=1
            for key in ('inclusive_s','exclusive_s'):s[key]+=row[key]
        accounted=sum(s['exclusive_s'] for s in stages.values())
        profile={'total_s':total,'stages':stages,'unattributed_s':max(0.,total-accounted),
                 'observed_hooks':self.observed,'missing_hooks':self.missing,
                 'timing':'synchronized wall-clock; exclusive times avoid nested double counting; instrumentation overhead included'}
        return result,profile


class ProfiledTTS:
    def __init__(self,backend):
        self.backend=backend
        self.profiler=StageProfiler(backend.sync)
        for name,stage in HOOKS:self.profiler.attach(backend.model,name,stage)
        self.profiler.attach(backend.model.audio_tokenizer,'decode','codec_decode')

    def __getattr__(self,name):return getattr(self.backend,name)

    def generate(self,text):
        (wave,sr,details),profile=self.profiler.measure(self.backend.generate,text)
        return wave,sr,{**details,'stage_profile':profile}

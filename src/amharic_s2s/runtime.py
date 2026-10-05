"""Resident single-GPU runtime. No HTTP, automatic conversation storage or model imports."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import logging
import threading
import time
from typing import Any, Callable
from uuid import uuid4


class RuntimeErrorBase(Exception):
    """Errors an application adapter can map to an explicit response."""


class InvalidInput(RuntimeErrorBase):pass
class Busy(RuntimeErrorBase):pass
class NotReady(RuntimeErrorBase):pass
class Cancelled(RuntimeErrorBase):pass
class RejectedAnswer(RuntimeErrorBase):pass
class NoSpeech(RejectedAnswer):
    """ASR produced no usable text; the resident models can serve another turn."""
class InferenceFailure(RuntimeErrorBase):pass


@dataclass(frozen=True)
class Limits:
    min_audio_s: float = 1.0
    max_audio_s: float = 30.0
    min_sample_rate: int = 8000
    max_sample_rate: int = 96000
    max_context_chars: int = 4096


@dataclass(frozen=True)
class VoiceRequest:
    """Audio is float PCM in [-1,1], frames or frames×channels (one/two channels)."""
    audio: Any
    sample_rate: int
    context: str = ''
    grounded: bool = False
    history: tuple = ()


@dataclass(frozen=True)
class VoiceResponse:
    request_id: str
    transcript: str
    answer: str
    audio: Any
    sample_rate: int
    timings: dict
    compilation: dict
    details: dict = field(default_factory=dict)


def validate_request(request, limits):
    import numpy as np
    if not isinstance(request,VoiceRequest):raise InvalidInput('Expected a VoiceRequest.')
    sr=request.sample_rate
    if type(sr) is not int or not limits.min_sample_rate<=sr<=limits.max_sample_rate:
        raise InvalidInput('Unsupported sample rate.')
    if not isinstance(request.context,str) or len(request.context)>limits.max_context_chars:
        raise InvalidInput('Context must be text within the configured length limit.')
    if type(request.grounded) is not bool:raise InvalidInput('grounded must be a boolean.')
    history=request.history
    if not isinstance(history,(list,tuple)) or len(history)>4 or len(history)%2:
        raise InvalidInput('History must contain at most two complete exchanges.')
    clean=[]
    for i,m in enumerate(history):
        if (not isinstance(m,dict) or set(m)!={'role','content'} or
            m['role']!=('user' if i%2==0 else 'assistant') or
            not isinstance(m['content'],str) or not 0<len(m['content'])<=2000):
            raise InvalidInput('Invalid conversation history.')
        clean.append(dict(m))
    a=np.asarray(request.audio)
    if a.ndim not in (1,2) or (a.ndim==2 and a.shape[1] not in (1,2)):
        raise InvalidInput('Use mono or stereo audio in frames×channels order.')
    if not limits.min_audio_s<=len(a)/sr<=limits.max_audio_s:
        raise InvalidInput(f'Audio must be {limits.min_audio_s:g}–{limits.max_audio_s:g} seconds.')
    if a.dtype.kind!='f':raise InvalidInput('Audio must be floating-point PCM in [-1,1].')
    if not np.isfinite(a).all() or np.max(np.abs(a))>1:
        raise InvalidInput('Audio contains invalid or out-of-range samples.')
    # Own the data before submitting it to a different thread.
    a=np.array(a,dtype=np.float32,copy=True)
    if a.ndim==2:a=a.mean(axis=1)
    if np.max(np.abs(a))<1e-5:raise InvalidInput('Audio is silent.')
    a.setflags(write=False)
    return VoiceRequest(a,sr,request.context,request.grounded,tuple(clean))


def validate_output(audio,sample_rate):
    import numpy as np
    a=np.asarray(audio,dtype=np.float32).squeeze()
    if type(sample_rate) is not int or sample_rate<=0 or a.ndim!=1 or not len(a):
        raise InferenceFailure('Speech generation returned invalid audio.')
    if not np.isfinite(a).all() or np.max(np.abs(a))<1e-7:
        raise InferenceFailure('Speech generation returned invalid audio.')
    a=np.array(a,copy=True);a.setflags(write=False)
    return a


class VoiceRuntime:
    """One persistent worker owns model loading, warmup and every inference call.

    start/respond/close are synchronous. The future HTTP adapter must call them off
    its event loop. Busy callers are rejected immediately, never queued on the GPU.
    Cancellation is cooperative; it cannot instantly stop an executing CUDA kernel.
    """
    def __init__(self,repo: Path,*,limits=None,backend_factory: Callable | None=None):
        self.repo=Path(repo)
        self.limits=limits or Limits()
        if backend_factory is None:
            import os
            engine=os.environ.get('AMHARIC_LLM_ENGINE','transformers_compiled')
            if engine=='llama_cpp':
                from .runtime_cpp import CppResidentBackend
                backend_factory=CppResidentBackend
            elif engine=='transformers_compiled':
                from .runtime_backend import ResidentBackend
                backend_factory=ResidentBackend
            else:raise ValueError('Unknown AMHARIC_LLM_ENGINE; choose llama_cpp or transformers_compiled.')
        self._factory=backend_factory
        self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='amharic-inference')
        self._gate=threading.Lock();self._lock=threading.RLock()
        self._state='new';self._backend=None;self._cancel=threading.Event()
        self._startup={};self._capacity=[];self._last_error=None;self._active=False

    def status(self):
        from copy import deepcopy
        with self._lock:
            return {'state':self._state,'ready':self._state=='ready',
                'busy':self._active,'accepting_requests':self._state=='ready' and not self._active,
                'startup':deepcopy(self._startup),'capacity':deepcopy(self._capacity),
                'last_error':self._last_error}

    def _capacity_update(self,row):
        with self._lock:self._capacity.append(dict(row))

    def start(self,warmup_requests):
        """Warm on non-private development inputs; idempotent once ready."""
        if not self._gate.acquire(blocking=False):raise Busy('Runtime is processing another operation.')
        try:
            with self._lock:
                if self._state=='ready':return self.status()
                if self._state!='new':raise NotReady('Create a new runtime after failure or shutdown.')
            cases=[validate_request(r,self.limits) for r in warmup_requests]
            if not cases:raise InvalidInput('At least one warmup request is required.')
            with self._lock:self._state='starting';self._cancel.clear()
            try:self._executor.submit(self._initialize,cases).result()
            except Exception:
                with self._lock:self._state='failed';self._last_error='startup_failed'
                raise InferenceFailure('Model startup/warmup failed; runtime is unavailable.') from None
            with self._lock:
                if self._state!='closing':self._state='ready'
            return self.status()
        finally:self._gate.release()

    def _initialize(self,cases):
        start=time.perf_counter()
        self._backend=self._factory(self.repo,self._capacity_update)
        loaded=time.perf_counter()
        try:
            # Identical worker/context for warmup and live requests.
            for _ in range(2):
                for request in cases:self._perform(request,warm=True)
            self._backend.finish_warmup()
            with self._lock:self._startup={'load_s':loaded-start,'warmup_s':time.perf_counter()-loaded,
                'warmup_requests':len(cases)*2}
        except Exception:
            self._dispose()
            raise

    def respond(self,request):
        if not self._gate.acquire(blocking=False):raise Busy('GPU is busy; try again after the current request.')
        try:
            with self._lock:
                if self._state!='ready':raise NotReady('Runtime must finish startup before accepting requests.')
            request=validate_request(request,self.limits)
            with self._lock:self._active=True;self._cancel.clear()
            try:return self._executor.submit(self._perform,request).result()
            except (InvalidInput,Cancelled,RejectedAnswer):raise
            except Exception:
                # Keep the actual live-inference traceback in the operator's log.
                # The HTTP response and readiness status remain sanitized.
                logging.getLogger(__name__).exception('Resident live inference failed; last stage: %s',
                    getattr(self,'_turn_stage','unknown'))
                with self._lock:self._state='failed';self._last_error='inference_failed'
                # A numerical/CUDA/compiler fault cannot silently fall back or serve more traffic.
                try:self._executor.submit(self._dispose).result()
                except Exception:
                    with self._lock:self._last_error='inference_cleanup_failed'
                raise InferenceFailure('Inference failed; create a new runtime before serving more requests.') from None
        finally:
            with self._lock:self._active=False
            self._gate.release()

    def cancel(self):
        """Stop at supported boundaries; busy remains true until work actually exits."""
        with self._lock:
            if self._active or self._state=='starting':self._cancel.set();return True
            return False

    def _check_cancelled(self):
        if self._cancel.is_set():raise Cancelled('Request cancelled.')

    def _perform(self,request,warm=False):
        from .pipeline_core import quality_flags
        b=self._backend;events={};origin=time.perf_counter()
        self._turn_stage='begin_turn'
        def mark(name):
            self._turn_stage=name
            events[name]=time.perf_counter()-origin
        self._check_cancelled();b.begin_turn()
        transcript,raw=b.transcribe(request.audio,request.sample_rate,mark)
        self._check_cancelled()
        if not isinstance(transcript,str) or not transcript.strip():
            raise NoSpeech('No speech transcript was produced.')
        self._turn_stage='llm_prepare'
        result,compilation=b.generate(transcript,request.context,request.grounded,self._cancel,mark,warm=warm,history=request.history)
        self._check_cancelled()
        if result.get('stop_reason')!='eos':raise RejectedAnswer('The model did not finish a complete answer.')
        answer=result['answer']
        if quality_flags(answer):raise RejectedAnswer('The answer failed basic speech-text checks.')
        mark('tts_start');wave,sr,tts_details=b.synthesize(answer.strip());mark('audio_ready')
        self._check_cancelled();wave=validate_output(wave,sr)
        memory=b.memory()
        delta=lambda a,z:events[z]-events[a]
        timings={'asr_total_s':delta('asr_prepare_start','asr_done'),
            'asr_inference_and_decode_s':delta('asr_inference_start','asr_done'),
            'llm_full_answer_s':delta('llm_start','llm_done'),
            'tts_generation_s':delta('tts_start','audio_ready'),
            'asr_to_tts_start_s':delta('asr_inference_start','tts_start'),
            'asr_to_audio_ready_s':delta('asr_inference_start','audio_ready'),
            'retrieval_search_s':result.get('retrieval',{}).get('search_ms',0)/1000,
            'request_processing_s':time.perf_counter()-origin}
        return VoiceResponse(uuid4().hex,transcript,answer,wave,sr,timings,compilation,
            {'stop_reason':result['stop_reason'],'generated_tokens':result['generated_tokens'],
             'tts':tts_details,'memory':memory,'quality_reviewed':False,
             'grounding':result.get('grounding',{}),'retrieval':result.get('retrieval')})

    def _dispose(self):
        b,self._backend=self._backend,None
        if b is not None:b.close()

    def close(self):
        # Wait for the one active call to finish/cancel before touching its weights.
        with self._lock:
            if self._state=='closed':return
            self._state='closing';self._cancel.set()
        with self._gate:
            with self._lock:
                if self._state=='closed':return
            try:self._executor.submit(self._dispose).result()
            finally:
                self._executor.shutdown(wait=True)
                with self._lock:self._state='closed';self._active=False

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def recorded_warmup(repo):
    """Load the three development recordings explicitly, without saving requests."""
    from .artifacts import read_json
    from .pipeline_benchmark import read_audio
    repo=Path(repo);requests=[]
    for s in read_json(repo/'data/pipeline_public/questions_v2.json')['samples']:
        audio,sr=read_audio(repo/'data/pipeline_v2_audio'/s['audio_file'])
        requests.append(VoiceRequest(audio,sr,s['context'],bool(s['context'])))
    return requests

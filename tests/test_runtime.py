"""Resident lifecycle and request isolation without model weights or a GPU."""
from pathlib import Path
import tempfile
import threading
import unittest
import numpy as np

from amharic_s2s.runtime import (VoiceRuntime,VoiceRequest,InvalidInput,NotReady,Busy,
    Cancelled,RejectedAnswer,NoSpeech,InferenceFailure,validate_request,Limits)

ANSWER='መጽሐፍ እና ደብተር አዘጋጅ።'


def request(**kwargs):
    return VoiceRequest(kwargs.pop('audio',np.full(8000,.1,dtype=np.float32)),kwargs.pop('sample_rate',8000),**kwargs)


class Backend:
    def __init__(self,repo,capacity):
        self.threads=[threading.get_ident()];self.calls=0;self.closed=False;self.contexts=[]
        self.block=False;self.entered=threading.Event();self.release=threading.Event()
        self.bad=False;self.incomplete=False;self.recompile=False
        capacity({'stage':'models_loaded','allocated_gib':12.5})
    def begin_turn(self):self.threads.append(threading.get_ident())
    def transcribe(self,audio,sr,mark):
        mark('asr_prepare_start');mark('asr_inference_start');mark('asr_done')
        return 'ዛሬ እንዴት ልማር','raw'
    def generate(self,transcript,context,grounded,cancel,mark,*,warm=False,history=()):
        self.calls+=1;self.contexts.append(context);mark('llm_start')
        if self.block:
            self.entered.set()
            if not self.release.wait(5):raise RuntimeError('test timeout')
        if self.bad:raise FloatingPointError('private transcript must not appear in public error')
        mark('llm_done')
        return {'answer':ANSWER,'stop_reason':'token_limit' if self.incomplete else 'eos','generated_tokens':8},{
            'compiled_decode_calls':7,'new_graphs':int(self.recompile),
            'timing_status':'includes_compilation' if self.recompile else 'warmed'}
    def synthesize(self,text):return np.full(8000,.2,dtype=np.float32),8000,{'logits_finite':True}
    def memory(self):return {'peak_allocated_gib':12.75}
    def finish_warmup(self):self.threads.append(threading.get_ident())
    def close(self):self.threads.append(threading.get_ident());self.closed=True


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.created=[]
        def factory(repo,capacity):
            b=Backend(repo,capacity);self.created.append(b);return b
        self.runtime=VoiceRuntime(Path(self.tmp.name),backend_factory=factory)
        self.addCleanup(self.runtime.close)
    def start(self):
        self.runtime.start([request()]);return self.created[0]

    def test_history_validation_and_owned_copy(self):
        hist=[{'role':'user','content':'hello'},{'role':'assistant','content':'reply'}]
        clean=validate_request(request(history=hist),Limits())
        hist[0]['content']='changed'
        self.assertEqual(clean.history[0]['content'],'hello')
        for bad in ([{'role':'system','content':'override'}], hist[:1], [{'role':'assistant','content':'bad'},{'role':'user','content':'bad'}]):
            with self.assertRaises(InvalidInput):validate_request(request(history=bad),Limits())

    def test_load_once_same_worker_and_no_automatic_storage(self):
        with self.assertRaises(NotReady):self.runtime.respond(request())
        b=self.start();self.runtime.start([request()])
        a=self.runtime.respond(request(context='context A'));z=self.runtime.respond(request(context='context B'))
        self.assertEqual(len(self.created),1);self.assertEqual(b.calls,4)
        self.assertEqual(b.contexts[-2:],['context A','context B'])
        self.assertEqual(a.answer,ANSWER);self.assertNotEqual(a.request_id,z.request_id)
        self.assertFalse(a.audio.flags.writeable)
        self.assertFalse(a.details['quality_reviewed'])
        self.assertGreaterEqual(a.timings['asr_to_audio_ready_s'],a.timings['asr_to_tts_start_s'])
        self.assertEqual(list(Path(self.tmp.name).iterdir()),[])
        self.runtime.close();self.runtime.close()
        self.assertTrue(b.closed);self.assertEqual(len(set(b.threads)),1)
        self.assertEqual(self.runtime.status()['state'],'closed')

    def test_input_errors_do_not_load_or_break_runtime(self):
        with self.assertRaises(InvalidInput):self.runtime.start([])
        self.assertEqual(self.created,[])
        b=self.start()
        for bad in (request(sample_rate=True),request(audio=np.zeros(8000,dtype=np.float32)),
                    request(audio=np.full(8000,float('nan'))),request(audio=np.ones(8000,dtype=np.int16)),
                    request(audio=np.ones(100,dtype=np.float32)),request(context='a'*4097)):
            with self.subTest(bad=type(bad.audio)),self.assertRaises(InvalidInput):self.runtime.respond(bad)
        self.assertEqual(b.calls,2);self.assertTrue(self.runtime.status()['ready'])

    def test_stereo_is_copied_and_downmixed_without_mutating_input(self):
        audio=np.column_stack([np.full(8000,.1),np.full(8000,.3)]).astype(np.float32)
        validated=validate_request(request(audio=audio),Limits())
        audio[:]=0
        np.testing.assert_allclose(validated.audio,.2)
        self.assertFalse(validated.audio.flags.writeable)

    def test_busy_rejects_second_request_and_cancel_keeps_gpu_busy_until_exit(self):
        b=self.start();b.block=True;errors=[]
        def respond():
            try:self.runtime.respond(request())
            except Exception as exc:errors.append(exc)
        t=threading.Thread(target=respond);t.start()
        self.assertTrue(b.entered.wait(3))
        try:
            self.assertTrue(self.runtime.status()['busy'])
            with self.assertRaises(Busy):self.runtime.respond(request())
            self.assertTrue(self.runtime.cancel())
            with self.assertRaises(Busy):self.runtime.respond(request())
        finally:b.release.set();t.join(3)
        self.assertFalse(t.is_alive());self.assertIsInstance(errors[0],Cancelled)
        self.assertTrue(self.runtime.status()['accepting_requests'])
        b.block=False
        self.assertEqual(self.runtime.respond(request()).answer,ANSWER)

    def test_numerical_failure_marks_failed_and_disposes_models(self):
        b=self.start();b.bad=True
        with self.assertLogs('amharic_s2s.runtime',level='ERROR') as logs:
            with self.assertRaises(InferenceFailure) as exc:self.runtime.respond(request())
        self.assertIn('FloatingPointError',logs.output[0])
        self.assertIn('last stage: llm_start',logs.output[0])
        self.assertNotIn('private transcript',str(exc.exception))
        self.assertTrue(b.closed);self.assertEqual(self.runtime.status()['state'],'failed')
        with self.assertRaises(NotReady):self.runtime.respond(request())
        with self.assertRaises(NotReady):self.runtime.start([request()])

    def test_empty_asr_keeps_models_ready_and_next_turn_succeeds(self):
        b=self.start(); transcribe=b.transcribe; calls=b.calls
        for blank in ('', '   ', None):
            b.transcribe=lambda *args:(blank, '')
            with self.assertRaises(NoSpeech):self.runtime.respond(request())
            self.assertTrue(self.runtime.status()['accepting_requests'])
            self.assertFalse(b.closed)
            self.assertEqual(b.calls,calls)
        def empty(*args):raise NoSpeech('ASR returned empty text.')
        b.transcribe=empty
        with self.assertRaises(NoSpeech):self.runtime.respond(request())
        self.assertTrue(self.runtime.status()['ready'])
        self.assertFalse(b.closed)
        self.assertEqual(b.calls,calls)
        b.transcribe=transcribe
        self.assertEqual(self.runtime.respond(request()).answer,ANSWER)
        self.assertEqual(len(self.created),1)

    def test_close_stops_admission_and_waits_for_active_work(self):
        b=self.start();b.block=True;errors=[]
        def respond():
            try:self.runtime.respond(request())
            except Cancelled:errors.append('cancelled')
        t=threading.Thread(target=respond);t.start();self.assertTrue(b.entered.wait(3))
        closer=threading.Thread(target=self.runtime.close);closer.start()
        # cancel event is also the deterministic signal that close began.
        self.assertTrue(self.runtime._cancel.wait(3))
        self.assertEqual(self.runtime.status()['state'],'closing')
        self.assertFalse(b.closed)
        with self.assertRaises(Busy):self.runtime.respond(request())
        b.release.set();t.join(3);closer.join(3)
        self.assertFalse(t.is_alive());self.assertFalse(closer.is_alive())
        self.assertEqual(errors,['cancelled']);self.assertTrue(b.closed)
        with self.assertRaises(NotReady):self.runtime.respond(request())

    def test_incomplete_answer_does_not_synthesize_and_does_not_poison_runtime(self):
        b=self.start();b.incomplete=True
        b.synthesize=lambda text:self.fail('Must not speak an incomplete answer')
        with self.assertRaises(RejectedAnswer):self.runtime.respond(request())
        self.assertTrue(self.runtime.status()['ready'])

    def test_new_graph_is_visible_not_mislabeled_as_warmed(self):
        b=self.start();b.recompile=True
        r=self.runtime.respond(request())
        self.assertEqual(r.compilation['new_graphs'],1)
        self.assertEqual(r.compilation['timing_status'],'includes_compilation')

    def test_startup_failure_never_advertises_readiness(self):
        def fail(*args):raise RuntimeError('no GPU')
        with VoiceRuntime(Path(self.tmp.name),backend_factory=fail) as runtime:
            with self.assertRaises(InferenceFailure):runtime.start([request()])
            self.assertFalse(runtime.status()['ready'])
            self.assertEqual(runtime.status()['last_error'],'startup_failed')

    def test_status_returns_copy_of_capacity_metadata(self):
        self.start();s=self.runtime.status();s['capacity'][0]['stage']='modified'
        self.assertEqual(self.runtime.status()['capacity'][0]['stage'],'models_loaded')

if __name__=='__main__':unittest.main()

"""ASR empty output and API recovery contracts; no weights or GPU."""
import asyncio
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from amharic_s2s.pipeline_backend import PipelineModels
from amharic_s2s.runtime import NoSpeech, InferenceFailure, VoiceResponse

class ASREmptyTests(unittest.TestCase):
    def test_blank_or_language_tag_is_recoverable_but_numerical_errors_are_not(self):
        # Input is already 16 kHz; resampling must not run in this test.
        modules=patch.dict(sys.modules,{'scipy':SimpleNamespace(),
            'scipy.signal':SimpleNamespace(resample_poly=lambda *a:self.fail('Unexpected resampling'))})
        modules.start();self.addCleanup(modules.stop)
        logits=SimpleNamespace(argmax=lambda _:SimpleNamespace(cpu=lambda:[]))
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None),
            inference_mode=nullcontext,isfinite=lambda _:np.array([True]))
        model=SimpleNamespace(torch=torch,feature=lambda *a,**k:{},
            asr=lambda **k:SimpleNamespace(logits=logits))
        for raw in ('', '  ', '[AMH] ', '[AMH]ሰላም'):
            model.asr_tokenizer=SimpleNamespace(batch_decode=lambda *a,**k:[raw])
            marks=[]
            if raw.endswith('ሰላም'):
                self.assertEqual(PipelineModels.transcribe(model,np.ones(16000),16000,marks.append),('ሰላም',raw))
            else:
                with self.assertRaises(NoSpeech):PipelineModels.transcribe(model,np.ones(16000),16000,marks.append)
            self.assertEqual(marks[-1],'asr_done')
        torch.isfinite=lambda _:np.array([False])
        with self.assertRaises(FloatingPointError):PipelineModels.transcribe(model,np.ones(16000),16000,lambda _:None)

class APIEmptyTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_speech_has_distinct_code_and_next_turn_completes(self):
        from amharic_s2s.api import Settings, TurnService
        # This tests HTTP error mapping. Worker ownership/recovery is covered
        # with real threads by test_runtime.py.
        async def inline(function,*args,**kwargs):return function(*args,**kwargs)
        boundary=patch('amharic_s2s.api.asyncio.to_thread',inline)
        boundary.start();self.addCleanup(boundary.stop)
        def empty(_):raise NoSpeech('private details')
        runtime=SimpleNamespace(respond=empty,status=lambda:{'ready':True})
        service=TurnService(Settings(Path('.'),'x'*40),runtime)
        key=service.submit(None)
        await asyncio.wait_for(service.task,timeout=3)
        self.assertEqual(service.get(key)['error'],'no_speech')
        self.assertIsNone(service.active)
        self.assertIsNone(service.get(key)['audio'])
        runtime.respond=lambda _:VoiceResponse('ok','ሰላም','ሰላም።',np.ones(8000,dtype=np.float32)*.1,8000,{}, {})
        key=service.submit(None)
        await asyncio.wait_for(service.task,timeout=3)
        self.assertEqual(service.get(key)['status'],'completed')
        def fail(_):raise InferenceFailure('private details')
        runtime.respond=fail
        key=service.submit(None)
        await asyncio.wait_for(service.task,timeout=3)
        self.assertEqual(service.get(key)['error'],'inference_failed')

if __name__=='__main__':unittest.main()

"""CPU contract tests for the resident GGUF adapter and Colab lifecycle."""
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from amharic_s2s.artifacts import read_json
from amharic_s2s.runtime import InvalidInput, InferenceFailure, Cancelled, VoiceRuntime
from amharic_s2s.runtime_cpp import CppLLM, CppResidentBackend, AuthenticatedServer, server_arguments

R = Path(__file__).resolve().parents[1]


class Tokenizer:
    length = 100
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages; self.options = kwargs; return 'rendered'
    def __call__(self, *args, **kwargs): return {'input_ids': list(range(self.length))}


class Client:
    def __init__(self, tokenizer): self.tokenizer = tokenizer; self.payload = None; self.reason = 'eos'; self.mismatch = False
    def json(self, path, data):
        return {'tokens': [999] if self.mismatch else list(range(self.tokenizer.length))}
    def request(self, path, data):
        self.payload = data
        rows = [{'content': 'ሰላም።', 'tokens': [12], 'stop': False},
                {'content': '', 'tokens': [1], 'stop': True, 'stop_type': self.reason,
                 'tokens_predicted': 2, 'tokens_evaluated': self.tokenizer.length, 'truncated': False}]
        self.response = io.BytesIO(b''.join(('data: '+json.dumps(r)+'\n').encode() for r in rows))
        return self.response


class CppTests(unittest.TestCase):
    def setUp(self):
        self.llm = CppLLM.__new__(CppLLM)
        self.llm.process = SimpleNamespace(poll=lambda: None)
        self.llm.tokenizer = Tokenizer(); self.llm.client = Client(self.llm.tokenizer)
        self.llm.calls = 0; self.llm.evidence = {'engine': 'llama_cpp', 'offloaded_layers': [43, 43]}
        self.cancel = threading.Event(); self.marks = []
    def run_llm(self): return self.llm.generate([{'role': 'user', 'content': 'ሰላም'}], self.cancel, self.marks.append)
    def test_pretokenized_greedy_stream_and_no_thinking(self):
        r, e = self.run_llm()
        self.assertEqual(r, {'answer': 'ሰላም።', 'stop_reason': 'eos', 'generated_tokens': 2})
        self.assertEqual(self.marks, ['llm_start', 'llm_done'])
        self.assertEqual(self.llm.client.payload['prompt'], list(range(100)))
        self.assertEqual(self.llm.client.payload['n_predict'], 128)
        self.assertFalse(self.llm.client.payload['cache_prompt'])
        self.assertFalse(self.llm.tokenizer.options['enable_thinking'])
        self.assertTrue(self.llm.client.response.closed); self.assertTrue(e['tokenizer_ids_match'])
    def test_prompt_budget_and_tokenizer_mismatch_fail_before_completion(self):
        self.llm.tokenizer.length = 385
        with self.assertRaises(InvalidInput): self.run_llm()
        self.assertIsNone(self.llm.client.payload)
        self.llm.tokenizer.length = 100; self.llm.client.mismatch = True
        with self.assertRaisesRegex(InferenceFailure, 'token IDs differ'): self.run_llm()
        self.assertIsNone(self.llm.client.payload)
    def test_limit_is_not_relabelled_eos_and_cancellation_is_not_a_success(self):
        self.llm.client.reason = 'limit'
        r, _ = self.run_llm(); self.assertEqual(r['stop_reason'], 'limit')
        self.cancel.set()
        with self.assertRaises(Cancelled): self.run_llm()
    def test_cancel_during_stream_closes_connection(self):
        original = self.llm.client.request
        def request(*args):
            response = original(*args); self.cancel.set(); return response
        self.llm.client.request = request
        with self.assertRaises(Cancelled): self.run_llm()
        self.assertTrue(self.llm.client.response.closed); self.assertEqual(self.llm.calls, 0)
    def test_prompt_and_grounding_policy_reused_without_compilation(self):
        b = CppResidentBackend.__new__(CppResidentBackend)
        b.instructions = read_json(R/'configs/runtime_instructions.json'); b.knowledge = None
        b.cpp = self.llm; b.models = SimpleNamespace(llm=self.llm)
        with patch('amharic_s2s.llm_compile.policy', side_effect=AssertionError('must not compile')):
            result, e = b.generate('ስንት ነው', '117 ብር።', True, self.cancel, self.marks.append)
        self.assertEqual(result['grounding']['route'], 'grounded')
        self.assertIn('117 ብር።', self.llm.tokenizer.messages[-1]['content'])
        self.assertEqual(e['engine'], 'llama_cpp'); self.assertEqual(e['timing_status'], 'warmed')
    def test_local_auth_and_server_flags(self):
        server = AuthenticatedServer(1234, 'local-secret')
        seen = []
        server.opener = SimpleNamespace(open=lambda req, **kw: seen.append(req))
        server.request('/tokenize', {'content': 'text'})
        self.assertEqual(seen[0].get_header('Authorization'), 'Bearer local-secret')
        self.assertTrue(seen[0].full_url.startswith('http://127.0.0.1:'))
        args = server_arguments({'server': 'llama-server'}, Path('/tmp/model.gguf'), 1234)
        self.assertEqual(args[args.index('--log-verbosity')+1], '4')
        self.assertIn('--no-context-shift', args); self.assertIn('--no-webui', args)
        self.assertNotIn('local-secret', args)
    def test_runtime_selects_explicit_engine_and_rejects_typo(self):
        with patch.dict('os.environ', {'AMHARIC_LLM_ENGINE': 'llama_cpp'}):
            runtime = VoiceRuntime(R)
        self.assertIs(runtime._factory, CppResidentBackend); runtime.close()
        with patch.dict('os.environ', {'AMHARIC_LLM_ENGINE': 'typo'}), self.assertRaises(ValueError): VoiceRuntime(R)
    def test_resident_loads_asr_and_native_16_step_tts_without_hf_gemma(self):
        from dataclasses import dataclass
        @dataclass
        class Generation:
            num_step: int = 16
        asr = SimpleNamespace(to=lambda _: SimpleNamespace(eval=lambda: None))
        factory = lambda **kw: SimpleNamespace(from_pretrained=lambda *a, **k: kw['value'])
        tf = SimpleNamespace(AutoFeatureExtractor=factory(value='feature'), AutoTokenizer=factory(value='tokenizer'),
                             AutoModelForCTC=factory(value=(asr, {})))
        cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: None,
                               mem_get_info=lambda: (8*2**30, 15*2**30), memory_allocated=lambda: 1,
                               memory_reserved=lambda: 1, empty_cache=lambda: None)
        torch = SimpleNamespace(cuda=cuda, float16='float16')
        cpp = SimpleNamespace(evidence={'engine':'llama_cpp'},logdir=Path('/tmp/unused-test-log'),close=lambda: None)
        tts = SimpleNamespace(settings={'steps':16},generation=Generation(),generate=lambda text: ('wave',24000,{}))
        with patch.dict('sys.modules', {'torch':torch}), \
             patch('amharic_s2s.llm_next.imports',return_value=tf), \
             patch('amharic_s2s.llm_next.NextLLM',side_effect=AssertionError('HF model must not load')), \
             patch('amharic_s2s.runtime_cpp.CppLLM',return_value=cpp), \
             patch('amharic_s2s.runtime_cpp.write_json'), \
             patch('amharic_s2s.tts_backend.LocalTTS',return_value=tts) as local, \
             patch('amharic_s2s.pipeline_v3.IntegratedMixedPrecisionTTS',return_value=tts), \
             patch.dict('os.environ', {'AMHARIC_KNOWLEDGE_INDEX':''}):
            b=CppResidentBackend(R,lambda _:None)
            self.assertEqual(local.call_args.kwargs['steps'],16)
            wave,sr,d=b.synthesize('complete answer')
            self.assertEqual(d['generation_config']['num_step'],16)
            self.assertFalse(d['external_sentence_splitting'])
            self.assertEqual((wave,sr),('wave',24000));b.close()
    def test_notebook_seven_cells_opt_in_process_group_and_partial_export(self):
        nb = read_json(R/'notebooks/ethio_speech_ai_colab.ipynb')
        cells = [c for c in nb['cells'] if c['cell_type'] == 'code']; self.assertEqual(len(cells), 7)
        for c in cells:
            ast.parse(''.join(c['source'])); self.assertEqual(c['outputs'], []); self.assertIsNone(c['execution_count'])
        setup = ''.join(cells[2]['source']); start = ''.join(cells[3]['source']); export = ''.join(cells[5]['source'])
        self.assertIn('AMHARIC_LLM_ENGINE', setup); self.assertIn('llama_cpp', setup)
        self.assertNotIn('bitsandbytes', setup); self.assertNotIn('Gemma4ForCausalLM', setup)
        self.assertIn('start_new_session=True', start); self.assertIn('SESSION_DIR', start)
        self.assertIn('if "BRIDGE" in globals()', export); self.assertIn('api-server.log', export)
        self.assertIn('os.killpg', ''.join(cells[6]['source']))


if __name__ == '__main__': unittest.main()

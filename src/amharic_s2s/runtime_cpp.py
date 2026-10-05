"""Opt-in resident Hohe + pinned llama.cpp Gemma + native 16-step OmniVoice."""
import gc
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time
from types import SimpleNamespace
import urllib.request

from .artifacts import read_json, write_json, digest, object_digest
from .llm_engine_compare import LocalServer, completion_payload, parse_sse, stop_server, tool_environment, verify_layer_offload
from .runtime import Cancelled, InvalidInput, InferenceFailure
from .runtime_backend import ResidentBackend


class AuthenticatedServer(LocalServer):
    def __init__(self, port, token):
        super().__init__(port); self.token = token
    def request(self, path, data=None, timeout=120):
        req = urllib.request.Request(self.base+path,
            data=None if data is None else json.dumps(data, ensure_ascii=False).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+self.token})
        return self.opener.open(req, timeout=timeout)


def server_arguments(installed, model, port):
    return [installed['server'], '--model', str(model), '--host', '127.0.0.1', '--port', str(port),
            '--n-gpu-layers', '99', '--split-mode', 'none', '--ctx-size', '512', '--parallel', '1',
            '--cache-ram', '0', '--no-context-shift', '--fit', 'off', '--perf',
            '--log-verbosity', '4', '--log-colors', 'off', '--no-webui']


class CppLLM:
    """Own one authenticated localhost server. No inference API leaves the VM."""
    def __init__(self, repo, tf):
        from huggingface_hub import hf_hub_download
        self.process = None; self.calls = 0
        self.cfg = read_json(repo/'configs/llm_engine_comparison.json')
        cfg = self.cfg
        if cfg['context_tokens'] != 512 or cfg['response_tokens'] != 128:
            raise ValueError('Use the reviewed 512/128 token configuration')
        work = Path(os.environ.get('AMHARIC_CPP_WORK', '/content/llm-engine-tools'))
        directory = work/cfg['llama_release']; installed = read_json(directory/'verified.json')
        if installed['config_sha256'] != object_digest(cfg):
            raise RuntimeError('Run the pinned llama.cpp setup first; config differs')
        self.logdir = Path(os.environ.get('AMHARIC_CPP_LOG_DIR', '/content/amharic-cpp-runtime'))
        self.logdir.mkdir(parents=True, exist_ok=True)
        candidate = read_json(repo/'configs/llm_next_candidates.json')['candidates']['gemma4_e4b']
        self.tokenizer = tf.AutoTokenizer.from_pretrained(candidate['model_id'], revision=candidate['revision'])
        g = cfg['gguf']; print('Downloading/reusing pinned Gemma Q4_K_M...', flush=True)
        model = Path(hf_hub_download(g['repo'], g['filename'], revision=g['revision']))
        if model.stat().st_size != g['bytes'] or digest(model) != g['sha256']:
            raise RuntimeError('GGUF checksum/size mismatch')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        token = secrets.token_urlsafe(32); self.client = AuthenticatedServer(port, token)
        args = server_arguments(installed, model, port)
        env = tool_environment(directory)
        # Only the generated local credential is needed by the child; do not pass Hub/API keys.
        for k in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'AMHARIC_API_TOKEN'):
            env.pop(k, None)
        env['LLAMA_API_KEY'] = token
        logpath = self.logdir/'llama-server.log'
        try:
            with logpath.open('w') as log:
                # Same process group as the API: notebook shutdown also kills this child.
                self.process = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)
            write_json(self.logdir/'server_pid.json', {'pid': self.process.pid, 'binary': installed['server']})
            deadline = time.monotonic()+300; notice = 0
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError('llama-server exited: '+logpath.read_text(errors='replace')[-5000:])
                try:
                    if self.client.json('/health', timeout=2).get('status') == 'ok': break
                except (OSError, ValueError): pass
                if time.monotonic() > deadline: raise TimeoutError('llama-server load exceeded five minutes')
                if time.monotonic() > notice:
                    print('Waiting for Gemma GGUF loading...', flush=True); notice = time.monotonic()+15
                time.sleep(1)
            self.offload = verify_layer_offload(logpath.read_text(errors='replace'))
            self.evidence = {'engine': 'llama_cpp', 'release': cfg['llama_release'], 'commit': cfg['llama_commit'],
                             'gguf': g, 'offloaded_layers': self.offload, 'context_tokens': 512,
                             'output_token_cap': 128, 'thinking': False, 'prompt_cache': False,
                             'placement_note': 'All reported layers offloaded; CPU-mapped tensors may remain.'}
            write_json(self.logdir/'engine.json', self.evidence)
            print('Gemma ready; GPU layer offload:', self.offload, flush=True)
        except BaseException:
            self.close(); raise

    def generate(self, messages, cancel, mark):
        if cancel.is_set(): raise Cancelled('Request cancelled')
        if self.process.poll() is not None: raise InferenceFailure('Local llama-server stopped')
        mark('llm_start')
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer(rendered, add_special_tokens=False)['input_ids']
        if len(ids)+128 > 512: raise InvalidInput('Prompt exceeds the 384-token budget; no truncation')
        native_ids = self.client.json('/tokenize', {'content': rendered, 'add_special': False, 'parse_special': True})['tokens']
        if ids != native_ids: raise InferenceFailure('Gemma HF/GGUF token IDs differ')
        def checked_lines(response):
            for line in response:
                if cancel.is_set(): raise Cancelled('Request cancelled')
                yield line
        started = time.perf_counter()
        with self.client.request('/completion', completion_payload({'input_ids': ids})) as response:
            r = parse_sse(checked_lines(response), started)
        mark('llm_done')
        if cancel.is_set(): raise Cancelled('Request cancelled')
        if r['tokens_evaluated'] != len(ids): raise InferenceFailure('Server prompt length differs')
        count = r['engine_generated_tokens']
        if type(count) is not int or not 0 < count <= 128:
            raise InferenceFailure('Invalid generated token count')
        self.calls += 1
        return {'answer': r['answer'], 'stop_reason': r['stop_reason'], 'generated_tokens': count}, {
            **self.evidence, 'tokenizer_ids_match': True, 'prompt_tokens': len(ids),
            'server_timings': r['server_timings'], 'first_observed_token_s': r['first_observed_token_s'],
            'compiled_decode_calls': 0, 'new_graphs': None, 'timing_status': 'warmed'}

    def close(self):
        if self.process is not None: stop_server(self.process)


class CppResidentBackend(ResidentBackend):
    """Reuse the existing prompt/grounding policy; substitute the reviewed engine."""
    def __init__(self, repo, capacity):
        from .llm_next import imports
        from .pipeline_backend import PipelineModels
        from .pipeline_v3 import IntegratedMixedPrecisionTTS
        from .tts_backend import LocalTTS, check_loading
        import torch
        tf = imports(preserve_audio=True)
        self.torch = torch; self._notify = capacity; self.models = None; self.cpp = None
        self.instructions = read_json(repo/'configs/runtime_instructions.json'); self.knowledge = None
        if os.environ.get('AMHARIC_KNOWLEDGE_INDEX'):
            from .knowledge import KnowledgeIndex
            self.knowledge = KnowledgeIndex.load(os.environ['AMHARIC_KNOWLEDGE_INDEX'])
            capacity({'stage':'knowledge_loaded','documents':len(self.knowledge.documents),
                      'chunks':len(self.knowledge.chunks),'corpus_sha256':self.knowledge.digest,
                      'version':self.knowledge.version})
        if not torch.cuda.is_available(): raise RuntimeError('Select a CUDA GPU runtime')
        try:
            self._capacity('before_models')
            self.cpp = CppLLM(repo, tf); self._capacity('gemma_cpp_loaded')
            # The ASR method is reused without constructing PipelineModels/NextLLM.
            self.models = SimpleNamespace(torch=torch, llm=self.cpp)
            c = read_json(repo/'configs/asr_candidates.json')['candidates']['hohe']
            opts = {'revision': c['revision'], 'trust_remote_code': False}
            self.models.feature = tf.AutoFeatureExtractor.from_pretrained(c['model_id'], **opts)
            self.models.asr_tokenizer = tf.AutoTokenizer.from_pretrained(c['model_id'], **opts)
            self.models.asr, info = tf.AutoModelForCTC.from_pretrained(c['model_id'], dtype=torch.float16,
                output_loading_info=True, **opts)
            check_loading(info); self.models.asr.to('cuda').eval(); self._capacity('hohe_loaded')
            self.models.transcribe = lambda audio, sr, mark: PipelineModels.transcribe(self.models, audio, sr, mark)
            cat = read_json(repo/'configs/tts_candidates.json')
            self.models.tts = IntegratedMixedPrecisionTTS(LocalTTS(cat['candidates']['omni_amharic'], cat['codec'], steps=16))
            self._capacity('all_models_loaded')
            write_json(self.cpp.logdir/'resident_models.json', {'llm': self.cpp.evidence, 'asr': c,
                'asr_precision': 'float16', 'tts': cat['candidates']['omni_amharic'],
                'tts_settings': self.models.tts.settings, 'external_sentence_splitting': False})
        except BaseException:
            self.close(); raise

    def _generate_model(self, messages, cancel, mark, *, warm=False):
        result, evidence = self.cpp.generate(messages, cancel, mark)
        evidence.update(warmup=warm, timing_status='warmup' if warm else 'warmed')
        return result, evidence

    def synthesize(self, text):
        from dataclasses import asdict
        wave, sr, details = self.models.tts.generate(text)
        return wave, sr, {**details, 'generation_config': asdict(self.models.tts.generation), 'external_sentence_splitting': False}

    def memory(self):
        free, total = self.torch.cuda.mem_get_info()
        return {**super().memory(), 'device_used_gib_now': (total-free)/2**30,
                'note': 'PyTorch peaks exclude the llama.cpp process; device use is a current whole-device reading.'}

    def finish_warmup(self):
        if not self.cpp.calls or self.cpp.process.poll() is not None:
            raise InferenceFailure('Resident llama.cpp warmup not verified')
        self._capacity('ready_after_all_models_warmed')

    def close(self):
        if self.cpp is not None: self.cpp.close()
        self.cpp = None; self.models = None
        gc.collect()
        if self.torch.cuda.is_available(): self.torch.cuda.empty_cache()

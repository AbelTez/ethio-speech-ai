"""The selected models together, with real inference events and live text callbacks."""
import math
import re
import time

from .artifacts import read_json
from .llm_next import imports, NextLLM, NEXT_PROTOCOL
from .runtime import NoSpeech


class PipelineModels:
    def __init__(self, repo, capacity):
        tf = imports(preserve_audio=True)
        import torch
        self.torch = torch
        if not torch.cuda.is_available(): raise RuntimeError('Select a CUDA GPU runtime.')
        capacity('before_models')
        c = read_json(repo/'configs/asr_candidates.json')['candidates']['hohe']
        opts = {'revision':c['revision'], 'trust_remote_code':False}
        self.feature = tf.AutoFeatureExtractor.from_pretrained(c['model_id'], **opts)
        self.asr_tokenizer = tf.AutoTokenizer.from_pretrained(c['model_id'], **opts)
        self.asr, info = tf.AutoModelForCTC.from_pretrained(c['model_id'], dtype=torch.float16,
                                                         output_loading_info=True, **opts)
        from .tts_backend import check_loading, LocalTTS
        check_loading(info)
        self.asr.to('cuda').eval()
        capacity('hohe_loaded')
        c = read_json(repo/'configs/llm_next_candidates.json')['candidates']['gemma4_e4b']
        self.llm = NextLLM(c,'nf4',NEXT_PROTOCOL,shared_audio=True)
        capacity('gemma4_loaded')
        catalog = read_json(repo/'configs/tts_candidates.json')
        self.tts = LocalTTS(catalog['candidates']['omni_amharic'],catalog['codec'],steps=32)
        capacity('all_models_loaded')
        self.llm_stream = torch.cuda.Stream()
        torch.cuda.synchronize()

    def transcribe(self, audio, sr, mark):
        import numpy as np
        from scipy.signal import resample_poly
        torch = self.torch
        mark('asr_prepare_start')
        if sr != 16000:
            gcd = math.gcd(sr,16000)
            audio = resample_poly(audio,16000//gcd,sr//gcd).astype(np.float32)
        inputs = self.feature(audio,sampling_rate=16000,return_tensors='pt')
        inputs = {k:v.to('cuda',dtype=torch.float16 if v.is_floating_point() else v.dtype)
                  for k,v in inputs.items()}
        torch.cuda.synchronize()
        mark('asr_inference_start')
        with torch.inference_mode():
            logits = self.asr(**inputs).logits
            if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite ASR logits.')
            ids = logits.argmax(-1).cpu()
        raw = self.asr_tokenizer.batch_decode(ids,skip_special_tokens=False)[0]
        text = re.sub(r'^\s*\[AMH\]\s*','',raw).strip()
        mark('asr_done')
        if not text: raise NoSpeech('ASR returned empty text.')
        return text, raw

    def generate_text(self, messages, emit, cancel, mark, cap=128):
        """Called on one producer thread; the main thread runs the single TTS consumer."""
        from transformers import StoppingCriteria, StoppingCriteriaList
        b, torch = self.llm, self.torch
        mark('llm_start')
        rendered = b.tokenizer.apply_chat_template(messages,tokenize=False,
                                                   add_generation_prompt=True,enable_thinking=False)
        with torch.cuda.stream(self.llm_stream), torch.inference_mode():
            inputs = b.tokenizer(rendered,add_special_tokens=False,return_tensors='pt',
                                 return_token_type_ids=False).to('cuda')
            count = inputs['input_ids'].shape[-1]
            if count > NEXT_PROTOCOL['max_input_tokens']: raise ValueError('Prompt too long; no truncation.')
            class Stream(b.TextStreamer):
                def on_finalized_text(self,text,stream_end=False):
                    if text: emit(text)
            class Stop(StoppingCriteria):
                def __call__(self,input_ids,scores,**kwargs): return cancel.is_set()
            output = b.model.generate(**inputs,generation_config=b.generation,do_sample=False,
                num_beams=1,num_return_sequences=1,max_new_tokens=cap,
                logits_processor=b.score_checks,stopping_criteria=StoppingCriteriaList([Stop()]),
                streamer=Stream(b.tokenizer,skip_prompt=True,skip_special_tokens=True))
            tokens = output[0,count:].tolist()
            answer = b.tokenizer.decode(tokens,skip_special_tokens=True).strip()
        self.llm_stream.synchronize()
        mark('llm_done')
        reason = ('cancelled' if cancel.is_set() else 'eos' if tokens and tokens[-1] in b.candidate['eos_token_ids']
                  else 'token_limit' if len(tokens)>=cap else 'time_or_other')
        return {'answer':answer,'stop_reason':reason,'generated_tokens':len(tokens),
                'rendered_prompt':rendered,'generated_token_ids':tokens}

    def warmup(self, audio, sr, capacity):
        import threading
        print('Warmup with all models resident (excluded from turn timings).',flush=True)
        self.transcribe(audio,sr,lambda name:None)
        self.generate_text([{'role':'user','content':'በአማርኛ ሰላም በል።'}],
                           lambda text:None,threading.Event(),lambda name:None,cap=8)
        self.tts.seed(42)
        self.tts.generate('ሰላም፣ እንኳን ደህና መጡ።')
        self.torch.cuda.synchronize()
        capacity('all_models_warmed')

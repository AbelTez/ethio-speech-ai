"""Lazy Transformers backend for text-only, single-GPU LLM screening."""
import re
import sys
import time


def text_only_imports(preserve_audio=False):
    # Colab often has incompatible optional vision/audio packages. These subprocesses
    # do text-only inference, so keep those optional integrations out of discovery.
    # This never uninstalls or changes the notebook's packages.
    for name in (('torchvision', 'timm') if preserve_audio else ('torchvision', 'timm', 'librosa')):
        if name in sys.modules and sys.modules[name] is not None:
            raise RuntimeError(f'{name} was already imported. Use the fresh runner subprocess.')
        sys.modules[name] = None


def first_clause(text):
    """A punctuation-ended clause candidate, not a semantic/safety approval."""
    match = re.search(r'[።!?፧]|\.(?=\s|$)', text)
    return text[:match.end()].strip() if match else None


def prompt_messages(dataset, sample):
    content = sample.get('instruction', dataset['instruction'])
    if sample['context']:
        content += '\n\nየተሰጠ መረጃ፦\n' + sample['context']
    content += '\n\nጥያቄ፦\n' + sample['question']
    # One user turn works with Gemma 2's lack of a system role as well as Qwen.
    return [{'role': 'user', 'content': content}]


def diagnostics(text):
    letters = [c for c in text if c.isalpha()]
    ethiopic = sum('\u1200' <= c <= '\u137f' or '\u1380' <= c <= '\u139f'
                   or '\u2d80' <= c <= '\u2ddf' or '\uab00' <= c <= '\uab2f' for c in letters)
    words = text.split()
    trigrams = [tuple(words[i:i+3]) for i in range(max(0, len(words)-2))]
    return {'empty_answer': not text.strip(),
            'ethiopic_letter_ratio': ethiopic / len(letters) if letters else 0,
            'repeated_trigram_fraction': 1-len(set(trigrams))/len(trigrams) if trigrams else 0,
            'reasoning_marker': '<think>' in text or '</think>' in text}


class TransformersLLM:
    def __init__(self, candidate, preset, protocol):
        text_only_imports()
        import torch
        from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                                  Gemma3ForConditionalGeneration, Gemma3nForCausalLM,
                                  Gemma3nTextConfig, GenerationConfig, TextStreamer,
                                  LogitsProcessor, LogitsProcessorList)
        from huggingface_hub import hf_hub_download
        from .artifacts import read_json
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Select a GPU runtime before running a model.')
        self.torch, self.protocol = torch, protocol
        self.candidate, self.preset = candidate, preset
        self.tokenizer = AutoTokenizer.from_pretrained(candidate['model_id'], revision=candidate['revision'])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        compute_dtype = torch.float32 if preset == 'nf4_fp32' else torch.float16
        kwargs = dict(revision=candidate['revision'], dtype=compute_dtype,
                      device_map={'': 0}, attn_implementation='sdpa', low_cpu_mem_usage=True,
                      output_loading_info=True)
        if preset in ('nf4', 'nf4_fp32'):
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=(['lm_head', 'correction_coefs', 'prediction_coefs']
                                       if candidate['loader'] == 'gemma3n_text' else None),
                bnb_4bit_compute_dtype=compute_dtype)
        elif preset != 'fp16':
            raise ValueError('Supported presets: fp16, nf4, nf4_fp32.')
        loader = AutoModelForCausalLM
        if candidate['loader'] == 'gemma3':
            loader = Gemma3ForConditionalGeneration
        elif candidate['loader'] == 'gemma3n_text':
            # With Transformers 4.57.6, omitting a root device entry stages
            # safetensors on CPU for dtype conversion. Both modules still run
            # entirely on GPU; this avoids duplicate large GPU embeddings at load.
            kwargs['device_map'] = {'model': 0, 'lm_head': 0}
            # Load the text submodel explicitly. Vision/audio towers are unused.
            config = read_json(hf_hub_download(candidate['model_id'], 'config.json', revision=candidate['revision']))
            kwargs['config'] = Gemma3nTextConfig(**config['text_config'])
            kwargs['key_mapping'] = {r'^model\.language_model\.': 'model.'}
            loader = Gemma3nForCausalLM
        self.model, info = loader.from_pretrained(candidate['model_id'], **kwargs)
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise RuntimeError(f'Incomplete model weight load: {info}')
        allowed_unused = ('model.vision_tower.', 'model.audio_tower.', 'model.embed_vision.', 'model.embed_audio.')
        unexpected = [k for k in info.get('unexpected_keys', [])
                      if not (candidate['loader'] == 'gemma3n_text' and k.startswith(allowed_unused))]
        if unexpected:
            raise RuntimeError(f'Unrecognized checkpoint weights: {unexpected[:20]}')
        self.loading_info = info
        self.model.eval()
        if any(p.device.type != 'cuda' for p in self.model.parameters()):
            raise RuntimeError('CPU/disk offload is not allowed in this GPU comparison.')
        self.generation = GenerationConfig.from_pretrained(candidate['model_id'], revision=candidate['revision'])
        self.generation.do_sample = False
        self.generation.num_beams = 1
        self.generation.num_return_sequences = 1
        self.generation.temperature = 1.0
        self.generation.top_p = 1.0
        self.generation.top_k = 50
        self.generation.repetition_penalty = 1.0
        self.generation.max_new_tokens = protocol['max_new_tokens']
        self.generation.max_time = protocol['max_time_s']
        self.generation.use_cache = True
        self.generation.pad_token_id = self.tokenizer.pad_token_id
        if self.generation.eos_token_id is None:
            self.generation.eos_token_id = self.tokenizer.eos_token_id
        self.generation.cache_implementation = 'dynamic'
        self.TextStreamer = TextStreamer

        class CheckScores(LogitsProcessor):
            def __call__(self, input_ids, scores):
                # Masked tokens can legitimately have -inf. NaN/+inf or no finite
                # choice is a numerical failure, even when using greedy decoding.
                valid = (~torch.isnan(scores).any() & ~torch.isposinf(scores).any()
                         & torch.isfinite(scores).any(dim=-1).all())
                if not valid.item():
                    raise FloatingPointError(
                        'Invalid generation scores (NaN/+inf or no finite token). '
                        'This configuration failed numerical validation; no answer is accepted.')
                return scores

        self.score_checks = LogitsProcessorList([CheckScores()])

    def generate(self, messages, max_new_tokens=None):
        torch = self.torch
        torch.cuda.synchronize()
        start = time.perf_counter()
        chat_kwargs = {'enable_thinking': False} if 'enable_thinking' in self.candidate else {}
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **chat_kwargs)
        inputs = self.tokenizer(rendered, add_special_tokens=False, return_tensors='pt', return_token_type_ids=False).to('cuda:0')
        input_count = inputs['input_ids'].shape[-1]
        if input_count > self.protocol['max_input_tokens']:
            raise ValueError(f'Prompt has {input_count} tokens, exceeds explicit input limit; no silent truncation.')
        torch.cuda.synchronize()
        prepared = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        backend = self

        class TimedStreamer(self.TextStreamer):
            def __init__(self):
                super().__init__(backend.tokenizer, skip_prompt=True, skip_special_tokens=True)
                self.text = ''
                self.events = {}
                self.clause_text = None
                self.clause_source = None

            def put(self, value):
                if not self.next_tokens_are_prompt and 'first_token_s' not in self.events:
                    torch.cuda.synchronize()
                    self.events['first_token_s'] = time.perf_counter() - prepared
                super().put(value)

            def on_finalized_text(self, text, stream_end=False):
                self.text += text
                elapsed = time.perf_counter() - prepared
                if self.text.strip() and 'first_text_s' not in self.events:
                    self.events['first_text_s'] = elapsed
                clause = first_clause(self.text)
                if clause and self.clause_text is None:
                    self.clause_text = clause
                    self.clause_source = 'punctuation'
                    self.events['first_clause_s'] = elapsed
                if stream_end and self.text.strip() and self.clause_text is None:
                    self.clause_text = self.text.strip()
                    self.clause_source = 'generation_end'
                    self.events['first_clause_s'] = elapsed

        streamer = TimedStreamer()
        cap = max_new_tokens or self.protocol['max_new_tokens']
        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=self.generation,
                                         **getattr(self, 'generation_overrides', {'use_model_defaults': False}),
                                         do_sample=False,
                                         num_beams=1, num_return_sequences=1,
                                         logits_processor=self.score_checks,
                                         max_new_tokens=cap, streamer=streamer)
        torch.cuda.synchronize()
        generated_at = time.perf_counter()
        token_ids = output[0, input_count:].tolist()
        answer = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        raw_answer = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        end = time.perf_counter()
        eos = self.generation.eos_token_id
        eos = eos if isinstance(eos, (list, tuple)) else [eos]
        reason = ('eos' if token_ids and token_ids[-1] in eos else
                  'token_limit' if len(token_ids) >= cap else
                  'time_limit' if generated_at-prepared >= self.protocol['max_time_s'] else 'other')
        metrics = {'input_tokens': input_count, 'generated_tokens': len(token_ids),
                   'prepare_s': prepared-start, 'generation_s': generated_at-prepared,
                   'full_answer_s': end-start, 'final_decode_s': end-generated_at,
                   'tokens_per_s': len(token_ids)/(generated_at-prepared),
                   'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                   'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30, **streamer.events}
        details = {'messages': messages, 'rendered_prompt': rendered, 'answer': answer,
                   'raw_answer': raw_answer, 'generated_token_ids': token_ids,
                   'first_clause': streamer.clause_text, 'clause_source': streamer.clause_source,
                   'stop_reason': reason, 'capped': reason in ('token_limit', 'time_limit'),
                   'diagnostics': diagnostics(answer)}
        return metrics, details

"""Local TTS only. Pin both the speech checkpoint and OmniVoice's audio codec."""
import sys,time
from .artifacts import read_json
from .tts_text import install_cached_romanizer
from .tts_chunking import install_omni_splitter,split_speech_text


def dataset_path(repo,suite):
    if suite not in ('short','long'):raise ValueError('Choose short or long suite.')
    return repo/'data/tts_public'/('prompts.json' if suite=='short' else 'long_prompts.json')


def imports():
    for name in ('torchvision','timm','peft'):
        if name in sys.modules and sys.modules[name] is not None:
            raise RuntimeError('Use a fresh TTS subprocess; optional integrations already imported.')
        sys.modules[name]=None
    import transformers
    if transformers.__version__!='5.17.0':
        raise RuntimeError('Use the dedicated TTS environment with Transformers 5.17.0.')
    return transformers


def check_loading(info):
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs','conversion_errors')):
        raise RuntimeError(f'Checkpoint did not load exactly: {info}')


def preflight(repo,key,suite='short'):
    tf=imports()
    from huggingface_hub import hf_hub_download
    catalog=read_json(repo/'configs/tts_candidates.json');c=catalog['candidates'][key]
    cfg=read_json(hf_hub_download(c['model_id'],'config.json',revision=c['revision']))
    expected='omnivoice' if c['loader']=='omni' else 'vits'
    if cfg['model_type']!=expected:raise ValueError('Wrong checkpoint architecture.')
    tokenizer=tf.AutoTokenizer.from_pretrained(c['model_id'],revision=c['revision'])
    samples=read_json(dataset_path(repo,suite))['samples']
    if c['loader']=='omni':
        from omnivoice import OmniVoice,OmniVoiceGenerationConfig
        from transformers import HiggsAudioV2TokenizerModel
        from omnivoice.utils.lang_map import LANG_NAMES
        # Fail before loading weights if text conservation or word fallback regresses.
        for sample in samples:
            text=sample['text']
            chunks=split_speech_text(text,100,3)
            assert ''.join(chunks)==text
            if len(text)>100:assert len(chunks)>1
            print('Chunk preflight:',sample['id'],len(chunks),'pieces at a 100-character diagnostic target')
        assert 'Amharic' in LANG_NAMES or 'amharic' in LANG_NAMES, 'Check language mapping'
        codec=catalog['codec']
        for name in ('config.json','preprocessor_config.json'):
            hf_hub_download(codec['model_id'],codec['subfolder']+'/'+name,revision=codec['revision'])
    else:
        import uroman  # mandatory: MMS expects romanized Amharic
        assert tokenizer.is_uroman
        probe=samples[0]['text']+' 117'
        native_ids=tokenizer(probe)
        install_cached_romanizer(tokenizer)
        if tokenizer(probe)!=native_ids:raise RuntimeError('Cached MMS preparation changed tokenization.')
        print('Native/cached MMS token IDs match for the preflight probe.')
    for s in samples:
        ids=tokenizer(s['text'])['input_ids']
        if not ids:raise ValueError('Empty text tokenization')
        print(s['id'],len(s['text'].split()),'words |',len(ids),'tokens')
    print('Preflight passed:',c['model_id'],c['revision'],'|',c['license'])
    print('No model weights loaded. First waveform timing is NOT measured by these batch APIs.')


class LocalTTS:
    def __init__(self,candidate,codec,device='cuda',steps=32,reference=None):
        tf=imports()
        import torch
        from huggingface_hub import snapshot_download
        self.torch,self.kind,self.device=torch,candidate['loader'],device
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Choose a GPU runtime.')
        if self.kind=='omni' and device!='cuda':raise ValueError('OmniVoice screening requires CUDA.')
        self.voice_prompt=None
        c=candidate
        path=snapshot_download(c['model_id'],revision=c['revision'],allow_patterns=[
            'config.json','model.safetensors','tokenizer.json','tokenizer_config.json',
            'special_tokens_map.json','vocab.json','chat_template.jinja'])
        if self.kind=='mms':
            if reference:raise ValueError('MMS uses its fixed voice; leave reference settings empty.')
            import uroman
            self.tokenizer=tf.AutoTokenizer.from_pretrained(path,local_files_only=True)
            self.romanizer=install_cached_romanizer(self.tokenizer)
            self.model,info=tf.VitsModel.from_pretrained(path,dtype=torch.float32,output_loading_info=True,local_files_only=True)
            check_loading(info);self.model.to(device).eval()
            self.sampling_rate=int(self.model.config.sampling_rate)
            self.settings={'precision':'float32','voice':'checkpoint default','tokenizer':'cached uroman + unchanged model normalization'}
        else:
            from omnivoice import OmniVoice,OmniVoiceGenerationConfig
            from omnivoice.utils.duration import RuleDurationEstimator
            # train=True skips OmniVoice's unpinned automatic codec download.
            # We immediately enter eval mode and attach the pinned inference components.
            self.model,info=OmniVoice.from_pretrained(path,train=True,dtype=torch.float32,
                attn_implementation='sdpa',output_loading_info=True,local_files_only=True)
            check_loading(info);self.model.to(device).eval()
            self.model.text_tokenizer=tf.AutoTokenizer.from_pretrained(path,local_files_only=True)
            codec_root=snapshot_download(codec['model_id'],revision=codec['revision'],allow_patterns=[codec['subfolder']+'/*'])
            from pathlib import Path
            codec_path=str(Path(codec_root)/codec['subfolder'])
            self.model.audio_tokenizer,codec_info=tf.HiggsAudioV2TokenizerModel.from_pretrained(
                codec_path,dtype=torch.float32,output_loading_info=True,local_files_only=True)
            check_loading(codec_info);self.model.audio_tokenizer.to(device).eval()
            self.model.feature_extractor=tf.AutoFeatureExtractor.from_pretrained(codec_path,local_files_only=True)
            self.model.sampling_rate=int(self.model.feature_extractor.sampling_rate)
            self.model.duration_estimator=RuleDurationEstimator()
            self.sampling_rate=self.model.sampling_rate
            self.generation=OmniVoiceGenerationConfig(num_step=steps,guidance_scale=2.0)
            if reference:
                import soundfile as sf
                wav,sr=sf.read(reference['path'],dtype='float32',always_2d=True)
                duration=len(wav)/sr
                if not 3<=duration<=20:raise ValueError('Use a clean 3–20 second reference with its exact transcript.')
                self.voice_prompt=self.model.create_voice_clone_prompt(
                    ref_audio=(torch.from_numpy(wav.T),sr),ref_text=reference['text'])
            from dataclasses import asdict
            self.settings={'precision':'float32','language':'Amharic','normalize_text':False,
                'voice':'reference' if reference else 'auto','generation':asdict(self.generation),'codec':codec}
            self.chunk_plans=[]
            install_omni_splitter(lambda plan:self.chunk_plans.append(plan))
            self.settings['splitter']='ethiopic-punctuation-with-word-fallback-v3'
            # Observe preparation at the native method boundaries, without replacing synthesis.
            for name in ('_preprocess_all','_prepare_inference_inputs'):
                setattr(self.model,name,self._timed_preparation(getattr(self.model,name)))
            for name in ('_generate_iterative','_generate_chunked'):
                setattr(self.model,name,self._count_calls(name,getattr(self.model,name)))
        self.loading_info={k:list(v) if isinstance(v,set) else v for k,v in info.items()}

    def _timed_preparation(self,method):
        def wrapped(*args,**kwargs):
            self.sync();start=time.perf_counter()
            try:return method(*args,**kwargs)
            finally:
                self.sync();self.preparation_s+=time.perf_counter()-start
        return wrapped

    def _count_calls(self,name,method):
        def wrapped(*args,**kwargs):
            self.native_calls[name]+=1
            return method(*args,**kwargs)
        return wrapped

    def sync(self):
        if self.device=='cuda':self.torch.cuda.synchronize()

    def seed(self,value):
        import random,numpy as np
        random.seed(value);np.random.seed(value);self.torch.manual_seed(value)

    def generate(self,text):
        self.preparation_s=0.0
        self.chunk_plans=[]
        self.native_calls={'_generate_iterative':0,'_generate_chunked':0}
        with self.torch.inference_mode():
            if self.kind=='mms':
                self.sync();start=time.perf_counter()
                inputs=self.tokenizer(text,return_tensors='pt').to(self.device)
                self.sync();self.preparation_s=time.perf_counter()-start
                wav=self.model(**inputs).waveform[0].detach().float().cpu().numpy()
                details={'text_sent_to_tokenizer':text}
            else:
                wav=self.model.generate(text=text,language='Amharic',normalize_text=False,
                    voice_clone_prompt=self.voice_prompt,generation_config=self.generation)[0]
                details={'text_sent_to_generate':text}
        details['preparation_s']=self.preparation_s
        details['native_synthesis_calls']=self.native_calls['_generate_iterative'] if self.kind=='omni' else 1
        details['native_chunked']=bool(self.native_calls['_generate_chunked'])
        details['chunk_plans']=self.chunk_plans
        details['chunk_characters']=[n for p in self.chunk_plans for n in p['chunk_characters']]
        details['chunk_words']=[n for p in self.chunk_plans for n in p['chunk_words']]
        return wav,self.sampling_rate,details

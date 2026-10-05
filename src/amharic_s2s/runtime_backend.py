"""GPU adapter preserving the reviewed v4 inference settings."""
from contextlib import ExitStack
import gc
import hashlib
import os
import time

from .artifacts import read_json
from .runtime import InvalidInput,InferenceFailure


class ResidentBackend:
    def __init__(self,repo,capacity):
        from .pipeline_backend import PipelineModels
        from .pipeline_v3 import IntegratedMixedPrecisionTTS
        from .tts_profile import ProfiledTTS
        from .llm_compile import DecodeCompiler,graphs
        import torch
        import torch._dynamo
        self.torch=torch;self.stack=ExitStack();self.models=None
        self._notify=capacity;self._old_suppress=torch._dynamo.config.suppress_errors
        self.instructions=read_json(repo/'configs/runtime_instructions.json')
        self.knowledge=None
        index_path=os.environ.get('AMHARIC_KNOWLEDGE_INDEX')
        if index_path:
            from .knowledge import KnowledgeIndex
            self.knowledge=KnowledgeIndex.load(index_path)
            capacity({'stage':'knowledge_loaded','documents':len(self.knowledge.documents),
                'chunks':len(self.knowledge.chunks),'corpus_sha256':self.knowledge.digest})
        try:
            torch._dynamo.config.suppress_errors=False
            self.models=PipelineModels(repo,self._capacity)
            self.models.tts=ProfiledTTS(IntegratedMixedPrecisionTTS(self.models.tts))
            self.dispatch=self.stack.enter_context(DecodeCompiler(self.models.llm).installed())
            self.dispatch.variant='compiled';self.initial_graphs=graphs()
        except Exception:
            self.close();raise

    def _capacity(self,stage):
        self.torch.cuda.synchronize();free,total=self.torch.cuda.mem_get_info()
        self._notify({'stage':stage,'free_gib':free/2**30,'total_gib':total/2**30,
            'allocated_gib':self.torch.cuda.memory_allocated()/2**30,
            'reserved_gib':self.torch.cuda.memory_reserved()/2**30})

    def begin_turn(self):
        self.torch.cuda.synchronize();self.torch.cuda.reset_peak_memory_stats()
        self.torch.manual_seed(42);self.models.tts.seed(42)

    def transcribe(self,audio,sr,mark):return self.models.transcribe(audio,sr,mark)

    def generate(self,transcript,context,grounded,cancel,mark,*,warm=False,history=()):
        from .llm_backend import prompt_messages
        from .llm_compile import policy,graphs,PROTOCOL
        retrieval=None; selected=[]; route=None; t0=time.perf_counter()
        requested_grounded=grounded
        index=getattr(self,'knowledge',None)
        if index is not None and getattr(index,'version',None)=='bank-lexical-v2' and not warm and context.strip():
            raise InvalidInput('Bank document mode uses the loaded corpus. Clear the manual reference text.')
        b=self.models.llm
        automatic=index is not None and not warm and not context.strip()
        instruction=self.instructions['grounded' if grounded or context.strip() else 'direct']
        if index is not None and not warm:
            from .knowledge import BANK_INSTRUCTION, NO_EVIDENCE, social_reply, retrieve_for_turn
            instruction=BANK_INSTRUCTION
            social=social_reply(transcript)
            if social:
                return self._fixed_answer(social, 'social_template', mark, index.digest)
            if automatic:
                retrieval=retrieve_for_turn(index,transcript,history)
                hits=retrieval['hits']
                if not hits:
                    return self._fixed_answer(NO_EVIDENCE, 'no_retrieval_match', mark, index.digest, retrieval)
                # The top result must fit as a WHOLE. Never cut away a fee condition
                # or the end of a FAQ answer, or substitute a lower-ranked topic.
                hit=hits[0]
                context='[1] '+hit['title']+'\n'+hit['content']
                selected=[hit]; grounded=True; route='documents'
        current=prompt_messages({'instruction':instruction},{'question':transcript,'context':context})
        retained=[dict(m) for m in history]
        while True:
            messages=retained+current
            rendered=b.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            tokens=b.tokenizer(rendered,add_special_tokens=False)['input_ids']
            if len(tokens)+128<=PROTOCOL['cache_length'] or not retained:break
            retained=retained[2:]
        if len(tokens)+128>PROTOCOL['cache_length']:
            if automatic:
                retrieval['packing_status']='top_passage_exceeds_prompt_budget'
                retrieval['rejected_top']={'id':hits[0].get('id'),'title':hits[0]['title'],
                    'sources':hits[0].get('sources',[]),'prompt_tokens':len(tokens)}
                return self._fixed_answer(NO_EVIDENCE,'retrieval_budget_abstention',mark,index.digest,retrieval)
            raise InvalidInput('Transcript/context exceeds the current 384-token prompt budget; no truncation applied.')
        if retrieval is not None:
            retrieval={k:v for k,v in retrieval.items() if k!='hits'}
            retrieval.update(selected=selected, packing_status='whole_top_passage',
                prepare_ms=(time.perf_counter()-t0)*1000)
        result,evidence=self._generate_model(messages,cancel,mark,warm=warm)
        result["grounding"] = {"route": route or ("grounded" if grounded or context.strip() else "direct"),
            "history_messages_supplied": len(history), "history_messages_used": len(retained),
            "prompt_tokens": len(tokens), "output_token_cap": 128,
            "requested_grounded": requested_grounded, "context_present": bool(context.strip()),
            "context_chars": len(context), "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
            "instruction_version": "bank-rag-v1" if index is not None and not warm else self.instructions.get("version", "legacy"),
            "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest()}
        if retrieval is not None:result['retrieval']=retrieval
        return result,evidence

    def _generate_model(self,messages,cancel,mark,*,warm=False):
        from .llm_compile import policy,graphs,PROTOCOL
        old_calls,old_graphs=self.dispatch.calls,graphs()
        with policy(self.models.llm,'compiled',warm=warm):
            result=self.models.generate_text(messages,lambda text:None,cancel,mark,cap=128)
        calls,new_graphs=self.dispatch.calls-old_calls,graphs()-old_graphs
        if not cancel.is_set() and result.get('generated_tokens',0)>1 and calls<=0:
            raise InferenceFailure('Compiled decode path was not used.')
        evidence={'compiled_decode_calls':calls,'new_graphs':new_graphs,
            'cache_types':sorted(self.dispatch.cache_types),'config':PROTOCOL['compile'].copy(),
            'timing_status':'warmup' if warm else 'includes_compilation' if new_graphs else 'warmed',
            'warmup':warm}
        return result,evidence

    def _fixed_answer(self,answer,route,mark,corpus_hash,retrieval=None):
        mark('llm_start');mark('llm_done')
        result={'answer':answer,'stop_reason':'eos','generated_tokens':0,
            'grounding':{'route':route,'instruction_version':'bank-rag-v1','corpus_sha256':corpus_hash}}
        if retrieval is not None:
            result['retrieval']={k:v for k,v in retrieval.items() if k!='hits'}
            result['retrieval']['selected']=[]
        return result,{'compiled_decode_calls':0,'new_graphs':0,'cache_types':[],
            'timing_status':'not_applicable','generation_bypassed':route,'warmup':False}

    def synthesize(self,text):return self.models.tts.generate(text)

    def memory(self):
        return {'peak_allocated_gib':self.torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_gib':self.torch.cuda.max_memory_reserved()/2**30}

    def finish_warmup(self):
        from .llm_compile import graphs
        if self.dispatch.calls<=0 or graphs()<=self.initial_graphs:
            raise InferenceFailure('Startup did not verify compiled decoding.')
        self._capacity('ready_after_compilation_and_warmup')

    def close(self):
        self.stack.close();self.models=None
        if hasattr(self,'dispatch'):del self.dispatch
        self.torch._dynamo.config.suppress_errors=self._old_suppress
        gc.collect()
        if self.torch.cuda.is_available():self.torch.cuda.empty_cache()

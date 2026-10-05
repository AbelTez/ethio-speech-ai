"""Check prompt and compilation policy at the production adapter boundary."""
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from amharic_s2s.artifacts import read_json
from amharic_s2s.runtime import InvalidInput,InferenceFailure
from amharic_s2s.runtime_backend import ResidentBackend

REPO=Path(__file__).resolve().parents[1]


class Tokenizer:
    length=100
    def apply_chat_template(self,messages,**kwargs):self.messages=messages;return 'rendered'
    def __call__(self,*args,**kwargs):return {'input_ids':list(range(self.length))}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.b=object.__new__(ResidentBackend)
        self.b.instructions=read_json(REPO/'configs/runtime_instructions.json')
        self.b.dispatch=SimpleNamespace(calls=0,cache_types={'StaticCache'})
        self.tokenizer=Tokenizer();self.calls=[];self.do_compile=True
        cfg=SimpleNamespace(max_time=60,max_new_tokens=128)
        llm=SimpleNamespace(tokenizer=self.tokenizer,generation=cfg)
        def generate(messages,emit,cancel,mark,cap):
            self.calls.append((messages,llm.generation.cache_implementation,llm.generation.max_time,cap))
            if self.do_compile:self.b.dispatch.calls+=10
            return {'answer':'answer','stop_reason':'eos','generated_tokens':11}
        self.b.models=SimpleNamespace(llm=llm,generate_text=generate)

    def test_conversational_prompt_and_generation_policy(self):
        samples=read_json(REPO/'data/pipeline_public/questions_v2.json')['samples']
        for s in samples:
            with patch('amharic_s2s.llm_compile.graphs',return_value=2):
                result,evidence=self.b.generate(s['reference'],s['context'],False,threading.Event(),lambda _:None)
            content=self.calls[-1][0][0]['content']
            route='grounded' if s['context'].strip() else 'direct'
            self.assertIn(self.b.instructions[route], content)
            self.assertIn(s['reference'], content)
            self.assertEqual(result['grounding']['route'],route)
            self.assertEqual(result['grounding']['instruction_version'],'conversation-v3')
            self.assertNotIn('context',result['grounding'])
            self.assertEqual(self.calls[-1][1:],('static',60,128))
            self.assertEqual(evidence['compiled_decode_calls'],10)
            self.assertEqual(evidence['timing_status'],'warmed')

    def test_history_is_paired_and_pruned_before_current_question(self):
        history=[{'role':'user','content':'old question'}, {'role':'assistant','content':'old answer'}]
        with patch('amharic_s2s.llm_compile.graphs',return_value=2):
            result,_=self.b.generate('follow up','',False,threading.Event(),lambda _:None,history=history)
        self.assertEqual(self.calls[-1][0][:2],history)
        self.assertEqual(result['grounding']['history_messages_used'],2)
        class BudgetTokenizer(Tokenizer):
            def __call__(self,*args,**kwargs):
                return {'input_ids':list(range(400 if len(self.messages)>1 else 100))}
        self.b.models.llm.tokenizer=BudgetTokenizer()
        with patch('amharic_s2s.llm_compile.graphs',return_value=2):
            result,_=self.b.generate('follow up','',False,threading.Event(),lambda _:None,history=history)
        self.assertEqual(result['grounding']['history_messages_supplied'],2)
        self.assertEqual(result['grounding']['history_messages_used'],0)
        self.assertIn('follow up',self.calls[-1][0][-1]['content'])
        self.assertEqual(len(history),2)

    def test_token_budget_rejects_instead_of_truncating_or_expanding_cache(self):
        self.tokenizer.length=385
        with self.assertRaisesRegex(InvalidInput,'384-token'):
            self.b.generate('question','context',True,threading.Event(),lambda _:None)
        self.assertEqual(self.calls,[])

    def test_explicit_grounding_without_context_keeps_insufficient_information_instruction(self):
        with patch('amharic_s2s.llm_compile.graphs',return_value=2):
            self.b.generate('question','',True,threading.Event(),lambda _:None)
        self.assertIn(self.b.instructions['grounded'],self.calls[0][0][0]['content'])

    def test_missing_compilation_is_failure_and_new_graphs_are_visible(self):
        self.do_compile=False
        with patch('amharic_s2s.llm_compile.graphs',return_value=2):
            with self.assertRaises(InferenceFailure):
                self.b.generate('question','',False,threading.Event(),lambda _:None)
        self.do_compile=True
        with patch('amharic_s2s.llm_compile.graphs',side_effect=[2,3]):
            _,evidence=self.b.generate('question','',False,threading.Event(),lambda _:None)
        self.assertEqual(evidence['timing_status'],'includes_compilation')

    def knowledge(self,hits):
        self.b.knowledge=SimpleNamespace(digest='corpus',search=lambda _:dict(hits=hits,search_ms=1))

    def test_opt_in_document_evidence_is_kept_whole_and_manual_context_overrides(self):
        hit={'title':'Terms','content':'Full answer including its exception.', 'sources':[{'source':'bank.docx'}]}
        self.knowledge([hit])
        with patch('amharic_s2s.llm_compile.graphs',return_value=2):
            r,_=self.b.generate('question','',False,threading.Event(),lambda _:None)
            self.assertIn(hit['content'],self.calls[-1][0][-1]['content'])
            self.assertEqual(r['retrieval']['selected'],[hit])
            self.assertEqual(r['grounding']['route'],'documents')
            self.assertFalse(r['grounding']['requested_grounded'])
            r,_=self.b.generate('question','manual facts',True,threading.Event(),lambda _:None)
        self.assertNotIn('retrieval',r)
        self.assertIn('manual facts',self.calls[-1][0][-1]['content'])
        self.assertNotIn(hit['content'],self.calls[-1][0][-1]['content'])

    def test_abstention_and_exact_social_route_do_not_claim_llm_inference(self):
        self.knowledge([])
        r,e=self.b.generate('question','',False,threading.Event(),lambda _:None)
        self.assertEqual(r['grounding']['route'],'no_retrieval_match')
        self.assertEqual(e['timing_status'],'not_applicable');self.assertEqual(self.calls,[])
        r,e=self.b.generate('አመሰግናለሁ','',False,threading.Event(),lambda _:None)
        self.assertEqual(r['grounding']['route'],'social_template')
        self.assertEqual(r['generated_tokens'],0)

    def test_overbudget_top_document_abstains_without_cutting_or_substitution(self):
        self.knowledge([{'title':'Terms','content':'whole answer'},{'title':'Wrong topic','content':'short'}])
        self.tokenizer.length=385
        r,e=self.b.generate('question','',False,threading.Event(),lambda _:None)
        self.assertEqual(r['grounding']['route'],'retrieval_budget_abstention')
        self.assertEqual(r['retrieval']['packing_status'],'top_passage_exceeds_prompt_budget')
        self.assertEqual(self.calls,[])

    def test_bank_v2_rejects_stale_manual_reference_and_keeps_document_route(self):
        self.knowledge([{'id':'faq','title':'Account','content':'Complete source answer.',
                         'sources':[{'source':'bank.docx'}]}])
        self.b.knowledge.version='bank-lexical-v2'
        with self.assertRaisesRegex(InvalidInput,'Clear the manual reference'):
            self.b.generate('question','Old fictional fee',True,threading.Event(),lambda _:None)
        self.assertEqual(self.calls,[])
        with patch('amharic_s2s.llm_compile.graphs',return_value=2):
            r,_=self.b.generate('question','',False,threading.Event(),lambda _:None)
        self.assertEqual(r['grounding']['route'],'documents')
        self.assertIn('Complete source answer.',self.calls[-1][0][-1]['content'])

if __name__=='__main__':unittest.main()

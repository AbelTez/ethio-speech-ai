import json
from pathlib import Path
import tempfile
import unittest
from amharic_s2s.bank_corpus import compile_bank
from amharic_s2s.knowledge import KnowledgeIndex
from test_knowledge import docx,paragraph

class BankCorpusTests(unittest.TestCase):
    def build(self,body):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        docx(root/'Amharic FAQ/a.docx',body)
        docx(root/'amharic text/b.docx',paragraph('General source with complete conditions.'))
        compile_bank(root,root/'out')
        return json.loads((root/'out/bank_index.json').read_text())
    def test_numbered_questions_keep_answer_steps_and_short_answers(self):
        d=self.build(paragraph('1. How to register?')+paragraph('መልስ፦ Go to branch.')+
            paragraph('1. Bring ID.')+paragraph('2. Sign the form.')+paragraph('*ጥያቄ፦ USSD?')+paragraph('መልስ፦ *123#'))
        q=[c for c in d['chunks'] if c['kind']=='faq']
        self.assertEqual(len(q),2);self.assertIn('2. Sign the form.',q[0]['content'])
        self.assertEqual(q[1]['content'],'*123#')
        self.assertEqual(q[0]['sources'][0]['paragraph_end'],4)
    def test_table_faq_columns_with_empty_serial_cell(self):
        def row(cells):return '<w:tr>'+''.join('<w:tc>'+paragraph(c)+'</w:tc>' for c in cells)+'</w:tr>'
        d=self.build('<w:tbl>'+row(['No','ጥያቄ','መልስ'])+row(['','Can I register?','Yes, with ID.'])+'</w:tbl>')
        c=next(c for c in d['chunks'] if c['kind']=='faq')
        self.assertEqual(c['title'],'Can I register?');self.assertEqual(c['content'],'Yes, with ID.')
    def test_heading_ends_previous_faq_before_next_product(self):
        head='<w:p><w:pPr><w:outlineLvl w:val="2"/></w:pPr><w:r><w:t>New product</w:t></w:r></w:p>'
        d=self.build(paragraph('ጥያቄ፦ First product?')+paragraph('መልስ፦ First answer.')+head+
            paragraph('ጥያቄ፦ Second product?')+paragraph('መልስ፦ Second answer.'))
        first=next(c for c in d['chunks'] if c['title']=='First product?')
        self.assertEqual(first['content'],'First answer.')
    def test_search_does_not_turn_unknown_topic_into_generic_work_match(self):
        d=self.build(paragraph('ጥያቄ፦ የፖስ መሣሪያ እንዴት ይሠራል?')+paragraph('መልስ፦ በካርድ ክፍያ ይቀበላል።'))
        k=KnowledgeIndex(d)
        self.assertEqual(k.search('ሽሮ እንዴት ይሠራል?')['hits'],[])
        self.assertTrue(k.search('የፖስ መሣሪያ እንዴት ይሠራል?')['hits'])

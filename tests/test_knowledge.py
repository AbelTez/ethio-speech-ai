import copy
import json
from pathlib import Path
import tempfile
import unittest
from xml.sax.saxutils import escape
import zipfile

from amharic_s2s.knowledge import compile_documents, docx_blocks, KnowledgeIndex, social_reply, retrieve_for_turn


def paragraph(s):return '<w:p><w:r><w:t>'+escape(s)+'</w:t></w:r></w:p>'


def docx(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('word/document.xml','<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+body+'</w:body></w:document>')


class KnowledgeTests(unittest.TestCase):
    def test_table_rows_survive_alongside_paragraphs_and_preserve_order(self):
        row=lambda a,b:'<w:tr>'+''.join('<w:tc>'+paragraph(s)+'</w:tc>' for s in (a,b))+'</w:tr>'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.docx'
            docx(p,paragraph('Before table')+'<w:tbl>'+row('Product','Amount')+row('Alpha','117')+'</w:tbl>'+paragraph('After table'))
            texts=[b['text'] for b in docx_blocks(p)]
            self.assertEqual(texts,['Before table','Product | Amount','Product | Amount\nAlpha | 117','After table'])

    def test_faq_numbered_answer_not_split_and_pdf_copy_not_double_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'Amharic FAQ/cards.docx'
            docx(p,paragraph('ጥያቄ፦ cardless withdrawal?')+paragraph('መልስ፦ Use mobile transfer.')+paragraph('1. Activate service first.')+paragraph('2. Collect funds at ATM.')+paragraph('ጥያቄ፦ What is Apollo?')+paragraph('መልስ፦ Apollo is a digital service.'))
            p.with_suffix('.pdf').write_bytes(b'paired-copy')
            report=compile_documents(root,root/'index.json')
            data=json.loads((root/'index.json').read_text())
            self.assertEqual(report['docx_documents'],1);self.assertEqual(report['paired_pdfs_skipped'],1)
            self.assertEqual(len(data['chunks']),2)
            self.assertIn('2. Collect funds at ATM.',data['chunks'][0]['content'])
            k=KnowledgeIndex(data)
            self.assertEqual(k.search('cardless withdrawal')['hits'][0]['title'],'cardless withdrawal?')
            self.assertEqual(k.search('unicorn football')['hits'],[])
            altered=copy.deepcopy(data);altered['chunks'][0]['content']='changed fee'
            with self.assertRaisesRegex(ValueError,'hash mismatch'):KnowledgeIndex(altered)

    def test_standalone_pdf_fails_visibly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'amharic text').mkdir();(root/'amharic text/only.pdf').touch()
            with self.assertRaisesRegex(ValueError,'standalone PDFs'):compile_documents(root,root/'out.json')

    def test_whole_greeting_only_and_followup_uses_user_not_generated_facts(self):
        self.assertIsNotNone(social_reply('በጣም አመሰግናለሁ።'))
        self.assertIsNone(social_reply('ሰላም። አፖሎ ምንድን ነው?'))
        class Index:
            def search(self,q):self.query=q;return {'hits':[]}
        idx=Index()
        r=retrieve_for_turn(idx,'ክፍያው ስንት ነው',[{'role':'user','content':'Apollo card'},{'role':'assistant','content':'invented'}])
        self.assertIn('Apollo card',idx.query);self.assertNotIn('invented',idx.query)
        self.assertTrue(r['history_used_for_search'])



if __name__=='__main__':unittest.main()

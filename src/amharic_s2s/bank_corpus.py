"""Build source-preserving bank FAQ/section records on CPU, without model calls."""
import hashlib
import json
from pathlib import Path
import re
from .knowledge import docx_blocks

VERSION='bank-lexical-v2'
QUESTION=re.compile(r'^\s*[*•\-]*\s*(?:ጥያቄ(?:\s*\d+)?|Q\s*\d*)\s*[:፦፡.\-]\s*(.*)',re.I)
ANSWER=re.compile(r'^\s*[*•\-]*\s*(?:መልስ|Answer)\s*[:፦፡]\s*',re.I)

def heading(b):
    text=b['text']
    return (b.get('section_heading',False) or
            (b.get('bold',False) and len(text)<180 and not re.search(r'[?፧።፡፦:]$',text))) and not QUESTION.match(text) and not ANSWER.match(text)

def question(b):
    s=b['text'];m=QUESTION.match(s)
    if m:return m[1]
    # Numbered answer steps remain attached. Only numbered questions split.
    m=re.match(r'^\s*\d+[.)፦]\s*(.*[?፧])\s*$',s)
    return m[1] if m else None

def compile_bank(source,destination):
    source,destination=Path(source),Path(destination)
    documents=[];chunks=[];by_key={};warnings=[];blocks_audit=[];excluded=[]
    for folder in ('Amharic FAQ','amharic text'):
        directory=source/folder
        files=sorted(directory.glob('*.docx'))
        if not files:raise ValueError('Missing DOCX sources: '+str(directory))
        for pdf in directory.glob('*.pdf'):
            if not pdf.with_suffix('.docx').is_file():raise ValueError('Unpaired PDF requires review: '+str(pdf))
        for path in files:
            rel=str(path.relative_to(source));sha=hashlib.sha256(path.read_bytes()).hexdigest()
            blocks=docx_blocks(path)
            if not blocks:raise ValueError('Empty document: '+rel)
            documents.append({'source':rel,'sha256':sha,'paragraphs':len(blocks)})
            blocks_audit.append({'source':rel,'sha256':sha,'blocks':blocks})
            pdf=path.with_suffix('.pdf')
            if pdf.exists():excluded.append({'source':str(pdf.relative_to(source)),
                'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
                'reason':'DOCX chosen as working representation; PDF text equivalence not asserted'})
            section=path.stem;pending=[];title=None;start=None;end=None
            def add(body, title, start, end, kind):
                if not body.strip():return
                # Deduplicate only identical question/section + answer; retain all origins.
                key=hashlib.sha256(re.sub(r'\s+',' ',title+'\n'+body).strip().encode()).hexdigest()
                origin={'source':rel,'document_sha256':sha,'paragraph':start,'paragraph_end':end}
                if key in by_key:by_key[key]['sources'].append(origin);return
                row={'id':key[:20],'title':title,'content':body.strip(),'kind':kind,'sources':[origin]}
                by_key[key]=row;chunks.append(row)
            def flush():
                nonlocal pending,title,start,end
                if pending and (title or pending!=[section]):
                    add('\n'.join(pending),title or section,start,end,'faq' if title else 'section')
                elif title:warnings.append({'source':rel,'block':start,'issue':'Question without answer','question':title})
                pending=[];title=None;start=end=None
            faq=folder=='Amharic FAQ'
            for b in blocks:
                text=b['text'];pos=b['paragraph']
                # Read FAQ tables by columns, preserving empty serial-number cells.
                headers=b.get('table_header',[])
                qcol=next((i for i,v in enumerate(headers) if v.strip() in ('ጥያቄ','Question')),None)
                acol=next((i for i,v in enumerate(headers) if v.strip() in ('መልስ','Answer')),None)
                if faq and qcol is not None and acol is not None:
                    flush()
                    if b['table_row']==0:continue
                    cells=b['table_cells'];q=cells[qcol].strip();a=cells[acol].strip()
                    if q and a:add(a,q,pos,pos,'faq')
                    else:warnings.append({'source':rel,'block':pos,'issue':'Incomplete FAQ table row'})
                    continue
                q=question(b) if faq else None
                if q is not None:
                    flush();title=q;start=end=pos
                    parts=re.split(r'\s*መልስ\s*[:፦፡]\s*',q,maxsplit=1)
                    title=parts[0]
                    if len(parts)>1:pending.append(parts[1])
                elif heading(b):
                    flush();section=text;start=end=pos;pending=[text]
                else:
                    if start is None:start=pos
                    pending.append(ANSWER.sub('',text) if title else text);end=pos
            flush()
    core={'version':VERSION,'documents':documents,'chunks':chunks}
    report={'docx_documents':len(documents),'paired_pdfs_excluded':len(excluded),'chunks':len(chunks),
        'faq_chunks':sum(c['kind']=='faq' for c in chunks),'section_chunks':sum(c['kind']=='section' for c in chunks),
        'deduplicated_origins':sum(len(c['sources'])-1 for c in chunks),
        'warnings':warnings,'source_status':'User supplied working reference; bank approval and currency not verified.',
        'long_records_over_1200_chars':sum(len(c['title'])+len(c['content'])>1200 for c in chunks),
        'long_record_policy':'Keep complete; if it exceeds the runtime token budget, abstain and retain the reason.'}
    data={**core,'corpus_sha256':hashlib.sha256(json.dumps(core,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        'build_report':report}
    destination.mkdir(parents=True,exist_ok=True)
    for name,obj in [('bank_index.json',data),('source_audit.json',{'documents':blocks_audit,'excluded_pdfs':excluded}),('build_report.json',report)]:
        (destination/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(compile_bank(args.source,args.output),ensure_ascii=False,indent=2))

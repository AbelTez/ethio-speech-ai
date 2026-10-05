"""CPU lexical retrieval for the bank-document pilot. Scores are not probabilities."""
from collections import Counter
import math
import re
import time
from .knowledge import normalize,STOP

EXTRA_STOP=set('ይሠራል ይሰራል ላድርግ ማድረግ መሆን ምንድን ምንድነው ምንድናቸው እንዴትስ እንደሚሰራ ምንም በጣም እሺ አመሰግናለሁ እፈልጋለሁ ነገር ነገሮች አለብኝ ማለት እባክዎን'.split())
def terms(text):return [w for w in normalize(text).split() if len(w)>1 and w not in STOP|EXTRA_STOP]
def ngrams(text):return Counter(g for w in terms(text) for g in ([w] if len(w)<3 else [w[i:i+3] for i in range(len(w)-2)]))
def intent(text):
    # Small transparent language hints; no facts or expected answers are injected.
    if re.search(r'መዝገብ|መዝገባ|ልመዝገብ|register|registration',text,re.I):return 'registration'
    if re.search(r'ምንድን|ምንድነው|what is|what are',text,re.I):return 'definition'
    return None

class BankSearch:
    def __init__(self,chunks):
        self.chunks=chunks;self.lex=[];self.char=[];self.titlegrams=[];df=Counter();gf=Counter()
        for c in chunks:
            words=Counter(terms(c['content']));words.update(terms(c['title'])*3)
            self.lex.append(words);df.update(words.keys())
            chars=ngrams(c['title'] if c['kind']=='faq' else c['title']+' '+c['content'])
            self.char.append(chars);gf.update(chars.keys());self.titlegrams.append(ngrams(c['title']))
        n=len(chunks);self.idf={k:math.log(1+(n-v+.5)/(v+.5)) for k,v in df.items()}
        self.gidf={k:math.log((n+1)/(v+1))+1 for k,v in gf.items()}
        self.lengths=[sum(d.values()) for d in self.lex];self.avg=sum(self.lengths)/n
        self.norm=[]
        for d in self.char:
            for k in d:d[k]=(1+math.log(d[k]))*self.gidf[k]
            self.norm.append(math.sqrt(sum(v*v for v in d.values())) or 1)
        self.intents=[intent(c['title']) for c in chunks]
    def search(self,query,top_k=4):
        started=time.perf_counter();q=Counter(terms(query));allgrams=ngrams(query)
        cg={k:(1+math.log(v))*self.gidf[k] for k,v in allgrams.items() if k in self.gidf}
        # Retain unseen query grams in the denominator so unknown topics cannot
        # disappear and leave only a generic 'how does it work' match.
        unknown=sum(v for k,v in allgrams.items() if k not in self.gidf)
        qnorm=math.sqrt(sum(v*v for v in cg.values())+unknown*math.log(len(self.chunks)+1)**2) or 1
        found=[];qi=intent(query)
        for i,d in enumerate(self.lex):
            bm=sum(self.idf[t]*d[t]*2.5/(d[t]+1.5*(.25+.75*self.lengths[i]/self.avg)) for t in q if t in d)
            cosine=sum(v*self.char[i].get(k,0) for k,v in cg.items())/(qnorm*self.norm[i])
            if not ((bm>0 and cosine>=.12) or cosine>=.42):continue
            bm_scaled=bm/(bm+10)
            same_intent=bool(qi and self.intents[i]==qi)
            score=.55*cosine+.45*bm_scaled+(.25 if same_intent else 0)
            found.append({**self.chunks[i],'bm25':bm,'character_cosine':cosine,'rank_score':score,'intent_match':same_intent})
        found.sort(key=lambda h:(-h['rank_score'],h['id']))
        return {'hits':found[:top_k],'search_ms':(time.perf_counter()-started)*1000,
                'method':'CPU BM25 + character TF-IDF; FAQ title weighting and intent hints',
                'query_intent':qi,'threshold_status':'heuristic, not calibrated confidence'}

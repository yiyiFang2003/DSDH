import re, jieba, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
text=open('corpus.txt').read(); parts=re.split(r'\n(?=[\u4e00-\u9fff]+篇\n)',text)
docs=[]; names=[]
for p in parts:
 lines=p.splitlines(); names.append(lines[0]); docs.append(''.join(lines[1:]))
stop=set('子曰孔子曰也之而其不以为与是有者乎焉矣也者哉于则而后吾我之乎何诸其人也'.split())
def tok(s): return [w for w in jieba.lcut(s) if re.fullmatch(r'[\u4e00-\u9fff]{2,}',w) and w not in stop]
X=TfidfVectorizer(tokenizer=tok,token_pattern=None,min_df=1).fit_transform(docs)
for k in [3,4]:
 lda=LatentDirichletAllocation(n_components=k,random_state=42,learning_method='batch',max_iter=100).fit(X)
 print('K',k)
 for i, row in enumerate(lda.components_):
  terms=np.array(TfidfVectorizer(tokenizer=tok,token_pattern=None,min_df=1).fit(docs).get_feature_names_out())
  print(i, ' '.join(terms[row.argsort()[-8:][::-1]]))
 print('perplexity',lda.perplexity(X))

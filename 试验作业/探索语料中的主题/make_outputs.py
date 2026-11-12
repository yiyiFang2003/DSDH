import re,json, jieba, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
text=open('corpus.txt').read(); parts=re.split(r'\n(?=[\u4e00-\u9fff]+篇\n)',text)
names=[]; docs=[]
for p in parts:
 ls=p.splitlines(); names.append(ls[0]); docs.append(''.join(ls[1:]))
base_stop=set('子曰孔子曰也之而其不以为与是有者乎焉矣也者哉于则而后吾我何诸'.split())
def tok(s,stop=base_stop): return [w for w in jieba.lcut(s) if re.fullmatch(r'[\u4e00-\u9fff]{2,}',w) and w not in stop]
vec=TfidfVectorizer(tokenizer=tok,token_pattern=None,min_df=1); X=vec.fit_transform(docs); terms=np.array(vec.get_feature_names_out())
rows=[]
for i,n in enumerate(names):
 for j in np.argsort(X[i].toarray()[0])[-10:][::-1]: rows.append({'document':n,'keyword':terms[j],'tfidf':round(float(X[i,j]),6)})
pd.DataFrame(rows).to_csv('tfidf_keywords.csv',index=False,encoding='utf-8-sig')
results={"corpus":"《论语》十篇","documents":names,"preprocessing":{"tokenizer":"jieba","stopwords":sorted(base_stop),"min_token_length":2},"models":{}}
for k in [3,4,5]:
 lda=LatentDirichletAllocation(n_components=k,random_state=42,learning_method='batch',max_iter=100).fit(X)
 topics=[]
 for row in lda.components_:
  ids=row.argsort()[-10:][::-1]; topics.append([{'word':terms[j],'weight':round(float(row[j]),6)} for j in ids])
 results['models'][str(k)]={'random_state':42,'max_iter':100,'perplexity':round(float(lda.perplexity(X)),6),'topics':topics}
# sensitivity across stopword variants and seeds
sens=[]
for stopname,stop in [('base',base_stop),('base_plus_names',base_stop|set(names))]:
 v=TfidfVectorizer(tokenizer=lambda s:tok(s,stop),token_pattern=None,min_df=1); Z=v.fit_transform(docs); tt=np.array(v.get_feature_names_out())
 for k in [3,4]:
  for seed in [0,42,99]:
   m=LatentDirichletAllocation(n_components=k,random_state=seed,learning_method='batch',max_iter=100).fit(Z)
   sens.append({'stopword_setting':stopname,'k':k,'random_state':seed,'perplexity':round(float(m.perplexity(Z)),6),'topics':[[''.join(tt[j] for j in row.argsort()[-8:][::-1])] for row in m.components_]})
json.dump({'settings':{'k_values':[3,4],'seeds':[0,42,99]},'runs':sens},open('lda_sensitivity.json','w'),ensure_ascii=False,indent=2)
json.dump(results,open('results.json','w'),ensure_ascii=False,indent=2)
# plot as standalone SVG
m=LatentDirichletAllocation(n_components=3,random_state=42,learning_method='batch',max_iter=100).fit(X)
W=900; H=420; svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">','<rect width="100%" height="100%" fill="white"/>','<text x="20" y="28" font-size="20">《论语》十篇 LDA 主题高权重词（K=3）</text>']
for i,row in enumerate(m.components_):
 ids=row.argsort()[-8:][::-1]; x=20+i*295
 svg.append(f'<text x="{x}" y="58" font-size="16">主题 {i+1}</text>')
 mx=max(row[ids])
 for q,j in enumerate(ids):
  y=78+q*38; bw=180*row[j]/mx
  svg += [f'<text x="{x}" y="{y+14}" font-size="13">{terms[j]}</text>',f'<rect x="{x+70}" y="{y}" width="{bw:.1f}" height="20" fill="#4C78A8"/>',f'<text x="{x+75+bw:.1f}" y="{y+14}" font-size="11">{row[j]:.1f}</text>']
svg.append('</svg>'); open('lda_topics.svg','w').write(''.join(svg))

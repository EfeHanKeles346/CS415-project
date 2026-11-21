"""Reproduce the progress-report audit and validation-only baseline experiments."""
from pathlib import Path
import gzip, hashlib, html, json, platform, re, subprocess, time, warnings
from collections import Counter
import numpy as np
import pandas as pd
import scipy, sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix, classification_report
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2), encoding='utf-8')

def normalize(text):
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]*>', ' ', html.unescape(text))).strip().lower()

def records(path):
    record = {}
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as stream:
        for line in stream:
            line = line.rstrip('\n\r')
            if not line:
                if record:
                    yield record
                    record = {}
            elif ': ' in line:
                key, value = line.split(': ', 1)
                record[key] = value
        if record:
            yield record

def split_product(product):
    # Fixed salt; partitioning is independent of labels and source ordering.
    u = int(hashlib.sha256(('cs415-v1:' + product).encode()).hexdigest()[:16], 16) / 2**64
    return 'train' if u < .70 else 'validation' if u < .85 else 'test'

def metrics(y, pred):
    return dict(accuracy=float(accuracy_score(y,pred)),
                macro_f1=float(f1_score(y,pred,labels=[1,2,3,4,5],average='macro',zero_division=0)),
                mae=float(mean_absolute_error(y,pred)),
                large_error_rate=float(np.mean(np.abs(y-pred)>=2)))

def main():
    start = time.perf_counter()
    source = DATA / 'finefoods.txt.gz'
    sha = hashlib.file_digest(source.open('rb'), 'sha256').hexdigest()
    rows=[]
    for idx, r in enumerate(records(source)):
        try:
            rating = float(r.get('review/score','nan'))
        except ValueError:
            rating = float('nan')
        text = normalize(r.get('review/text',''))
        rows.append((idx, r.get('product/productId',''), rating, text))
    df=pd.DataFrame(rows,columns=['source_index','product','rating','text'])
    del rows
    raw_counts=df.rating.value_counts().sort_index().to_dict()
    valid=(df.text.str.len()>0)&df.rating.isin([1,2,3,4,5])&(df['product'].str.len()>0)
    audit=dict(source_url='https://snap.stanford.edu/data/finefoods.txt.gz',
        source_sha256=sha, compressed_bytes=source.stat().st_size,
        raw_rows=len(df),raw_products=int(df['product'].nunique()),raw_rating_counts=raw_counts,
        empty_texts=int((df.text.str.len()==0).sum()),invalid_ratings=int((~df.rating.isin([1,2,3,4,5])).sum()),
        missing_product_ids=int((df['product'].str.len()==0).sum()),invalid_rows=int((~valid).sum()),
        normalization='HTML unescape, strip tags, whitespace collapse, lowercase; no stemming or stopword removal.')
    df=df.loc[valid].copy()
    grouped=df.groupby('text',sort=False).rating.nunique()
    conflicts=set(grouped[grouped>1].index)
    audit['conflicting_text_groups']=len(conflicts)
    audit['conflicting_rows_removed']=int(df.text.isin(conflicts).sum())
    audit['repeated_text_rows_before_conflict_removal']=int(df.duplicated('text').sum())
    df=df.loc[~df.text.isin(conflicts)].copy()
    audit['same_label_duplicate_rows_removed']=int(df.duplicated('text').sum())
    df=df.drop_duplicates('text',keep='first').reset_index(drop=True)
    df['rating']=df.rating.astype(int)
    df['split']=df['product'].map(split_product)
    df['words']=df.text.str.split().str.len()
    df['text_sha256']=df.text.map(lambda x:hashlib.sha256(x.encode()).hexdigest())
    audit['clean_rows']=len(df)
    audit['clean_products']=int(df['product'].nunique())
    audit['clean_rating_counts']=df.rating.value_counts().sort_index().to_dict()
    audit['word_length_quantiles']={str(q):float(df.words.quantile(q)) for q in [.5,.9,.95,.99]}
    audit['fraction_over_256_words']=float((df.words>256).mean())
    audit['splits']={}
    for split, part in df.groupby('split'):
        audit['splits'][split]=dict(rows=len(part),products=int(part['product'].nunique()),
            rating_counts=part.rating.value_counts().sort_index().to_dict())
    audit['cross_split_product_overlap']=0
    audit['cross_split_normalized_text_overlap']=0
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        pa,pb=df[df.split==a],df[df.split==b]
        assert not set(pa['product']) & set(pb['product'])
        assert not set(pa.text_sha256) & set(pb.text_sha256)
    df[['source_index','product','rating','split','text_sha256','words']].to_csv(DATA/'split_manifest.csv.gz',index=False)
    df.to_pickle(DATA/'clean_reviews.pkl')
    dump('audit.json',audit)
    print('AUDIT',json.dumps(audit),flush=True)

    train=df[df.split=='train']; val=df[df.split=='validation']
    ytr=train.rating.to_numpy(); yval=val.rating.to_numpy()
    majority=int(train.rating.mode().iloc[0])
    baseline=metrics(yval,np.full(len(yval),majority))
    baseline.update(model='Majority',seed=None,fit_seconds=0.0,epochs=0)
    results=[baseline]
    vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=60000,
        sublinear_tf=True, dtype=np.float32, lowercase=False)
    t=time.perf_counter()
    xtr=vectorizer.fit_transform(train.text)
    xval=vectorizer.transform(val.text)
    vector_seconds=time.perf_counter()-t
    print('FEATURES',xtr.shape,'seconds',vector_seconds,flush=True)
    config=dict(seeds=[17,29,43],alpha=1e-5,max_iter=30,tol=1e-4,
        ngram_range=[1,2],min_df=3,max_features=60000,actual_features=xtr.shape[1],
        sublinear_tf=True,normalization='L2',word_token_pattern=vectorizer.token_pattern,
        train_rows=len(train),validation_rows=len(val),vectorization_seconds=vector_seconds,
        variants=['unweighted','balanced'],evaluation_split='validation',test_predictions_computed=False,
        classifier='SGDClassifier: one-vs-rest binary logistic losses; not a five-class softmax CE model',
        weighting='balanced: n_train / (5 * training count for class)',
        python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),machine=platform.machine(),
        cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
        thread_limit=1)
    dump('experiment_config.json',config)
    for variant in ['unweighted','balanced']:
        for seed in config['seeds']:
            model=SGDClassifier(loss='log_loss',alpha=config['alpha'],max_iter=config['max_iter'],
                tol=config['tol'],class_weight=None if variant=='unweighted' else 'balanced',
                random_state=seed,n_jobs=1)
            t=time.perf_counter()
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(xtr,ytr)
            fit=time.perf_counter()-t
            pred=model.predict(xval)
            item=metrics(yval,pred)
            item.update(model='TFIDF-'+variant,seed=seed,fit_seconds=fit,epochs=int(model.n_iter_),
                warnings=[str(w.message) for w in caught])
            results.append(item)
            pd.DataFrame({'source_index':val.source_index,'truth':yval,'prediction':pred}).to_csv(
                OUT/f'predictions_{variant}_{seed}.csv.gz',index=False)
            dump(f'confusion_{variant}_{seed}.json',confusion_matrix(yval,pred,labels=[1,2,3,4,5]).tolist())
            dump(f'per_class_{variant}_{seed}.json',classification_report(yval,pred,labels=[1,2,3,4,5],output_dict=True,zero_division=0))
            dump('baseline_runs.json',results)
            print('RUN',json.dumps(item),flush=True)
    pd.DataFrame(results).to_csv(OUT/'baseline_runs.csv',index=False)
    summary={}
    for name in ['Majority','TFIDF-unweighted','TFIDF-balanced']:
        items=[r for r in results if r['model']==name]
        summary[name]={key:{'mean':float(np.mean([r[key] for r in items])),
            'sd':float(np.std([r[key] for r in items],ddof=1)) if len(items)>1 else 0.0}
            for key in ['accuracy','macro_f1','mae','large_error_rate','fit_seconds']}
    dump('baseline_summary.json',summary)
    dump('run_completion.json',dict(completed=True,total_seconds=time.perf_counter()-start,
        test_evaluated=False))
    print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':
    main()

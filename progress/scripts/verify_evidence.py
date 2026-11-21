"""Independently recompute reported metrics from the saved predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'
a=json.loads((R/'audit.json').read_text())
runs=json.loads((R/'baseline_runs.json').read_text())
manifest=pd.read_csv(ROOT/'data/split_manifest.csv.gz')
assert len(manifest)==a['clean_rows']==393001
assert a['raw_rows']-a['invalid_rows']-a['conflicting_rows_removed']-a['same_label_duplicate_rows_removed']==len(manifest)
assert manifest.text_sha256.nunique()==len(manifest)
assert manifest.groupby('product').split.nunique().max()==1
v=manifest[manifest.split=='validation']
assert len(v)==56791
assert set(v.rating)=={1,2,3,4,5}
for row in runs:
    if row['model']=='Majority':
        truth=v.rating.to_numpy();pred=np.full(len(v),5)
    else:
        variant=row['model'].removeprefix('TFIDF-');seed=row['seed']
        p=pd.read_csv(R/f'predictions_{variant}_{seed}.csv.gz')
        assert np.array_equal(p.source_index.to_numpy(),v.source_index.to_numpy())
        assert np.array_equal(p.truth.to_numpy(),v.rating.to_numpy())
        truth=p.truth.to_numpy();pred=p.prediction.to_numpy()
        assert set(pred).issubset({1,2,3,4,5})
        cm=confusion_matrix(truth,pred,labels=[1,2,3,4,5])
        assert np.array_equal(cm,np.asarray(json.loads((R/f'confusion_{variant}_{seed}.json').read_text())))
    computed=dict(accuracy=accuracy_score(truth,pred),macro_f1=f1_score(truth,pred,labels=[1,2,3,4,5],average='macro',zero_division=0),
        mae=mean_absolute_error(truth,pred),large_error_rate=np.mean(np.abs(truth-pred)>=2))
    for key,value in computed.items():
        assert abs(value-row[key])<1e-12,(row['model'],key,value,row[key])
assert not list(R.glob('*test*prediction*'))
record=dict(verified=True,rows_checked=len(manifest),prediction_files_checked=6,
    metrics_recomputed=28,product_overlap=0,normalized_text_overlap=0,
    final_test_predictions_present=False)
(R/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))

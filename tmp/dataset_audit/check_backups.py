import subprocess, zlib, json, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def audit(category):
 url=f'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{category}.jsonl.gz'
 h=subprocess.run(['curl','-sSI','--max-time','20',url],capture_output=True,text=True)
 size=re.search(r'content-length:\s*(\d+)',h.stdout,re.I)
 r=subprocess.run(['curl','-sS','--fail','--max-time','30','--range','0-1048575','--max-filesize','1048576',url],capture_output=True)
 if r.returncode: return category,{'error':r.stderr.decode()}
 raw=zlib.decompressobj(16+zlib.MAX_WBITS).decompress(r.stdout).decode('utf-8',errors='replace')
 rows=[json.loads(line) for line in raw.split('\n')[:-1] ][:1000]
 texts=[re.sub(r'\s+',' ',str(x.get('text',''))).strip().lower() for x in rows]
 return category,{'compressed_file_bytes':int(size.group(1)) if size else None,'downloaded_bytes':len(r.stdout),'checked_records':len(rows),'blank_texts':sum(not t for t in texts),'invalid_ratings':sum(x.get('rating') not in [1,2,3,4,5] for x in rows),'repeated_normalized_text_rows':len(texts)-len(set(texts)),'texts_under_3_words':sum(len(t.split())<3 for t in texts),'missing_product_ids':sum(not x.get('parent_asin') for x in rows),'rating_counts':dict(Counter(str(x.get('rating')) for x in rows))}
with ThreadPoolExecutor(max_workers=3) as pool:
 results=dict(pool.map(audit,['Amazon_Fashion','Appliances','Handmade_Products']))
u='https://datasets-server.huggingface.co/rows?dataset=Yelp%2Fyelp_review_full&config=yelp_review_full&split=train&offset=0&length=100'
r=subprocess.run(['curl','-sS','--fail','--max-time','30','--max-filesize','1048576',u],capture_output=True)
if r.returncode: results['Yelp Review Full']={'error':r.stderr.decode()}
else:
 data=json.loads(r.stdout)
 rows=[x['row'] for x in data.get('rows',[])]
 texts=[re.sub(r'\s+',' ',x.get('text','')).strip().lower() for x in rows]
 results['Yelp Review Full']={'downloaded_bytes':len(r.stdout),'checked_records':len(rows),'blank_texts':sum(not t for t in texts),'invalid_encoded_labels':sum(x.get('label') not in [0,1,2,3,4] for x in rows),'repeated_normalized_text_rows':len(texts)-len(set(texts)),'note':'Labels 0..4 correspond to stars 1..5; no product/business ID in this version.'}
report={'sampling':'Non-random prefix samples; basic structure check only, not full quality certification.','datasets':results}
Path('tmp/dataset_audit/backup_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))

import subprocess, zlib, json, re
from collections import Counter
from pathlib import Path

sources = {
 'Fine Foods': 'https://snap.stanford.edu/data/finefoods.txt.gz',
 'All Beauty': 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz'
}
report = {'sampling': 'First 1000 complete records in a bounded file prefix, not random or representative.', 'datasets': {}}
for name,url in sources.items():
 r=subprocess.run(['curl','-sS','-L','--fail','--max-time','30','--range','0-1048575','--max-filesize','1048576',url],capture_output=True)
 if r.returncode:
  report['datasets'][name]={'error':r.stderr.decode()}; continue
 raw=zlib.decompressobj(16+zlib.MAX_WBITS).decompress(r.stdout).decode('utf-8',errors='replace')
 records=[]
 if name=='All Beauty':
  for line in raw.split('\n')[:-1]:
   try: records.append(json.loads(line))
   except json.JSONDecodeError: pass
  records=records[:1000]
  rows=[(str(x.get('text','')),x.get('rating'),x.get('parent_asin')) for x in records]
 else:
  for block in raw.split('\n\n')[:-1]:
   fields=dict(line.split(': ',1) for line in block.splitlines() if ': ' in line)
   if 'review/text' in fields: records.append(fields)
  records=records[:1000]
  rows=[(x.get('review/text',''),float(x['review/score']) if x.get('review/score') else None,x.get('product/productId')) for x in records]
 texts=[re.sub(r'\s+',' ',t).strip().lower() for t,_,_ in rows]
 report['datasets'][name]={
  'source':url, 'downloaded_bytes':len(r.stdout),'sample_size':len(rows),
  'blank_texts':sum(not t for t in texts),
  'invalid_ratings':sum(y not in [1,2,3,4,5] for _,y,_ in rows),
  'rating_counts':dict(Counter(str(y) for _,y,_ in rows)),
  'repeated_normalized_text_rows':len(texts)-len(set(texts)),
  'texts_under_3_words':sum(len(t.split())<3 for t in texts),
  'missing_product_ids':sum(not pid for _,_,pid in rows),
  'distinct_products':len(set(pid for _,_,pid in rows)),
  'scope_limit': 'Checks schema and basic cleanliness only; does not verify language, authenticity or rating correctness.'}
Path('tmp/dataset_audit/prefix_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))

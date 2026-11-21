from pathlib import Path
import subprocess, re
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
ids={'pang2005':'P05-1015','wang2012':'P12-2018','kim2014':'D14-1181','glove2014':'D14-1162','bert2019':'N19-1423','chen2021':'2021.eval4nlp-1.19','oll2022':'2022.coling-1.407','mll2024':'2024.findings-acl.320','blitzer2007':'P07-1056'}
def get(item):
    key, anthology=item
    url='https://aclanthology.org/'+anthology+'.bib'
    text=subprocess.check_output(['curl','-fsSL','--retry','2',url],text=True)
    text=text[:text.index('{')+1]+key+text[text.index(','):]
    return text
with ThreadPoolExecutor(max_workers=5) as pool:
    entries=list(pool.map(get,ids.items()))
extra=r'''
@article{survey2016,
 author={Guti{\'e}rrez, Pedro Antonio and P{\'e}rez-Ortiz, Mar{\'i}a and S{\'a}nchez-Monedero, Javier and Fern{\'a}ndez-Navarro, Francisco and Herv{\'a}s-Mart{\'i}nez, C{\'e}sar},
 title={Ordinal Regression Methods: Survey and Experimental Study},
 journal={IEEE Transactions on Knowledge and Data Engineering}, volume={28}, number={1}, pages={127--146},year={2016}, doi={10.1109/TKDE.2015.2457911}}
@article{coral2020,
 author={Cao, Wenzhi and Mirjalili, Vahid and Raschka, Sebastian},
 title={Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation},
 journal={Pattern Recognition Letters},volume={140},pages={325--331},year={2020},doi={10.1016/j.patrec.2020.11.008}}
@article{dang2021,
 author={Dang, Chien N. and Moreno-Garc{\'i}a, Mar{\'i}a N. and De la Prieta, Fernando},
 title={An Approach to Integrating Sentiment Analysis into Recommender Systems},
 journal={Sensors},volume={21},number={16},pages={5666},year={2021},doi={10.3390/s21165666}}
@misc{distil2019,
 author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
 title={{DistilBERT}, a Distilled Version of {BERT}: Smaller, Faster, Cheaper and Lighter},
 howpublished={arXiv:1910.01108},year={2019},url={https://arxiv.org/abs/1910.01108}}
@misc{finefoods,
 author={{Stanford Network Analysis Project}},title={{Amazon Fine Foods} Reviews},
 url={https://snap.stanford.edu/data/web-FineFoods.html},note={Dataset; accessed Nov. 16, 2026}}
'''
raw='\n\n'.join(entries)+extra
(ROOT/'sources/references_original.bib').write_text(raw)
venues={'pang2005':'Proc. ACL', 'wang2012':'Proc. ACL', 'kim2014':'Proc. EMNLP', 'glove2014':'Proc. EMNLP', 'bert2019':'Proc. NAACL-HLT', 'chen2021':'Proc. Eval4NLP', 'oll2022':'Proc. COLING', 'mll2024':'Findings of ACL', 'blitzer2007':'Proc. ACL'}
clean=[]
for key,entry in zip(ids,entries):
    entry=re.sub(r'    (editor|publisher|address|month)\s*=\s*(?:"[^"]*"|[^,\n]+),?\n','',entry,flags=re.S)
    entry=re.sub(r'booktitle\s*=\s*"[^"]*"', 'booktitle = "'+venues[key]+'"',entry)
    if key=='mll2024':
        entry=re.sub(r'author\s*=\s*"[^"]*"', 'author = "Kasa, Siva Rajesh and Goel, Aniket and Gupta, Karan and Roychowdhury, Sumegh and Bhanushali, Anish and Pattisapu, Nikhil and Srinivasa Murthy, Prasanna"',entry)
    clean.append(entry)
extra=extra.replace('and De la Prieta, Fernando','and {De la Prieta}, Fernando')
(ROOT/'references.bib').write_text('\n\n'.join(clean)+extra)
print('Saved',len(entries)+5,'references')

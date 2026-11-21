from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.opc.constants import RELATIONSHIP_TYPE as RT

root=Path('/Users/efehankeles/Documents/ChatGPT/CS 415 project 2')
doc=Document()
s=doc.sections[0]
s.page_width=Inches(8.27);s.page_height=Inches(11.69)
s.top_margin=s.bottom_margin=Inches(.65)
s.left_margin=s.right_margin=Inches(.72)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2']:
 st=doc.styles[name];st.font.name='Arial';st.font.size=Pt(9);st.font.color.rgb=RGBColor(0,0,0)
 st.paragraph_format.space_after=Pt(5)
 st.paragraph_format.line_spacing=1.08
 st.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Arial')
for name in ['Title','Heading 1','Heading 2']:
 doc.styles[name].font.bold=True
 doc.styles[name].paragraph_format.keep_with_next=True
 doc.styles[name].paragraph_format.space_before=Pt(8)
doc.styles['Title'].paragraph_format.space_before=Pt(0)
doc.styles['Subtitle'].font.italic=False

def p(text,style=None):
 x=doc.add_paragraph(text,style)
 x.paragraph_format.widow_control=True
 if len(doc._element.body) > 2 and x._p.getprevious() is not None and x._p.getprevious().tag == qn('w:tbl'):
  x.paragraph_format.space_before=Pt(6)
 return x

def h(text):return p(text,'Heading 1')
def guide(text):
 x=p('['+text+']')
 for r in x.runs:r.italic=True
 return x

def table(headers,rows,widths):
 t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 for c,w in zip(t.columns,widths):c.width=Inches(w)
 pr=t._tbl.tblPr
 borders=OxmlElement('w:tblBorders')
 for edge in ['top','left','bottom','right','insideH','insideV']:
  e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
 pr.append(borders)
 for j,v in enumerate(headers):t.rows[0].cells[j].text=v
 rep=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(rep)
 for row in rows:
  for cell,v in zip(t.add_row().cells,row):cell.text=v
 for i,row in enumerate(t.rows):
  cant=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(cant)
  for cell,w in zip(row.cells,widths):
   cell.width=Inches(w);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   cp=cell._tc.get_or_add_tcPr();m=OxmlElement('w:tcMar')
   for side in ['top','left','bottom','right']:
    v=OxmlElement('w:'+side);v.set(qn('w:w'),'75');v.set(qn('w:type'),'dxa');m.append(v)
   cp.append(m)
   if i==0:
    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'EDEFF2');cp.append(sh)
   for para in cell.paragraphs:
    para.paragraph_format.space_after=Pt(0);para.paragraph_format.line_spacing=1.05
    for r in para.runs:
     r.font.name='Arial';r.font.size=Pt(9);r.font.bold=(i==0)
 return t

def link(para,text,url):
 hl=OxmlElement('w:hyperlink');hl.set(qn('r:id'),para.part.relate_to(url,RT.HYPERLINK,is_external=True))
 r=OxmlElement('w:r');rp=OxmlElement('w:rPr')
 for tag,val in [('w:color','000000'),('w:sz','18'),('w:u','single')]:
  e=OxmlElement(tag);e.set(qn('w:val'),val);rp.append(e)
 rf=OxmlElement('w:rFonts');rf.set(qn('w:ascii'),'Arial');rf.set(qn('w:hAnsi'),'Arial');rp.append(rf)
 r.append(rp);tx=OxmlElement('w:t');tx.text=text;r.append(tx);hl.append(r);para._p.append(hl)

p('Product Review Rating Prediction with Convolutional Neural Networks','Title')
p('CS 415 Project Proposal','Subtitle')
table(['Group member names','Project responsibilities'],[
 ('[Name(s)]','Data exploration, cleaning, duplicate checks and dataset splits'),
 ('[Name(s)]','Model implementation, training and controlled comparisons'),
 ('[Name(s)]','Evaluation, error analysis and literature review')],[1.55,5.28])
guide('Replace the name fields and adjust responsibilities to the actual team. All members will contribute to writing and review. Bracketed text is guidance to replace before submission.')

h('1 Problem Definition and Motivation')
problem=['People often write reviews after buying products online. They share their opinions about the product and give it a rating from one to five stars. In this project, we want to build a deep learning model that predicts this rating by reading the review. The model will use only the review text to make its prediction. We will use the Amazon Fine Food Reviews dataset, which includes food reviews and their star ratings [9].', 'During training, the model will learn from reviews and their correct ratings. Later, we will give it reviews that it has not seen before. It will predict a rating for each review. We will then compare these predictions with the real ratings to check how well the model works. The model will choose one of five classes: one, two, three, four or five stars.', 'This task can be difficult because one review may include both positive and negative opinions. For example, a customer may write, "The taste is good, but the price is too high." The customer may give this product three or four stars. Different people can also give different ratings for similar experiences. We therefore need to measure both how often the model is correct and how far its wrong predictions are from the real ratings. Predicting two stars instead of one is a smaller error than predicting five stars instead of one [1].', 'A model like this could help companies understand customer feedback when a written review has no star rating. We plan to use a convolutional neural network, or CNN, to learn useful patterns in groups of words [2]. We also want to test the model on reviews of products that were not included in its training data. This will show how well it works on new food products. However, good results on food reviews do not mean that the model will also work well on electronics reviews or reviews in another language. Testing on another product category would be a separate step [7].']
for text in problem:p(text)

h('2 Related Work')
guide('Write a connected overview linking each paper to a design choice. Explain rating order using Pang and Lee [1], local text patterns using Kim [2], and pretrained word representations using GloVe [3]. Use Tang et al. [4] and Yang et al. [5] for recurrent and attention-based alternatives. Discuss Dang et al. [6] as a study using the same Fine Food dataset, while keeping its evaluation protocol distinct from ours. Use Blitzer et al. [7] for domain shift and Devlin et al. [8] if a BERT comparison is included. Select the references relevant to the final scope.')

h('3 Dataset and Preprocessing')
p('Amazon Fine Food Reviews contains 568,454 reviews of 74,258 products by 256,059 users, collected between October 1999 and October 2012 [9]. We propose using Text as the model input and Score as the target. ProductId will support grouping during evaluation rather than being used as a prediction feature.')
guide('Report the retained sample size, rating frequencies and review lengths after inspection. Specify removal of empty texts and duplicate handling before splitting. Prevent duplicate or near-duplicate review groups from crossing splits. Fit the vocabulary on training text only, preserve negation, and state tokenization, maximum sequence length, padding and truncation choices. Confirm whether all cleaned data or a reproducible subset will be used.')

h('4 Proposed Method and Research Hypothesis')
p('Suggested core experiment: compare the same TextCNN with randomly initialized trainable embeddings and pretrained GloVe embeddings that are fine-tuned during training [2,3]. Both variants will use matched embedding dimensions and the same data splits and tuning budget. The network will combine word embeddings, parallel one-dimensional convolutions, ReLU, global max pooling, dropout and a five-output classification layer. Training will use cross-entropy, with model selection based on validation macro-F1. A majority-class predictor will provide a simple reference.')
p('Proposed H1: pretrained embeddings will improve mean macro-F1 on the test set of unseen products by at least 0.02 absolute compared with random initialization. The two variants will use three matched training seeds. A smaller gain or a negative result will be reported as evidence against this hypothesis.')
guide('Confirm the core comparison, embedding source and version, dimensions, filter widths, optimizer and training budget. Treat BiLSTM or BERT as an optional additional comparison rather than promising every architecture.')

# Keep evaluation and references together as a coherent second page.
doc.add_page_break()
h('5 Evaluation and Measurable Success Criteria')
p('The primary protocol will separate products between training, validation and test sets, targeting approximately 80/10/10 percent of reviews while preserving all five rating classes where feasible. Duplicate groups will remain separated even if they span product IDs. Actual sizes and class distributions will be reported. Hyperparameters and checkpoints will be selected using validation data; the test set will remain untouched until evaluation. Each final configuration will be evaluated over three training seeds, reporting mean and standard deviation.')
p('The following are suggested proposal targets, not measured results. Confirm or revise them using training and validation evidence before final submission and before inspecting test performance.')
table(['Measure','Definition or purpose','Suggested criterion'],[
 ('Macro-F1','Unweighted mean of F1 across all five ratings','Mean >= 0.55 on unseen-product test'),
 ('MAE','Mean absolute difference between predicted and true stars','Mean <= 0.65 stars on the same test'),
 ('Embedding comparison','Pretrained minus random-init macro-F1 on identical splits','Mean gain >= 0.02 absolute to support H1'),
 ('Accuracy and class-level results','Exact rating accuracy, per-class F1 and confusion matrix','Report all 5 classes for both variants')
],[1.15,3.1,2.58])
p('An optional review-level random-split experiment may be reported separately, with a model trained for each protocol. Differences between protocols will be interpreted alongside training sizes and class distributions. A product-disjoint split evaluates transfer to unseen products within the food domain; it is not a cross-category OOD benchmark.')
guide('If cross-category evaluation is retained, specify the external English dataset, its 1-5 rating labels and sample size. Freeze the food-trained model and preprocessing before this test. Report external macro-F1 and MAE separately. Any target-domain adaptation requires its own training/validation data and a separate held-out test.')

h('6 Expected Outcomes and Deliverables')
guide('Describe the expected outputs without inventing findings: a reproducible data pipeline, trained model variants, a comparison of embedding initialization, and error analysis of neighboring ratings and unseen products. State limitations such as English-only data, subjective ratings and possible distribution shift. Add the actual hardware, schedule and final responsibility assignments.')

h('7 References')
refs=[
('B. Pang and L. Lee. 2005. Seeing Stars: Exploiting Class Relationships for Sentiment Categorization with Respect to Rating Scales. ACL, pp. 115-124.','https://aclanthology.org/P05-1015/'),
('Y. Kim. 2014. Convolutional Neural Networks for Sentence Classification. EMNLP, pp. 1746-1751.','https://aclanthology.org/D14-1181/'),
('J. Pennington, R. Socher and C. D. Manning. 2014. GloVe: Global Vectors for Word Representation. EMNLP, pp. 1532-1543.','https://aclanthology.org/D14-1162/'),
('D. Tang, B. Qin and T. Liu. 2015. Document Modeling with Gated Recurrent Neural Network for Sentiment Classification. EMNLP, pp. 1422-1432.','https://aclanthology.org/D15-1167/'),
('Z. Yang, D. Yang, C. Dyer, X. He, A. Smola and E. Hovy. 2016. Hierarchical Attention Networks for Document Classification. NAACL-HLT, pp. 1480-1489.','https://aclanthology.org/N16-1174/'),
('C. N. Dang, M. N. Moreno-Garcia and F. De la Prieta. 2021. An Approach to Integrating Sentiment Analysis into Recommender Systems. Sensors, 21(16), 5666.','https://doi.org/10.3390/s21165666'),
('J. Blitzer, M. Dredze and F. Pereira. 2007. Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification. ACL, pp. 440-447.','https://aclanthology.org/P07-1056/'),
('J. Devlin, M.-W. Chang, K. Lee and K. Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT, pp. 4171-4186.','https://aclanthology.org/N19-1423/'),
('Stanford SNAP. Amazon Fine Foods Reviews. Dataset. Accessed 21 October 2026.','https://snap.stanford.edu/data/web-FineFoods.html')]
for i,(txt,url) in enumerate(refs,1):
 para=p(f'[{i}] {txt} ');para.paragraph_format.space_after=Pt(5);link(para,'Source',url)
 if i==9:
  para.add_run(' | ');link(para,'Kaggle distribution','https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews')

out=root/'output/documents/product_review_rating_proposal_template.docx'
doc.core_properties.title='Product Review Rating Prediction with Convolutional Neural Networks'
doc.core_properties.subject='CS 415 project proposal template'
doc.core_properties.author=''
for tree in [doc.styles.element, doc._element]:
 for node in list(tree.iter(qn('w:pBdr'))):
  node.getparent().remove(node)
for name in ['Title','Subtitle']:
 for node in list(doc.styles[name].element.iter(qn('w:spacing'))):
  if node.getparent().tag == qn('w:rPr'):
   node.getparent().remove(node)
doc.save(out)
print(out)
print('Words:',sum(len(x.text.split()) for x in doc.paragraphs))

from pathlib import Path
import re, html, xml.etree.ElementTree as ET
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph
from pypdf import PdfReader

src=Path('/Users/efehankeles/Desktop/proposal.tex').read_text()
keys=re.findall(r'\\bibitem\{([^}]+)\}',src)
refs={k:str(i+1) for i,k in enumerate(keys)}
s=Path('/tmp/cs415-proposal.html').read_text()
body=ET.fromstring(re.search(r'<body>(.*?)</body>',s,re.S).group(0))
for suffix, filename in [('', 'Arial.ttf'),('-Bold','Arial Bold.ttf'),('-Italic','Arial Italic.ttf'),('-BoldItalic','Arial Bold Italic.ttf')]:
    pdfmetrics.registerFont(TTFont('Arial'+suffix,'/System/Library/Fonts/Supplemental/'+filename))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-BoldItalic')

def esc(t): return html.escape(re.sub(r'\s+',' ',t or '').replace('–','-').replace('—','-'))
def inline(e):
    text=esc(e.text)
    for c in e:
        inner=inline(c)
        if c.tag=='span' and 'data-cites' in c.attrib:
            inner='['+', '.join(refs[k] for k in c.attrib['data-cites'].split())+']'
        elif c.tag in ('strong','b'): inner='<b>'+inner+'</b>'
        elif c.tag in ('em','i'): inner='<i>'+inner+'</i>'
        elif c.tag=='a': inner='<a href="'+html.escape(c.attrib['href'],quote=True)+'">'+inner+'</a>'
        text+=inner+esc(c.tail)
    return text.strip()

base=ParagraphStyle('body',fontName='Arial',fontSize=9,leading=10.5,alignment=TA_JUSTIFY,spaceAfter=5)
head=ParagraphStyle('head',parent=base,fontName='Arial-Bold',alignment=TA_CENTER,spaceBefore=9,spaceAfter=5,keepWithNext=True)
sub=ParagraphStyle('sub',parent=base,fontName='Arial-Italic',spaceBefore=5,spaceAfter=4,keepWithNext=True)
bullet=ParagraphStyle('bullet',parent=base,leftIndent=10,firstLineIndent=0,bulletIndent=1,spaceAfter=3)
bib=ParagraphStyle('bib',parent=base,leftIndent=16,firstLineIndent=-16,spaceAfter=4)
story=[]
roman=['I','II','III','IV','V','VI']; section=0; subsection=0
for e in body:
    if e.tag=='header': continue
    if e.tag=='h1':
        label=inline(e)
        if e.attrib.get('class')!='unnumbered':
            label=roman[section]+'. '+label;section+=1
        subsection=0
        story.append(Paragraph(label,head))
    elif e.tag=='h2':
        story.append(Paragraph(chr(65+subsection)+'. '+inline(e),sub));subsection+=1
    elif e.tag=='p': story.append(Paragraph(inline(e),base))
    elif e.tag=='ul':
        for li in e:
            story.append(Paragraph(inline(li),bullet,bulletText='•'))
    elif e.tag=='div' and e.attrib.get('class')=='thebibliography':
        story.append(Paragraph('References',head))
        entries=list(e)[1:]
        assert len(entries)==len(keys)
        for i,p in enumerate(entries,1): story.append(Paragraph(f'[{i}] '+inline(p),bib))

W,H=A4; margin=42.5; bottom=42.5; gap=18; width=(W-2*margin-gap)/2
out='/Users/efehankeles/Desktop/proposal.pdf'
def frame_set(top):
    return [Frame(margin+i*(width+gap),bottom,width,H-top-bottom,id=f'col{i}',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0) for i in range(2)]
def page(canvas,doc):
    canvas.saveState()
    if doc.page==1:
        canvas.setFont('Arial',17)
        canvas.drawCentredString(W/2,H-48,'Product Review Rating Prediction')
    canvas.setFont('Arial',8)
    canvas.drawCentredString(W/2,25,str(doc.page))
    canvas.restoreState()
doc=BaseDocTemplate(out,pagesize=A4,title='Product Review Rating Prediction',author='Efe Han Keleş; Mehmet Ege Aşan; Faris Aleem Ahmad')
doc.addPageTemplates([PageTemplate(id='first',frames=frame_set(71),onPage=page,autoNextPageTemplate='rest'),PageTemplate(id='rest',frames=frame_set(42.5),onPage=page)])
doc.build(story)
r=PdfReader(out)
text='\n'.join(p.extract_text() for p in r.pages)
for needle in ['Efe Han Keleş','Mehmet Ege Aşan','Faris Aleem Ahmad','Numerical Success Criteria','[10]','3%','5%','0.01']:
    assert needle in text,needle
print('Created:',out,'Pages:',len(r.pages),'Bytes:',Path(out).stat().st_size)
for i,p in enumerate(r.pages,1):print('Page',i,'size',list(p.mediabox),'characters',len(p.extract_text()))

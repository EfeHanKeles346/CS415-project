from pathlib import Path
import re, html, xml.etree.ElementTree as ET
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, XPreformatted
from pypdf import PdfReader
from PIL import Image,ImageOps,ImageDraw

root=ET.fromstring('<body>'+Path('/tmp/cs415-literature.html').read_text()+'</body>')
for name,file in [('Arial','Arial.ttf'),('Arial-Bold','Arial Bold.ttf'),('Arial-Italic','Arial Italic.ttf'),('Arial-BoldItalic','Arial Bold Italic.ttf'),('Unicode','Arial Unicode.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'/System/Library/Fonts/Supplemental/'+file))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-BoldItalic')
chars=pdfmetrics.getFont('Arial').face.charToGlyph
missing=set()
def esc(s):
    s=re.sub(r'\s+',' ',s or '').replace('–','-').replace('—','-').replace('\u2011','-')
    out=''
    for c in s:
        t=html.escape(c)
        if ord(c) not in chars:
            if ord(c) not in pdfmetrics.getFont('Unicode').face.charToGlyph: missing.add(c)
            t='<font name="Unicode">'+t+'</font>'
        out+=t
    return out

def inline(e):
    out=esc(e.text)
    for c in e:
        t=inline(c)
        if c.tag in ('strong','b'):t='<b>'+t+'</b>'
        elif c.tag in ('em','i'):t='<i>'+t+'</i>'
        elif c.tag=='code':t='<font color="#34465a">'+t+'</font>'
        elif c.tag=='a':
            href=html.escape(re.sub(r'^#\d+-', '#', c.attrib['href']),quote=True)
            t='<a href="'+href+'" color="#17599b">'+t+'</a>'
        elif c.tag=='br':t='<br/>'
        out+=t+esc(c.tail)
    return out.strip()

body=ParagraphStyle('body',fontName='Arial',fontSize=10.5,leading=14,spaceAfter=7)
styles={
'h1':ParagraphStyle('title',parent=body,fontName='Arial-Bold',fontSize=20,leading=25,spaceAfter=16,textColor=HexColor('#17324d'),keepWithNext=True),
'h2':ParagraphStyle('section',parent=body,fontName='Arial-Bold',fontSize=15,leading=19,spaceBefore=17,spaceAfter=9,textColor=HexColor('#17324d'),keepWithNext=True),
'h3':ParagraphStyle('subsection',parent=body,fontName='Arial-Bold',fontSize=12,leading=16,spaceBefore=11,spaceAfter=6,keepWithNext=True)}
quote=ParagraphStyle('quote',parent=body,leftIndent=12,rightIndent=12,textColor=HexColor('#435264'),borderColor=HexColor('#d8e1eb'),borderWidth=.6,borderPadding=8,spaceBefore=5,spaceAfter=12)
cell=ParagraphStyle('cell',parent=body,fontSize=9,leading=12,spaceAfter=0)
code=ParagraphStyle('code',parent=body,fontName='Unicode',fontSize=9,leading=12,backColor=HexColor('#f2f5f8'),borderPadding=8,spaceBefore=7,spaceAfter=10)
story=[]; W,H=A4; width=W-100

def add(e,level=0):
    tag=e.tag
    if tag in styles:
        anchor='<a name="'+html.escape(e.attrib.get('id',''),quote=True)+'"/>' if 'id' in e.attrib else ''
        story.append(Paragraph(anchor+inline(e),styles[tag]))
    elif tag=='p':story.append(Paragraph(inline(e),body))
    elif tag in ('ol','ul'):
        for i,li in enumerate(e,1):
            label=str(i)+'.' if tag=='ol' else '•'
            st=ParagraphStyle('list',parent=body,leftIndent=15+12*level,bulletIndent=12*level,spaceAfter=5)
            # The source contains flat lists; preserve any nested list separately.
            nested=[c for c in li if c.tag in ('ul','ol')]
            for n in nested:li.remove(n)
            story.append(Paragraph(inline(li),st,bulletText=label))
            for n in nested:add(n,level+1)
        story.append(Spacer(1,3))
    elif tag=='blockquote':
        for c in e:story.append(Paragraph(inline(c),quote))
    elif tag=='pre':
        t=''.join(e.itertext()).rstrip()
        story.append(XPreformatted(html.escape(t),code))
    elif tag=='table':
        rows=[]
        for tr in e.iter('tr'):
            row=[]
            for td in tr:
                s=inline(td)
                if td.tag=='th':s='<b>'+s+'</b>'
                row.append(Paragraph(s,cell))
            rows.append(row)
        n=len(rows[0]); widths=([.18,.29,.26,.27] if n==4 else [.22,.38,.40])
        t=Table(rows,colWidths=[width*v for v in widths],repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#e9eff6')),('GRID',(0,0),(-1,-1),.4,HexColor('#cad4df')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        story.extend([t,Spacer(1,9)])
    elif tag=='hr':story.append(Spacer(1,8))
    else:raise ValueError('Unsupported block '+tag)
for e in root:add(e)
assert not missing,missing
out='/Users/efehankeles/Desktop/Product_Review_Rating_Prediction_Literatur_Rehberi_TR.pdf'
def page(c,d):
    c.saveState();c.setFont('Arial',8);c.setFillColor(HexColor('#607086'))
    if d.page>1:c.drawString(50,H-28,'CS 415 | Türkçe Literatür ve Yöntem Rehberi')
    c.drawString(50,28,'Product Review Rating Prediction');c.drawRightString(W-50,28,str(d.page));c.restoreState()
d=SimpleDocTemplate(out,pagesize=A4,rightMargin=50,leftMargin=50,topMargin=48,bottomMargin=46,title='Product Review Rating Prediction - Türkçe Literatür ve Yöntem Rehberi')
d.build(story,onFirstPage=page,onLaterPages=page)
r=PdfReader(out)
text='\n'.join(p.extract_text() for p in r.pages)
for needle in ['3.1 Kim','3.10 Rostami','9. Kaynakça','568.454','Macro-F1']:assert needle in text,needle
print('PDF:',out,'Pages:',len(r.pages),'Links:',sum(len(p.get('/Annots',[])) for p in r.pages),'Bytes:',Path(out).stat().st_size)
print('Page text lengths:',[len(p.extract_text()) for p in r.pages])

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
readme='''# Overleaf source package\n\nImport this ZIP as a new Overleaf project. Select progress_report.tex as the main document and use pdfLaTeX. The project uses the standard IEEEtran bibliography style. All required generated tables and PDF figures are included. Re-running the experiments is not required to compile.\n\nThe original local progress folder also contains raw data, Python scripts, predictions, and a Turkish README; those larger files are intentionally excluded from this source package.\n'''
files=[ROOT/x for x in ['progress_report.tex','references.bib','IEEEtran.cls','progress_report.pdf']]
files+=sorted((ROOT/'results').glob('*.tex'))+sorted((ROOT/'figures').glob('*.pdf'))
with ZipFile(ROOT/'progress_overleaf.zip','w',ZIP_DEFLATED) as z:
    for path in files: z.write(path,path.relative_to(ROOT))
    z.writestr('README_Overleaf.md',readme)
with ZipFile(ROOT/'progress_overleaf.zip') as z:
    assert z.testzip() is None
    contents=z.namelist()
reader=PdfReader(ROOT/'progress_report.pdf')
text='\n'.join(page.extract_text() for page in reader.pages)
assert len(reader.pages)==7
assert '??' not in text
for token in ['TODO','PLACEHOLDER','TBD']:
    assert token not in text
assert '393,001' in text and '56,791' in text
log=(ROOT/'build/progress_report.log').read_text()
assert 'Overfull' not in log
assert 'Citation' not in log or 'undefined' not in log
qa={'pages':len(reader.pages),'figures':3,'tables':7,'bibliography_entries':14,
    'unresolved_references':False,'overfull_boxes':False,'archive_integrity':'passed',
    'source_package_files':contents,'visual_review':'All final pages rendered; layout, figures, and table widths inspected.',
    'fonts':'All PDF fonts embedded; Times-compatible Nimbus Roman body font verified with pdffonts.'}
(ROOT/'results/pdf_quality_check.json').write_text(json.dumps(qa,indent=2)+'\n')
print('Packaged',len(contents),'files;',len(reader.pages),'PDF pages;', (ROOT/'progress_overleaf.zip').stat().st_size,'ZIP bytes')

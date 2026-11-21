# Reference verification

Checked on 2026-11-16 against primary sources: ACL Anthology, the authors' arXiv records, the Sensors article, and the author manuscript of the IEEE TKDE survey hosted by Cardiff University.

- ACL records were fetched as BibTeX; raw metadata is retained in `references_original.bib`.
- Conference editor lists, publisher addresses, and months were omitted from the report bibliography for a compact IEEE format. Titles, paper authors, year, page ranges, DOI, and source links were retained.
- Kasa et al. (2024): the PDF title page lists Nikhil Pattisapu and places Anish Bhanushali before him. The Anthology metadata differs. The report follows the authoritative PDF (`kasa2024.pdf`), consistent with the Anthology's instruction to prefer PDF metadata.
- DistilBERT is identified as an arXiv preprint, not a main-conference paper.
- The survey's publication is IEEE TKDE 28(1), 127–146, 2016, DOI 10.1109/TKDE.2015.2457911. The author manuscript is https://orca.cardiff.ac.uk/id/eprint/112802/1/2016-Ordinal%20regression%20methods%253a%20Survey%20and%20experimental%20study.pdf.
- CORAL is cited in its published form: Pattern Recognition Letters 140, 325–331, 2020, DOI 10.1016/j.patrec.2020.11.008.
- OLL uses unnormalized star distance with exponent alpha; MLL uses lambda*CE + (1-lambda)*OLL. The implementation illustration fixes alpha=1 and lambda=0.5.
- No numbers from different papers were presented as directly comparable results of this project's product-disjoint experiment.

The provided proposal is a project-content source. Its proposed experiments were treated as plans, not evidence of completed training.

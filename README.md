# CS415 project

Product Review Rating Prediction — proposal, literature review, progress report, reproducible baseline experiments, and supporting data.

## Reports

- [Adjusted project proposal](proposal_original_adjusted.pdf)
- [Original proposal PDF](proposal.pdf) and [LaTeX source](proposal.tex)
- [IEEE progress report](progress/progress_report.pdf)
- [Progress report LaTeX source](progress/progress_report.tex)
- [Overleaf source package](progress/progress_overleaf.zip)
- [Turkish progress guide](progress/README_TR.md)
- [Turkish literature guide](Product_Review_Rating_Prediction_Literatur_Rehberi_TR.md)

## Contents

| Location | Contents |
|---|---|
| Repository root | Proposal versions, literature guides, and reference material |
| `progress/scripts/` | Baseline training, evidence verification, figures, and report packaging |
| `progress/data/` | Fine Foods source archive, cleaned reviews, and split manifest |
| `progress/results/` | Per-seed predictions, metrics, configurations, and generated LaTeX tables |
| `progress/figures/` | PDF and PNG figures |
| `progress/sources/` | Proposal copy, prior audits, and literature verification material |
| `progress/build/` | LaTeX logs and rendered report pages |
| `output/` | Editable proposal document |
| `tmp/` | Earlier proposal/literature preparation scripts and intermediate artifacts |

All project artifacts from the local project folder are included. Installed Python environments, bytecode, operating-system caches, and credentials are excluded; package versions are recorded in `progress/requirements.txt`.

## Recorded progress

The November 16, 2026 report audits 568,454 raw reviews and retains 393,001 normalized, deduplicated records. Product-disjoint partitions contain 276,844 training, 56,791 validation, and 59,366 reserved test reviews.

Two TF–IDF linear baselines were each trained with seeds 17, 29, and 43. These are completed validation experiments. TextCNN, DistilBERT, and the CE/OLL/MLL comparison are planned work in this snapshot; no neural-model training results are claimed in these files.

## Downloading all files

The three main dataset artifacts are tracked with Git LFS. Install Git LFS before cloning, then run:

```bash
git lfs install
git clone https://github.com/EfeHanKeles346/CS415-project.git
cd CS415-project
git lfs pull
```

## Reproducing the experiments

```bash
cd progress
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_experiments.py > results/run.log 2>&1
.venv/bin/python scripts/verify_evidence.py
.venv/bin/python scripts/make_report_assets.py
```

The recorded experiment used Python 3.12 on macOS. The CPU-description command in `run_experiments.py` uses macOS `sysctl`; adapt that line when running on another operating system. Re-running experiments replaces generated result files, and measured timings may change.

With Tectonic installed, compile the existing report without rerunning the experiments:

```bash
cd progress  # if currently at the repository root
bash build_report.sh
```

Alternatively, import `progress/progress_overleaf.zip` into Overleaf and select `progress_report.tex` as the main document.

The dataset comes from [Stanford SNAP Amazon Fine Foods](https://snap.stanford.edu/data/web-FineFoods.html); source attribution, checksum, and cleaning rules are retained in the progress artifacts. Third-party data and publications retain their respective rights.

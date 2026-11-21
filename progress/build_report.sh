#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
tectonic --keep-logs --keep-intermediates progress_report.tex
mkdir -p build
for extension in aux bbl blg log out; do
  if [[ -f "progress_report.$extension" ]]; then
    mv "progress_report.$extension" "build/progress_report.$extension"
  fi
done

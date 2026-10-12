#!/usr/bin/env bash
# One-command setup for plm-evolve. Creates an isolated venv and installs the
# core plus the ESM-C SDK and plotting extras. No conda needed.
#
#   ./setup.sh            # default: .venv, python3, core + esmc + viz
#   PYTHON=python3.11 ./setup.sh
#   VENV=.venv-esmc ./setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

echo "[plm-evolve] creating venv at $VENV (python: $PYTHON) ..."
"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install -U pip wheel

echo "[plm-evolve] installing core + esmc + viz (this pulls torch, may take a few minutes) ..."
"$VENV/bin/pip" install -e ".[esmc,viz]"

echo
echo "[plm-evolve] done."
echo "  activate:  source $VENV/bin/activate"
echo "  check:     plm-evolve --help"
echo "  ESM-2 weights download from HuggingFace on first use; ESM-C from the esm SDK."

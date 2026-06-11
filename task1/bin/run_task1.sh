#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python3 task1/src/recommender.py "$@"

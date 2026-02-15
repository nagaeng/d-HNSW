#!/bin/bash
#
# Download benchmark datasets for DHNSW
# Usage: ./scripts/datasets.sh [--sift-only]
#
# Supported datasets: SIFT1M, GIST1M, SIFT10M, DEEP100M, TEXT10M
#

set -e

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TOOLS_DIR="$PROJECT_ROOT/tools"

mkdir -p datasets && cd datasets
HOST="$(hostname -s)"
pip install numpy

# SIFT1M
if [ ! -d "sift" ]; then
  echo "Downloading SIFT1M"
  wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
  tar -xzvf sift.tar.gz
fi

# GIST1M
if [ ! -d "gist" ]; then
  echo "Downloading GIST1M"
  wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
  tar -xzvf gist.tar.gz
fi

# SIFT 10M (multi-node: base vectors on node-0, queries on other nodes)
if [ ! -d "sift10M" ]; then
  echo "Downloading SIFT10M"
  mkdir sift10M && cd sift10M
  if [ "$HOST" == "node-0" ]; then
    wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
    gunzip bigann_base.bvecs.gz
  else
    wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
    wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz 
    gunzip bigann_query.bvecs.gz
    tar -xzvf bigann_gnd.tar.gz
  fi
  cd ..
fi

# DEEP100M (multi-node: base vectors on node-0, queries on other nodes)
if [ ! -d "deep10M" ]; then
  echo "Downloading DEEP10M"
  mkdir deep10M && cd deep10M
  if [ "$HOST" == "node-0" ]; then
    wget https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin -O deep1b_base.fbin 
    python3 "$TOOLS_DIR/pick_vecs.py" --src deep1b_base.fbin --dst deep10M_base.fvecs --topk 10000000 --dim 96
  else
    wget https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin -O deep10M_query.fbin
    wget https://github.com/matsui528/deep1b_gt/releases/download/v0.1.0/gt.zip
    python3 "$TOOLS_DIR/convert.py" fbin deep10M_query.fbin deep10M_query.fvecs
    unzip gt
  fi
  cd ..
fi

# TEXT10M (multi-node: base vectors on node-0, queries on other nodes)
# TODO: text10M gt is wrong
if [ ! -d "text10M" ]; then
  echo "Downloading TEXT10M"
  mkdir text10M && cd text10M
  if [ "$HOST" == "node-0" ]; then
    wget https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.10M.fbin
    python3 "$TOOLS_DIR/convert.py" fbin base.10M.fbin text10M.fvecs 
  else
    wget https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin
    wget https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/groundtruth.public.100K.ibin
    python3 "$TOOLS_DIR/convert.py" fbin query.public.100K.fbin text10M_query.fvecs
    python3 "$TOOLS_DIR/convert.py" fbin groundtruth.public.100K.ibin text10M_groundtruth.fvecs 
  fi
  cd ..
fi

echo "Dataset download completed."

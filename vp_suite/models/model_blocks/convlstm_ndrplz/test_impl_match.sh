#!/bin/bash

if [[ "$(dirname "$0")" != "." ]]; then
  echo 'Please change to the folder containing this script before execution. Exiting...'
  exit
fi

if [[ ! -d "ConvLSTM_pytorch" ]]; then
  git clone https://github.com/ndrplz/ConvLSTM_pytorch.git
fi

python ./_test_impl_match.py

rm -rf ./ConvLSTM_pytorch
#!/bin/bash

if [[ "$(dirname "$0")" != "." ]]; then
  echo 'Please change to the folder containing this script before execution. Exiting...'
  exit
fi

if [[ ! -d "Precipitation-Nowcasting" ]]; then
  git clone https://github.com/Hzzone/Precipitation-Nowcasting.git
fi

python -O ./_test_impl_match.py  # reference code has failing assertions by default

# rm -rf ./Precipitation-Nowcasting
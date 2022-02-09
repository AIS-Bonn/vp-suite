#!/bin/bash

if [[ "$(dirname "$0")" != "." ]]; then
  echo 'Please change to the folder containing this script before execution. Exiting...'
  exit
fi

if [[ ! -d "Precipitation-Nowcasting" ]]; then
  git clone https://github.com/Hzzone/Precipitation-Nowcasting.git
fi

# needs to run with assertions deactivated
# since the reference code has a failing assertion by default...
python -O ./_test_impl_match.py  #

rm -rf ./Precipitation-Nowcasting
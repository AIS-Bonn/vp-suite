#!/bin/bash

if [[ "$(dirname "$0")" != "." ]]; then
  echo 'Please change to the folder containing this script before execution. Exiting...'
  exit
fi

if [[ ! -d "predrnn-pytorch" ]]; then
  git clone https://github.com/thuml/predrnn-pytorch.git
fi

python ./_test_impl_match.py\
    --is_training 0 \
    --device cuda \
    --img_channel 3 \
    --num_hidden 128,128,128,128 \
    --layer_norm 0 \
    --reverse_scheduled_sampling 1 \
    --batch_size 4 \

#!/bin/bash

### CONFIG ###
TARGET_DIR=$1
if [ -z $TARGET_DIR ]; then
  echo "Must specify target directory"
  exit 1
fi

if [ ! -d $TARGET_DIR ]; then
  mkdir -p $TARGET_DIR
fi

SETS="00 01 02 03 04 05 06 07 08 09 10"
BASE_URL="https://drive.google.com/drive/folders/1IBlcJP8YsCaT81LwQ2YwQJac8bf1q8xF"

for set_nr in $SETS; do
  src=${BASE_URL}/set${set_nr}.tar
  dst=${TARGET_DIR}/set${set_nr}.tar
  unzip_dst=${TARGET_DIR}/set${set_nr}
  echo "Downloading from ${src}"
  wget ${src} -O ${dst}
  exit 0
  echo "Unpacking to ${unzip_dst}..."
  tar -xf ${dst} -C ${unzip_dst}
  rm -r ${dst}
done
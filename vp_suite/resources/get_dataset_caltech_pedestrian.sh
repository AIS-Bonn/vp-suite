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

file_ids=(
  1tPeaQr1cVmSABNCJQsd8OekOZIjpJivj
  1apo5VxoZA5m-Ou4GoGR_voUgLN0KKc4g
  1yvfjtQV6EnKez6TShMZQq_nkGyY9XA4q
  1jvF71hw4ztorvz0FWurtyCBs0Dy_Fh0A
  11Q7uZcfjHLdwpLKwDQmr5gT8LoGF82xY
  1Q0pnxM5cnO8MJJdqzMGIEryZaEKk_Un_
  1ft6clVXKdaxFGeihpth_jdBQxOIirSk7
  1-E_B3iAPQKTvkZ8XyuLcE2Lytog3AofW
  1oXCaTPOV0UYuxJJrxVtY9_7byhOLTT8G
  1f0mpL2C2aRoF8bVex8sqWaD8O3f9ZgfR
  18TvsJ5TKQYZRlj7AmcIvilVapqAss97X
) # file_ids for set00, set01, ..., set10 (on google drive)

i=0
for file_id in "${file_ids[@]}"; do
  # downloading
  set_nr=$(printf "%02d" $i)
  dst=${TARGET_DIR}/set${set_nr}.tar
  file_link="https://drive.google.com/file/d/${file_id}/view?usp=sharing"
  gdown --fuzzy $file_link -O $dst

  # unpacking
  unzip_dst=${TARGET_DIR}/set${set_nr}
  mkdir ${unzip_dst}
  echo "Unpacking to ${unzip_dst}..."
  tar -xf ${dst} -C ${TARGET_DIR}
  rm -r ${dst}

  ((++i))
done

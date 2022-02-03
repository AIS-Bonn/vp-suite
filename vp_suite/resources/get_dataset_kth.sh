#!/bin/bash

# Originally from https://github.com/edenton/svg,
# Modified by Ani Karapetyan and further modified here.

TARGET_DIR=$1
CLASSES="boxing handclapping handwaving jogging running walking"
IMG_SIZE=64
IMG_STR="${IMG_SIZE}x${IMG_SIZE}"

### DOWNLOAD ###
if [ -z $TARGET_DIR ]; then
  echo "Must specify target directory"
else
  mkdir -p $TARGET_DIR/processed
  mkdir -p $TARGET_DIR/raw
  URL=http://www.cs.nyu.edu/~denton/datasets/kth.tar.gz
  echo "Downloading ${URL}"
  wget -q $URL -P $TARGET_DIR/processed
  tar -zxf $TARGET_DIR/processed/kth.tar.gz -C $TARGET_DIR/processed/
  rm $TARGET_DIR/processed/kth.tar.gz

  for class in $CLASSES; do
    URL=http://www.nada.kth.se/cvap/actions/"$class".zip
    echo "Downloading ${URL}"
    wget -q $URL -P $TARGET_DIR/raw
    mkdir $TARGET_DIR/raw/$class
    unzip -q $TARGET_DIR/raw/"$class".zip -d $TARGET_DIR/raw/$class
    rm $TARGET_DIR/raw/"$class".zip
  done
fi

### CONVERT ###
for class in $CLASSES; do
  echo "Unpacking videos for: ${class}"
  for fname in $TARGET_DIR/raw/$class/*; do
    fname="$(basename -- $fname)"
    mkdir -p $TARGET_DIR/processed/$class/${fname:0:-11}
    src="${TARGET_DIR}/raw/${class}/${fname}"
    dst="${TARGET_DIR}/processed/${class}/${fname:0:-11}/image-%03d_${IMG_STR}.png"
    ffmpeg -hide_banner -loglevel error -i $src -r 25 -f image2 -s $IMG_STR $dst
  done
done

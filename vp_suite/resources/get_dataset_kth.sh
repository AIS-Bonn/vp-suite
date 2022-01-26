#!/bin/bash

TARGET_DIR=$1
CLASSES="boxing handclapping handwaving jogging running walking"
IMG_SIZE=64

### DOWNLOAD ###
if [ -z $TARGET_DIR ]
then
  echo "Must specify target directory"
else
  mkdir $TARGET_DIR/processed
  mkdir $TARGET_DIR/raw
  URL=http://www.cs.nyu.edu/~denton/datasets/kth.tar.gz
  wget $URL -P $TARGET_DIR/processed
  tar -zxvf $TARGET_DIR/processed/kth.tar.gz -C $TARGET_DIR/processed/
  rm $TARGET_DIR/processed/kth.tar.gz

  for c in walking jogging running handwaving handclapping boxing
  do  
    URL=http://www.nada.kth.se/cvap/actions/"$c".zip
    wget $URL -P $TARGET_DIR/raw
    mkdir $TARGET_DIR/raw/$c
    unzip $TARGET_DIR/raw/"$c".zip -d $TARGET_DIR/raw/$c
    rm $TARGET_DIR/raw/"$c".zip
  done
fi

### CONVERT ###
for class in $CLASSES; do
	for fname in $TARGET_DIR/raw/$class/*; do
		fname="$(basename -- $fname)"
		mkdir -p $TARGET_DIR/processed/$class/${fname:0:-11}
		ffmpeg -i $TARGET_DIR/raw/$class/${fname} -r 25 -f image2 -s ${IMG_SIZE}x${IMG_SIZE} $TARGET_DIR/processed/$class/${fname:0:-11}/image-%03d_${IMG_SIZE}x${IMG_SIZE}.png
	done
done

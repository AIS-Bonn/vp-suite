#!/bin/bash
# Translated and modified from:
# https://github.com/kotaro-inoue/human3.6m_downloader

### CONFIG ###
TARGET_DIR=$1
if [ -z $TARGET_DIR ]; then
  echo "Must specify target directory"
  exit 1
fi

BASEURL='http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath='
DATATYPE="Videos"

### DOWNLOAD DATASET ###
read -t 300 -p "Enter username for http://vision.imar.ro/human3.6m/: " username
read -s -t 300 -p "Enter password: " password

# prep
rm -f cookies.txt checklogin.php
wget --no-check-certificate --keep-session-cookies --save-cookies cookies.txt\
 --post-data username=${username}\&password=${password}\
 https://vision.imar.ro/human3.6m/checklogin.php

# Download Training dataset by subject
TRAIN_SUBJECTS="1,1 6,5 7,6 2,7 3,8 4,9 5,11"
for subject in $TRAIN_SUBJECTS; do
  IFS=',' read -a subj <<< "${subject}"
  fdir=$TARGET_DIR/training/subject/s${subj[1]}/
  mkdir -p $fdir
  fname_url="&filename=SubjectSpecific_${subj[0]}.tgz&downloadname=S${subj[1]}"
  src="${BASEURL}${DATATYPE}${fname_url}"
  dst="${fdir}${DATATYPE}.tgz"
  echo $dst
  if [ ! -f $dst ]; then
    wget --no-check-certificate --load-cookies cookies.txt $src -O $dst
  fi
done

# Download Testing dataset by subject
TEST_SUBJECTS="1,2 2,3 3,4 4,10"
for subject in $TEST_SUBJECTS; do
  IFS=',' read -a subj <<< "${subject}"
  fdir=$TARGET_DIR/testing/subject/s${subj[1]}/
  mkdir -p $fdir
  fname_url="&filename=SubjectSpecific_${subj[0]}.tgz&downloadname=S${subj[1]}"
  src="${BASEURL}${DATATYPE}${fname_url}"
  dst="${fdir}${DATATYPE}.tgz"
  if [ ! -f $dst ]; then
    wget --no-check-certificate --load-cookies cookies.txt $src -O $dst
  fi
done

rm cookies.txt checklogin.php

### UNPACK DATASET ###
# training dataset
echo "\nunpacking training dataset..."
FLIST="${TARGET_DIR}/training/subject/s*/*.tgz"
SUBJECTS="1 5 6 7 8 9 11"
for file in $FLIST; do
  tar -zxf $file
done
rm -r ${TARGET_DIR}/training/subject/*
for subject in $SUBJECTS; do
  mv -f S${subject} ${TARGET_DIR}/training/subject/
done

# unpack testing dataset
echo "\nunpacking testing dataset..."
FLIST="${TARGET_DIR}/testing/subject/s*/*.tgz"
SUBJECTS="1 7 8 9"
for file in $FLIST; do
  tar -zxf $file
done
rm -r ${TARGET_DIR}/testing/subject/*
for subject in $SUBJECTS; do
  mv -f S${subject} ${TARGET_DIR}/testing/subject/
done

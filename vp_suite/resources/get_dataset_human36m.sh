#!/bin/bash
# Translated and modified from:
# https://github.com/kotaro-inoue/human3.6m_downloader

### CONFIG ###
TARGET_DIR=$1
if [ -z $TARGET_DIR ]; then
  echo "Must specify target directory"
  exit 1
fi

if [ ! -d "${TARGET_DIR}/training" ]; then
  mkdir -p "${TARGET_DIR}/training"
fi

if [ ! -d "${TARGET_DIR}/testing" ]; then
  mkdir -p "${TARGET_DIR}/testing"
fi

BASEURL='http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath='
DATATYPE="Videos"

### DOWNLOAD DATASET ###
read -t 300 -p "Enter username for http://vision.imar.ro/human3.6m/: " username
read -s -t 300 -p "Enter password: " password

# prep
rm -f cookies.txt checklogin.php
wget --no-check-certificate --keep-session-cookies --save-cookies cookies.txt \
  --post-data username=${username}\&password=${password} \
  https://vision.imar.ro/human3.6m/checklogin.php

# first index specifies download index, second index specifies actual subject id.
# NOTE: "testing,4,10" is omitted since for subject 10, no videos are available
SUBJECTS=(
  "training,1,1"
  "training,6,5"
  "training,7,6"
  "training,2,7"
  "training,3,8"
  "training,4,9"
  "training,5,11"
  "testing,1,2"
  "testing,2,3"
  "testing,3,4"
  #"testing,4,10"
)

# download dataset by subject
for subject in "${SUBJECTS[@]}"; do
  IFS=',' read -a subj <<<"${subject}"
  is_test=""
  if [ "${subj[0]}" == "testing" ]; then
    is_test="Rest"
  fi
  tar_url="&filename=SubjectSpecific${is_test}_${subj[1]}.tgz&downloadname=S${subj[2]}"
  src="${BASEURL}${DATATYPE}${tar_url}"
  dst="${TARGET_DIR}/${subj[0]}/subject/s${subj[2]}"
  mkdir -p "${dst}"
  dst_tar="${dst}/${DATATYPE}.tgz"
  if [ ! -f "${dst}" ]; then
    wget --no-check-certificate --load-cookies cookies.txt "${src}" -O "${dst_tar}"
  fi
done

rm cookies.txt checklogin.php

# unpack training dataset by subject
TAR_FILES="${TARGET_DIR}/*/subject/s*/${DATATYPE}.tgz"
for tar_file in $TAR_FILES; do
  tar_dir="$(dirname "${tar_file}")"
  if [ ! -d "${tar_dir}/${DATATYPE}" ]; then
    echo "unpacking ${tar_file} to ${tar_dir}"
    tar -zxf "${tar_file}" -C "${tar_dir}"
  fi
  S_dir="${tar_dir}/S*"
  for S_d in ${S_dir}; do
    mv "${S_d}/${DATATYPE}" "${tar_dir}"
    rm -r "${S_d}"
  done
  rm "${tar_file}"
done

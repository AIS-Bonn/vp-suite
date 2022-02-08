#!/bin/bash

if [[ "$(dirname "$0")" != "." ]]; then
  echo 'Please change to the folder containing this script before execution. Exiting...'
  exit
fi

# remove old api docs
mv source/index.rst source/index
mv source/readme.rst source/readme
rm -f source/vp_suite*.rst
mv source/index source/index.rst
mv source/readme source/readme.rst

# build api docs
sphinx-apidoc -f -e -M -d 1 -t _templates -o ./source ../vp_suite
rm -f source/modules.rst  # modules.rst is auto-generated but not used by our index.rst

# remove the words 'module' and 'package' from all generated RST files except source and readme
RST_FILES="source/*.rst"
for file in $RST_FILES; do
  if [ "$file" = "source/index.rst" ] || [ "$file" = "source/readme.rst" ]; then
    continue
  fi
  sed -e "s/ module//g" -i $file
  sed -e "s/ package//g" -i $file
done

# build HTML from api docs
make clean
make html
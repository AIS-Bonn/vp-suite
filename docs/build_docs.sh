if [[ $PWD/ != */vp-suite/docs/ ]]; then
  echo 'Please change to the documentation folder before runnning this script.'
  echo 'Exiting...'
  exit
fi

mv source/index.rst source/index.rst_
mv source/readme.rst source/readme.rst_
rm -f source/*.rst
sphinx-apidoc -feMo ./source ../vp_suite
mv source/index.rst_ source/index.rst
mv source/readme.rst_ source/readme.rst

make clean
make html
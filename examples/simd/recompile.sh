#!/bin/sh

export CXX=g++-mp-4.9
export OCCA_DEVELOPER=1
export OCCA_DIR=`pwd`
export OCCA_CXX=$CXX
export OCCA_CXXFLAGS="-march=native -ftree-vectorize -fopt-info-vec-missed -O3 -Wa,-q"

rm -rf ~/._occa

cd $OCCA_DIR
make clean
make -j

cd examples/simd
make clean
make -j

./main

To compile OpenMP on the Mac compile occa with something like

```
CXX=g++-mp-4.9 ./main
```

To compile OpenMP at runtime something like this is needed

```
OCCA_CXX=g++-mp-4.9 OCCA_CXXFLAGS="-march=native -ftree-vectorize -fopt-info-vec-all -O3 -Wa,-q" ./main
```

To disassemble code on the mac you can use

```
otool -tvV /Users/lucas/._occa/kernels/9cf6de24edaee6c0/binary
```

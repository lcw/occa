#include <iostream>

#include "occa.hpp"

int main(int argc, char **argv){
  occa::printAvailableDevices();

  int entries = 1024;

  float *a  = new float[4*entries];
  float *b  = new float[4*entries];
  float *ab = new float[4*entries];

  for(int i = 0; i < 4*entries; ++i){
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }
  occa::setVerboseCompilation(true);

  occa::device device;
  occa::kernel simd;
  occa::memory o_a, o_b, o_ab;

  //---[ Device setup with string flags ]-------------------
  // device.setup("mode = Serial");
  device.setup("mode = OpenMP  , schedule = compact, chunk = 256");
  // device.setup("mode = OpenCL  , platformID = 0, deviceID = 1");
  // device.setup("mode = CUDA    , deviceID = 0");
  // device.setup("mode = Pthreads, threadCount = 4, schedule = compact, pinnedCores = [0, 0, 1, 1]");
  // device.setup("mode = COI     , deviceID = 0");
  //========================================================

  //---[ Device setup with Python-like arguments ]----------
  //    device.setup("OpenMP",
  //                 occa::schedule = "compact",
  //                 occa::chunk    = 10);
  //
  //    device.setup("OpenCL",
  //                 occa::platformID = 0,
  //                 occa::deviceID   = 0);
  //
  //    device.setup("CUDA",
  //                 occa::deviceID = 0);
  //
  //    device.setup("Pthreads",
  //                 occa::threadCount = 4,
  //                 occa::schedule    = "compact",
  //                 occa::pinnedCores = "[0, 0, 1, 1]");
  //
  //    device.setup("COI",
  //                 occa::deviceID = 0);
  //========================================================

  o_a  = device.malloc(4*entries*sizeof(float));
  o_b  = device.malloc(4*entries*sizeof(float));
  o_ab = device.malloc(4*entries*sizeof(float));

  // OKL: OCCA Kernel Language
  simd = device.buildKernelFromSource("simd.occa",
                                      "simd");

  //---[ Don't need to set these up when using OKL/OFL ]----
  int dims = 1;
  int itemsPerGroup(32);
  int groups((entries + itemsPerGroup - 1)/itemsPerGroup);

  simd.setWorkingDims(dims, itemsPerGroup, groups);
  //========================================================

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  simd(entries, o_a, o_b, o_ab);

  o_ab.copyTo(ab);

  for(int i = 0; i < 4*entries; ++i){
    if(ab[i] != (a[i] + b[i]))
      throw 1;
  }

  delete [] a;
  delete [] b;
  delete [] ab;

  simd.free();
  o_a.free();
  o_b.free();
  o_ab.free();
  device.free();

  return 0;
}

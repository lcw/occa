// intrinsic operations for OpenMP and simple implementations for OpenCL, CUDA                                                                                                                                          
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#cats=Load&text=_mm256_load&techs=AVX,AVX2&expand=3052
#if (OCCA_USING_OPENMP)

#include "immintrin.h"
#include "xmmintrin.h"
#include "pmmintrin.h"

// vector load operations
inline void occaLoad(const float4 &SRC, float4 &DEST){
  
  *((__m128*)&DEST) = _mm_load_ps((float*)&SRC);
  
}

inline void occaLoad(const float8 &SRC, float8 &DEST){
  
  *((__m256*)&DEST) = _mm256_load_ps((float*)&SRC);
  
}


inline void occaLoad(const float16 &SRC, float16 &DEST){
  
  *((__m512*)&DEST) = _mm512_load_ps((float*)&SRC);
  
}


// vector store operations
inline void occaStore(const float4 &SRC, float4 &DEST){

  _mm_store_ps((float*)&DEST, *((__m128*)&SRC));

}

inline void occaStore(const float8 &SRC, float8 &DEST){

  _mm256_store_ps((float*)&DEST, *((__m256*)&SRC));

}

inline void occaStore(const float16 &SRC, float16 &DEST){

  _mm512_store_ps((float*)&DEST, *((__m512*)&SRC));

}


// entrywise vector add (B += A)
inline void occaAdd(const float4 &A, float4 &B){
  
  *((__m128*)&B) = _mm_add_ps(*((__m128*)&A), *((__m128*)&B));
  
}

inline void occaAdd(const float8 &A, float8 &B){
  
  *((__m256*)&B) = _mm256_add_ps(*((__m256*)&A), *((__m256*)&B));
  
}


inline void occaAdd(const float16 &A, float16 &B){
  
  *((__m512*)&B) = _mm512_add_ps(*((__m512*)&A), *((__m512*)&B));
  
}


// entrywise vector multliply (B += A)
inline void occaMultiply(const float4 &A, float4 &B){
  
  *((__m128*)&B) = _mm_mul_ps(*((__m128*)&A), *((__m128*)&B));
  
}

inline void occaMultiply(const float8 &A, float8 &B){
  
  *((__m256*)&B) = _mm256_mul_ps(*((__m256*)&A), *((__m256*)&B));
  
}


inline void occaMultiply(const float16 &A, float16 &B){
  
  *((__m512*)&B) = _mm512_mul_ps(*((__m512*)&A), *((__m512*)&B));
  
}


// multiply all entries in a vector with the same float
inline void occaMultiply(const float &A, const float4 &B){

  __m128 tmpA =  _mm_load_ps1(&A);
  
  *((__m128*)&B) = _mm_mul_ps(tmpA, *((__m128*)&B));  
}

#if 0
// BROKEN: need to find mm256 analog of load from single float
inline void occaMultiply(const float &A, float8 &B){

  __m256 tmpA =  _mm256_load_ps1(&A);
  
  *((__m256*)&B) = _mm256_mul_ps(tmpA, *((__m256*)&B));  
}

// BROKEN: need to find mm512 analog of load from single float
inline void occaMultiply(const float &A, float16 &B){

  __m512 tmpA =  _mm512_load_ps1(&A);
  
  *((__m512*)&B) = _mm512_mul_ps(tmpA, *((__m512*)&B));  
}
#endif

inline void occaMultiply(const float4 &A, const float &B, float4 &C){

  __m128 tmpB =  _mm_load_ps1(&B);
  
  *((__m128*)&C) = _mm_mul_ps(*((__m128*)&A),tmpB);  
}


#if 0
// BROKEN - need to find float8
inline void occaMultiply(const float8 &A, const float &B, float8 &C){

  __m256 tmpB =  _mm256_load_ps1(&B);
  
  *((__m256*)&C) = _mm256_mul_ps(*((__m256*)&A),tmpB);  
}


inline void occaMultiply(const float16 &A, const float &B, float16 &C){

  __m512 tmpB =  _mm512_load_ps1(&B);
  
  *((__m512*)&C) = _mm512_mul_ps(*((__m512*)&A),tmpB);  
}

#endif


// entrywise vector multliply (B += A)
inline void occaMultiplyAdd(const float4 &A, const float4 &B, float4 &C){
  
  *((__m128*)&C) = _mm_fmadd_ps(*((__m128*)&A), *((__m128*)&B), *((__m128*)&C));
  
}

#if 0
// SOME PROBLEM HERE ?
inline void occaMultiplyAdd(const float8 &A, const float8 &B, float8 &C){
  
  *((__m256*)&C) = _mm256_fmadd_ps(*((__m256*)&A), *((__m256*)&B), *((__m256)&C));
  
}

// SOME PROBLEM HERE ?
inline void occaMultiplyAdd(const float16 &A, const float16 &B, float16 &C ){
  
  *((__m512*)&C) = _mm512_fmadd_ps(*((__m512*)&A), *((__m512*)&B), *((__m512)&C));
  
}
#endif




#endif

#if ( (OCCA_USING_OPENCL) || (OCCA_USING_CUDA) )

inline void occaLoad(const float4  &SRC, float4  &DEST){ DEST = SRC; }
inline void occaLoad(const float8  &SRC, float8  &DEST){ DEST = SRC; }
inline void occaLoad(const float16 &SRC, float16 &DEST){ DEST = SRC; }

inline void occaStore(const float4  &SRC, float4  &DEST){  DEST = SRC; }
inline void occaStore(const float8  &SRC, float8  &DEST){  DEST = SRC; }
inline void occaStore(const float16 &SRC, float16 &DEST){  DEST = SRC; }

inline void occaAdd(const float4  &A, const float4  &B, float4  &APB){  APB = A+B; }
inline void occaAdd(const float8  &A, const float8  &B, float8  &APB){  APB = A+B; }
inline void occaAdd(const float16 &A, const float16 &B, float16 &APB){  APB = A+B; }

inline void occaMultiply(const float4  &A, const float4  &B, float4  &APB){  APB = A*B; }
inline void occaMultiply(const float8  &A, const float8  &B, float8  &APB){  APB = A*B; }
inline void occaMultiply(const float16 &A, const float16 &B, float16 &APB){  APB = A*B; }

inline void occaMultiplyAdd(const float4  &A, const float4  &B, float4  &C){  C += A*B; }
inline void occaMultiplyAdd(const float8  &A, const float8  &B, float8  &C){  C += A*B; }
inline void occaMultiplyAdd(const float16 &A, const float16 &B, float16 &C){  C += A*B; }

#endif

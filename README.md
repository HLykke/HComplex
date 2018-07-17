# HComplex
This is a small library of AVX/AVX2-optimized functions to process vectors of complex numbers.

# Processor requirements
The functions in this repo take advantage of Intel's intrinsic instruction sets AVX and AVX2, thus only AVX-supported CPUs have the opportunity to benefit from this repo. There are two shared libraries to choose from. If your CPU supports AVX, but not AVX2 the functions marked with "(AVX2)" in the documentation below will not work and you should use `libHComplexAVX.so`. Otherwise use `libHComplexAVX2.so`.

# Functions
  `void Hcmul(float *dst, float *a, float *b, const int n)`  
  Complex multiplication: Performs complex multiplication on vectors `a` and `b`, both of length `n`, and stores result in `dst`.
  All vectors are on Intertwined format (see bottom of the page if unfamiliar).
  
  
  `void Hcmul_sep(float *dst_real, float *dst_imag, float *a, float *b, const int n);`  
  Complex multiplication: Performs complex multiplication on vectors `a` and `b`, both of length `n`. The resulting real parts are stored in `dst_real` and the imaginary parts in `dst_imag`. `a` and `b` are on Intertwined format. 
  
  
  `float Hdot(float *a, float *b, const int n)`  
  Dot product: Performs a dot product on regular float vectors (not complex) `a` and `b`, both of length `n`, and returns result.
  
  
  `void Hmagnitude(float *dst, float *src, const int n)` (AVX2)  
  Magnitude: Calculates the magnitude of the complex float vector `src` of length `n` and stores the result in `dst`. `dst` must be on Intertwined format. 
  
  
  `float Hvsum(float *vec, const int n)`  
  Vertical sum: Adds all elements of a vector `vec` of length `n` and returns the resulting sum. 

  
## Intertwined format of complex vectors
[Re{z1}, Im{z1}, Re{z2}, Im{z2}, ...]

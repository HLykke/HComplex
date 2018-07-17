/*
 * =====================================================================================
 *
 *       Filename:  HComplex.h
 *
 *    Description:  AVX-optimized functions to process vectors of complex numbers.
 *
 *        Version:  1.0
 *        Created:  17. juli 2018 kl. 14.31 +0200
 *       Revision:  none
 *       Compiler:  If AVX2 supported CPU:
 						gcc -mavx2 HComplex.c
				else if only AVX supported CPU (Hmagnitude will not work):
						gcc -mavx HComplex.c
 *
 *         Author:  Harald Lykke Joakimsen
 *   Organization:  UiT, the Arctic Univercity of Norway
 *
 * =====================================================================================
 */

#ifndef COMPLEXM_AVX
#define COMPLEXM_AVX
#include "HComplex.c"


// Add up all values in a float vector.
float Hvsum(float *vec, const int n);


// Complex multiplication on float vectors a and b of length n.
// Result stored in dst.
// All vectors are on the form [real1, imag1, real2, imag2, ...]
// [ar1,ai1, ar2,ai2,...]*[br1,bi1, br2,bi2,...]-->[cr1,ci1, cr2,ci2,...]
void Hcmul(float *dst, float *a, float *b, const int n);

// Perform complex multiplication on float vectors a and b of length n
// Real resulting values stored in dst_real and imaginary values dst_imag, both of length n/2.

void Hcmul_sep(float *dst_real, float *dst_imag, float *a, float *b, const int n);

float Hdot(float *a, float *b, const int n);

// If a==b: 0, else: 1
float Hcmp_vecs(__m256 a, __m256 b);

#ifdef AVX2
void Hmagnitude(float *dst, float *src, const int n)
#endif

#endif

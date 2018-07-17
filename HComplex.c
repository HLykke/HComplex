/*
 * =====================================================================================
 *
 *       Filename:  Complexm_avx.c
 *
 *    Description:  Complex multiplication using avx
 *
 *        Version:  1.0
 *        Created:  25. juni 2018 kl. 10.23 +0200
 *       Revision:  none
 *       Compiler:  gcc -mavx <source.c> -lfftw3f
 *
 *         Author:  Harald Lykke Joakimsen
 *   Organization:	Uit, the Arctic University of Norway
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <stdio.h>
#include "immintrin.h"
#include <sys/time.h>


float Hvsum(float *vec, const int n) // Add up all values in a float vector
{
	int iscale = 8;
	int i;
	float sum = 0.0;
 	float temp[8];
	int rest = n%iscale;
	__m256 resvec = _mm256_setzero_ps();

    // [1,3,..<6>..,2,4,..<6>..] --> [3,7,..<6>..]
	for(i=0; i<n-rest; i+=iscale)
	{
		__m256 va = _mm256_loadu_ps(&vec[i]);
        resvec = _mm256_add_ps(resvec, va);
	}
    _mm256_storeu_ps(temp, resvec);

    // [3,7,..<6>..] --> 3+7+..<6>..
    for(i=0;i<iscale;i++)
    {
        sum += temp[i];
    }
    // Adding up the rest with regular C.
    for(i=n-rest;i<n;i++)
    {
        sum += vec[i];
    }
    return sum;
}

// Complex multiplication 1:
// [real_a1,imag_a1,...]x[real_b1,imag_b1,...] = [real_c1,imag_c1,...]
void Hcmul(float *dst, float *a, float *b, const int n)
{
	int rest = n%8;
	__m256 n_mask = _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
	for(int i=0; i<n-rest; i+=8)
	{
		__m256 v1 = _mm256_loadu_ps(&a[i]);
		__m256 v2 = _mm256_loadu_ps(&b[i]);
		__m256 v3 = _mm256_mul_ps(v1, v2);
		v2 = _mm256_permute_ps(v2, 0b10110001);
		v2 = _mm256_mul_ps(v2, n_mask);
		__m256 v4 = _mm256_mul_ps(v1, v2);
		v1 = _mm256_hsub_ps(v3, v4);
		v1 = _mm256_permute_ps(v1, 0b11011000);
		_mm256_storeu_ps(dst+i, v1);
	}

	for(int i=(n-rest)/2; i<n/2; i++)
	{
		dst[2*i] = a[2*i]*b[2*i] - a[2*i+1]*b[2*i+1];
		dst[2*i+1] = a[2*i]*b[2*i+1] + a[2*i+1]*b[2*i];
	}
}

// Complex multiplication 2:
// [real_a1,imag_a1,...]x[real_b1,imag_b1,...] = [real_c1, real_c2,...], [imag_c1, imag_c2]
void Hcmul_sep(float *dst_real, float *dst_imag, float *a, float *b, const int n)
{
	int rest = n%(2*8);
	// negation mask
	__m256 n_mask = _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
	for(int i=0; i<n-rest; i+=2*8)
	{
		__m256 v1 = _mm256_loadu_ps(&a[i]);
		__m256 v2 = _mm256_loadu_ps(&b[i]);
		__m256 v3 = _mm256_mul_ps(v1, v2);
		v2 = _mm256_permute_ps(v2, 0b10110001);
		v2 = _mm256_mul_ps(v2, n_mask);
		__m256 v4 = _mm256_mul_ps(v1, v2);
		v1 = _mm256_hsub_ps(v3, v4);

		__m256 w1 = _mm256_loadu_ps(&a[i + 8]);
		__m256 w2 = _mm256_loadu_ps(&b[i + 8]);
		__m256 w3 = _mm256_mul_ps(w1, w2);
		w2 = _mm256_permute_ps(w2, 0b10110001);
		w2 = _mm256_mul_ps(w2, n_mask);
		__m256 w4 = _mm256_mul_ps(w1, w2);
		w1 = _mm256_hsub_ps(w3, w4);

		__m256 vv1 = _mm256_permute2f128_ps(v1, w1, 0b00100000);
		__m256 ww1 = _mm256_permute2f128_ps(v1, w1, 0b00110001);
		__m256 res_real = (__m256)_mm256_unpacklo_pd((__m256d)vv1, (__m256d)ww1);
		__m256 res_imag = (__m256)_mm256_unpackhi_pd((__m256d)vv1, (__m256d)ww1);

		_mm256_storeu_ps(dst_real+i/2, res_real);
		_mm256_storeu_ps(dst_imag+i/2, res_imag);
	}

    // Taking care of the rest with regular C.
	for (int i=(n-rest)/2; i<n/2; i++)
	{
		dst_real[i] = a[2*i]*b[2*i] - a[2*i+1]*b[2*i+1];
		dst_imag[i] = a[2*i]*b[2*i+1] + a[2*i+1]*b[2*i];
	}
}

// Regular dot product
float Hdot(float *a, float *b, const int n)
{
	int iscale = 8;
	float sum = 0.0; int i;
    float temp[8];
	int rest = n%iscale;
	__m256 resvec = _mm256_setzero_ps();

	for(i=0; i<n-rest; i+=iscale)
	{
		__m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        resvec = _mm256_add_ps(resvec, _mm256_mul_ps(va, vb)); // multiply and store in resvec
	}
    _mm256_storeu_ps(temp, resvec);

    for(i=0;i<iscale;i++)
    {
        sum += temp[i];
    }

    for(i=n-rest;i<n;i++)
    {
        sum += a[i] * b[i];
    }

    return sum;
}

// AVX2 stuff

#ifdef AVX2
void Hmagnitude(float *dst, float *src, const int n)
{
	int rest = n%16; int i;
	for(i=0; i<n-rest; i+=16)
	{
		__m256 a = _mm256_loadu_ps(&src[i]);
		__m256 b = _mm256_loadu_ps(&src[i+8]);
		__m256 a2 = _mm256_mul_ps(a,a);
		__m256 b2 = _mm256_mul_ps(b,b);
		__m256 c = _mm256_hadd_ps(a2,b2);
		c = (__m256)_mm256_permute4x64_pd((__m256d)c, 0b11011000);
		_mm256_storeu_ps(&dst[i], c); //[gmf2[0],gmf2[1],..]
	}

	for(i=n-rest; i<n; i+=2)
	{
		dst[i] = src[i]*src[i] + src[i+1]*src[i+1];
	}
}
#endif

int main() {
	int n = 4;
	//float *a = (float*)malloc(sizeof(float)*n);
	//float *b = (float*)malloc(sizeof(float)*n);

	float *real = (float*)malloc(n/2* sizeof(float));
	float *imag = (float*)malloc(n/2* sizeof(float));
	float *dst = (float*)calloc(n, sizeof(float));

	float a[] = {1,2,3,4};//, 5,6,7,8, 9,0,1,2, 3,4,5,6 ,7,8};
	float b[] = {0,1,2,3};//, 4,5,6,7, 8,9,0,1, 2,3,4,5 ,6,7};
	printf("a = {%f,%f,%f,%f}\n", a[0], a[1], a[2], a[3]);
	printf("b = {%f,%f,%f,%f}\n", b[0], b[1], b[2], b[3]);
	Hcmul((float *)dst,a,b,n);
	Hcmul_sep(real, imag, a, b, n);

	printf("Hcmul_sep of a and b yields:\n\tReal dst: %f %f\n\tImag dst: %f %f\n", real[0], real[1], imag[0], imag[1]);

	printf("Hcmul of a and b yields:\n\tDst: %f %f %f %f\n", dst[0], dst[1], dst[2], dst[3]);

#ifdef AVX2
	Hmagnitude(dst, a, n);
	printf("Hmagnitude of a yields:\n\tDst: %f %f %f %f\n", dst[0], dst[1], dst[2], dst[3]);
#endif

	free(real);
	free(imag);
	free(dst);

	return 0;
}

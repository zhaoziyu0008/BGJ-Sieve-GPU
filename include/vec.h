#ifndef __VECOPS_H
#define __VECOPS_H

#include "config.h"
#include <immintrin.h>

/* aligned double vec operations, n should be divided by 8. */
double dot(double *src1, double *src2, long n);             //return the dot product
void red(double *dst, double *src, double q, long n);       //dst -= src * q
void copy(double *dst, double *src, long n);                //dst = src
void mul(double *dst, double q, long n);                    //dst *= q
void add(double *dst, double *src1, double *src2, long n);  //dst = src1 + src2
void sub(double *dst, double *src1, double *src2, long n);  //dst = src1 - src2

/* n may not divided by 8, the data still need to be aligned. */
double dot_(double *src1, double *src2, long n);

/* aligned float vec operations, n should be divided by 16. */
float dot(float *src1, float *src2, long n);                //return the dot product
float norm(float *src, long n);                             //return dot(a,a,n)
void set_zero(float *dst, long n);                          //dst = 0
void red(float *dst, float *src, float q, long n);          //dst -= src * q
void copy(float *dst, float *src, long n);                  //dst = src
void mul(float *dst, float q, long n);                      //dst *= q
void add(float *dst, float *src, long n);                   //dst += src
void sub(float *dst, float *src, long n);                   //dst -= src
void add(float *dst, float *src1, float *src2, long n);     //dst = src1 + src2
void sub(float *dst, float *src1, float *src2, long n);     //dst = src1 - src2

/* aligned short vec operations, n should be divided by 32. */
void red(short *dst, short *src, short q, long n);          //dst -= src * q
void copy(short *dst, short *src, long n);                  //dst = src
void add(short *dst, short *src1, short *src2, long n);     //dst = src1 + src2
void sub(short *dst, short *src1, short *src2, long n);     //dst = src1 - src2
void add(short *dst, short *src, long n);                   //dst += src
void sub(short *dst, short *src, long n);                   //dst -= src


/* faster for small n. */
float dot_avx2(float *src1, float *src2, long n);                 //n should be divided by 8
float dot_sse(float *src1, float *src2, long n);                  //n should be divided by 4
inline float dot_small(float *src1, float *src2, long n){return dot_sse(src1, src2, n);};
double dot_avx2(double *src1, double *src2, long n);
void mul_avx2(double *dst, double q, long n);
void copy_avx2(double *dst, double *src, long n);
void red_avx2(double *dst, double *src, double q, long n);


inline float dot_aux2(float *src1, float *src2, long n) {
    __m256 r0 = _mm256_setzero_ps();
    __m256 x0;
    long i = 0;
    while (i < n - 7) {
        x0 = _mm256_loadu_ps(src1+i);
        r0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(src2+i), r0);
        i += 8;
    }
    __m128 r128 = _mm_add_ps(_mm256_castps256_ps128(r0), _mm256_extractf128_ps(r0, 1));
    if (i < n - 3) {
        __m128 xx0 = _mm_loadu_ps(src1 + i);
        r128 = _mm_fmadd_ps(xx0, _mm_loadu_ps(src2 + i), r128);
        i += 4;
    }
    r128 = _mm_add_ps(r128, _mm_permute_ps(r128, 78)); 
    r128 = _mm_add_ps(r128, _mm_shuffle_ps(r128, r128, 85));
    float ret = _mm_cvtss_f32(r128);
    while (i < n) {
        ret += src1[i] * src2[i];
        i++;
    }
    return ret;
}
void red_avx2(float *dst, float *src, float q, long n);
void copy_avx2(float *dst, float *src, long n);
inline void mul_avx2(float *dst, float q, long n){
    __m256 q1 = _mm256_set1_ps(q);
    for (long i = 0; i < n; i += 8){
        __m256 x0 = _mm256_load_ps(dst+i);
        _mm256_store_ps(dst+i, _mm256_mul_ps(q1, x0));
    }
    return;
}
inline void set_zero_avx2(float *dst, long n){
    __m256 r;
    r = _mm256_setzero_ps();
    for (long i = 0; i < n; i += 8){
        _mm256_store_ps(dst+i, r);
    }
    return;
}

inline void copy_avx2(int8_t *dst, int8_t *src, long n) {
    for (long i = 0; i < n; i += 32) {
        _mm256_store_si256((__m256i *)(dst+i), _mm256_load_si256((__m256i *)(src+i)));
    }
}
inline void add_avx2(int8_t *dst, int8_t *src, long n) {
    for (long i = 0; i < n; i += 32) {
        _mm256_store_si256((__m256i *)(dst+i), _mm256_add_epi8(_mm256_load_si256((__m256i *)(src+i)), _mm256_load_si256((__m256i *)(dst+i))));
    }
}
inline void sub_avx2(int8_t *dst, int8_t *src, long n) {
    for (long i = 0; i < n; i += 32) {
        _mm256_store_si256((__m256i *)(dst+i), _mm256_sub_epi8(_mm256_load_si256((__m256i *)(dst+i)), _mm256_load_si256((__m256i *)(src+i))));
    }
}

inline void sub_avx2(float *dst, float *src1, float *src2, long n){
    __m256 x0;
    for (long i = 0; i < n; i += 8){
        x0 = _mm256_load_ps(src1+i);
        _mm256_store_ps(dst+i, _mm256_sub_ps(x0, _mm256_load_ps(src2+i)));
    }
    return;
}

inline void add_avx2(float *dst, float *src1, float *src2, long n){
    __m256 x0;
    for (long i = 0; i < n; i += 8){
        x0 = _mm256_load_ps(src1+i);
        _mm256_store_ps(dst+i, _mm256_add_ps(x0, _mm256_load_ps(src2+i)));
    }
    return;
}

/* need not to be aligned. used in LLL_DEEP_QP*/
double tri_dot_slow(double *a, double *b, double *c, long n);

void vec_collect(int8_t *dst, int8_t **src_list, int n, int CSD, int CSD16);

#endif
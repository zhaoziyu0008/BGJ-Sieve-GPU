#include "../include/vec.h"


/* aligned double vec operations, n should be divided by 8. */
double dot(double *src1, double *src2, long n){
#ifdef __AVX512F__
    __m512d r0 = _mm512_setzero_pd();
    for (long i = 0; i < n; i += 8){
        __m512d x0 = _mm512_load_pd(src1 + i);
        r0 = _mm512_fmadd_pd(x0, _mm512_load_pd(src2+i), r0);
    }
    __m256d r256 = _mm256_add_pd(_mm512_castpd512_pd256(r0), _mm512_extractf64x4_pd(r0, 1));
    __m128d r128 = _mm_add_pd(_mm256_castpd256_pd128(r256), _mm256_extractf128_pd(r256, 1));
    r128 = _mm_add_pd(r128, _mm_unpackhi_pd(r128, r128));
    return _mm_cvtsd_f64(r128);
#else
    __m256d r0 = _mm256_setzero_pd();
    for (long i = 0; i < n; i += 4){
        __m256d x0 = _mm256_load_pd(src1 + i);
        r0 = _mm256_fmadd_pd(x0, _mm256_load_pd(src2+i), r0);
    }
    __m128d r128 = _mm_add_pd(_mm256_castpd256_pd128(r0), _mm256_extractf128_pd(r0, 1));
    r128 = _mm_add_pd(r128, _mm_unpackhi_pd(r128, r128));
    return _mm_cvtsd_f64(r128);
#endif
}
void red(double *dst, double *src, double q, long n){
#ifdef __AVX512F__
    __m512d q1 = _mm512_set1_pd(q);
    for (long i = 0; i < n; i += 8){
        __m512d x1 = _mm512_load_pd(dst + i);
        _mm512_store_pd(dst + i, _mm512_fnmadd_pd(_mm512_load_pd(src + i), q1, x1));
    }
    return;
#else
    __m256d q1 = _mm256_set1_pd(q);
    for (long i = 0; i < n; i += 4){
        __m256d x1 = _mm256_load_pd(dst + i);
        _mm256_store_pd(dst + i, _mm256_fnmadd_pd(_mm256_load_pd(src + i), q1, x1));
    }
    return;  
#endif
}
void copy(double *dst, double *src, long n){
#ifdef __AVX512F__
    for (long i = 0; i < n; i += 8){
        _mm512_store_pd(dst + i, _mm512_load_pd(src + i));
    }
    return;
#else
    for (long i = 0; i < n; i += 4){
        _mm256_store_pd(dst + i, _mm256_load_pd(src + i));
    }
    return;
#endif
}
void mul(double *dst, double q, long n){
#ifdef __AVX512F__
    __m512d q1 = _mm512_set1_pd(q);
    for (long i = 0; i < n; i += 8){
        _mm512_store_pd(dst + i, _mm512_mul_pd(_mm512_load_pd(dst + i), q1));
    }
    return;
#else
    __m256d q1 = _mm256_set1_pd(q);
    for (long i = 0; i < n; i += 4){
        _mm256_store_pd(dst + i, _mm256_mul_pd(_mm256_load_pd(dst + i), q1));
    }
    return;
#endif
}
void add(double *dst, double *src1, double *src2, long n){
#ifdef __AVX512F__
    __m512d x0;
    for (long i = 0; i < n; i += 8){
        x0 = _mm512_load_pd(src1+i);
        _mm512_store_pd(dst+i, _mm512_add_pd(_mm512_load_pd(src2+i), x0));
    }
    return;
#else
    __m256d x0;
    for (long i = 0; i < n; i += 4){
        x0 = _mm256_load_pd(src1+i);
        _mm256_store_pd(dst+i, _mm256_add_pd(_mm256_load_pd(src2+i), x0));
    }
    return;
#endif
}
void sub(double *dst, double *src1, double *src2, long n){
#ifdef __AVX512F__
    __m512d x0;
    for (long i = 0; i < n; i += 8){
        x0 = _mm512_load_pd(src1+i);
        _mm512_store_pd(dst+i, _mm512_sub_pd(x0, _mm512_load_pd(src2+i)));
    }
    return;
#else
    __m256d x0;
    for (long i = 0; i < n; i += 4){
        x0 = _mm256_load_pd(src1+i);
        _mm256_store_pd(dst+i, _mm256_sub_pd(x0, _mm256_load_pd(src2+i)));
    }
    return;
#endif
}



/* n may not divided by 8, the data still need to be aligned. */
double dot_(double *src1, double *src2, long n){
#ifdef __AVX512F__
    __m512d r0 = _mm512_setzero_pd();
    __m512d r1 = _mm512_setzero_pd();
    __m512d r2 = _mm512_setzero_pd();
    __m512d r3 = _mm512_setzero_pd();

    __m512d x0,x1,x2,x3,y0,y1,y2,y3;

    long i;
    for(i = 0; i < n/32; i++){
        x0 = _mm512_load_pd(src1+i*32+0);
        x1 = _mm512_load_pd(src1+i*32+8);
        x2 = _mm512_load_pd(src1+i*32+16);
        x3 = _mm512_load_pd(src1+i*32+24);
        
        y0 = _mm512_load_pd(src2+i*32+0);
        y1 = _mm512_load_pd(src2+i*32+8);
        y2 = _mm512_load_pd(src2+i*32+16);
        y3 = _mm512_load_pd(src2+i*32+24);
        
        r0 = _mm512_fmadd_pd(x0,y0,r0);
        r1 = _mm512_fmadd_pd(x1,y1,r1);
        r2 = _mm512_fmadd_pd(x2,y2,r2);
        r3 = _mm512_fmadd_pd(x3,y3,r3);
    }
    i = i * 32;
    if (i < n - 15){
        x0 = _mm512_load_pd(src1+i+0);
        x1 = _mm512_load_pd(src1+i+8);
        
        y0 = _mm512_load_pd(src2+i+0);
        y1 = _mm512_load_pd(src2+i+8);
        
        r0 = _mm512_fmadd_pd(x0,y0,r0);
        r1 = _mm512_fmadd_pd(x1,y1,r1);
        i += 16;
    }
    if (i < n-7){
        x0 = _mm512_load_pd(src1+i+0);
        y0 = _mm512_load_pd(src2+i+0);
        r0 = _mm512_fmadd_pd(x0,y0,r0);
        i = i + 8;
    }
    
    r2 = _mm512_add_pd(r2,r3);
    r0 = _mm512_add_pd(r0,r1);
    r0 = _mm512_add_pd(r0,r2);

    __m256d r256 = _mm256_add_pd(_mm512_castpd512_pd256(r0), _mm512_extractf64x4_pd(r0, 1));
    if (i < n - 3){
        __m256d x0_ = _mm256_load_pd(src1+i);
        __m256d y0_ = _mm256_load_pd(src2+i);
        r256 = _mm256_fmadd_pd(x0_,y0_,r256);
        i = i + 4;		
    }
    __m128d r128 = _mm_add_pd(_mm256_castpd256_pd128(r256), _mm256_extractf128_pd(r256, 1));
    r128 = _mm_add_pd(r128, _mm_unpackhi_pd(r128,r128));

    if (i == n){
        return _mm_cvtsd_f64(r128);
    }
    if (i == n-2){
        return _mm_cvtsd_f64(r128)+src1[n-1]*src2[n-1]+src1[n-2]*src2[n-2];
    }
    if (i == n-3){
        return _mm_cvtsd_f64(r128)+src1[n-1]*src2[n-1]+src1[n-2]*src2[n-2]+src1[n-3]*src2[n-3];
    }
    return _mm_cvtsd_f64(r128)+src1[n-1]*src2[n-1];
#else


    __m256d r0 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
	__m256d r1 = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
	
	__m256d x0,x1,x2,x3,y0,y1,y2,y3;
	long i;

  	for(i = 0; i < n/16; i++) {
    	x0 = _mm256_load_pd(src1+i*16);
		x1 = _mm256_load_pd(src1+i*16+4);
		x2 = _mm256_load_pd(src1+i*16+8);
		x3 = _mm256_load_pd(src1+i*16+12);

    	y0 = _mm256_load_pd(src2+i*16);
		y1 = _mm256_load_pd(src2+i*16+4);
		y2 = _mm256_load_pd(src2+i*16+8);
		y3 = _mm256_load_pd(src2+i*16+12);

    	r0 = _mm256_fmadd_pd(x0,y0,r0);
		r1 = _mm256_fmadd_pd(x1,y1,r1);
		r0 = _mm256_fmadd_pd(x2,y2,r0);
		r1 = _mm256_fmadd_pd(x3,y3,r1);
	}
	i = i * 16;
	if (i < n - 7){
		x0 = _mm256_load_pd(src1+i);
		x1 = _mm256_load_pd(src1+i+4);
		y0 = _mm256_load_pd(src2+i);
		y1 = _mm256_load_pd(src2+i+4);

		r0 = _mm256_fmadd_pd(x0,y0,r0);
		r1 = _mm256_fmadd_pd(x1,y1,r1);
		i = i + 8;		
	}
	if (i < n - 3){
		x0 = _mm256_load_pd(src1+i);
		y0 = _mm256_load_pd(src2+i);
		r0 = _mm256_fmadd_pd(x0,y0,r0);
		i = i + 4;		
	}
	r0 = _mm256_add_pd(r0,r1);

	__m128d r = _mm_add_pd(_mm256_castpd256_pd128(r0), _mm256_extractf128_pd(r0, 1));
    r = _mm_add_pd(r, _mm_unpackhi_pd(r, r));
	if (i == n){
		return _mm_cvtsd_f64(r);
	}
	if (i == n-2){
		return _mm_cvtsd_f64(r)+src1[n-1]*src2[n-1]+src1[n-2]*src2[n-2];
	}
	if (i == n-3){
		return _mm_cvtsd_f64(r)+src1[n-1]*src2[n-1]+src1[n-2]*src2[n-2]+src1[n-3]*src2[n-3];
	}
	return _mm_cvtsd_f64(r)+src1[n-1]*src2[n-1];
#endif
}



/* aligned float vec operations, n should be divided by 16. */
float dot(float *src1, float *src2, long n){
#ifdef __AVX512F__
    __m512 r0 = _mm512_setzero_ps();
    __m512 x0;
    for (long i = 0; i < n; i += 16){
        x0 = _mm512_load_ps(src1+i);
        r0 = _mm512_fmadd_ps(x0, _mm512_load_ps(src2+i), r0);
    }

    __m256 r256 = _mm256_add_ps(_mm512_castps512_ps256(r0), _mm512_extractf32x8_ps(r0, 1));
    __m128 r128 = _mm_add_ps(_mm256_castps256_ps128(r256), _mm256_extractf32x4_ps(r256, 1));
    r128 = _mm_add_ps(r128, _mm_permute_ps(r128, 78)); 
    r128 = _mm_add_ps(r128, _mm_shuffle_ps(r128, r128, 85));
    return _mm_cvtss_f32(r128);
#else
    __m256 r0 = _mm256_setzero_ps();
    __m256 x0;
    for (long i = 0; i < n; i += 8){
        x0 = _mm256_load_ps(src1+i);
        r0 = _mm256_fmadd_ps(x0, _mm256_load_ps(src2+i), r0);
    }
    __m128 r128 = _mm_add_ps(_mm256_castps256_ps128(r0), _mm256_extractf128_ps(r0, 1));
    r128 = _mm_add_ps(r128, _mm_permute_ps(r128, 78)); 
    r128 = _mm_add_ps(r128, _mm_shuffle_ps(r128, r128, 85));
    return _mm_cvtss_f32(r128);
#endif
}
float norm(float *src, long n){
#ifdef __AVX512F__
    __m512 r0 = _mm512_setzero_ps();
    __m512 x0;
    for (long i = 0; i < n; i+=16){
        x0 = _mm512_load_ps(src+i);
        r0 = _mm512_fmadd_ps(x0, x0, r0);
    }

    __m256 r256 = _mm256_add_ps(_mm512_castps512_ps256(r0), _mm512_extractf32x8_ps(r0, 1));
    __m128 r128 = _mm_add_ps(_mm256_castps256_ps128(r256), _mm256_extractf32x4_ps(r256, 1));
    r128 = _mm_add_ps(r128, _mm_permute_ps(r128, 78)); 
    r128 = _mm_add_ps(r128, _mm_shuffle_ps(r128, r128, 85));
    return _mm_cvtss_f32(r128);
#else
    __m256 r0 = _mm256_setzero_ps();
    __m256 x0;
    for (long i = 0; i < n; i+=8){
        x0 = _mm256_load_ps(src+i);
        r0 = _mm256_fmadd_ps(x0, x0, r0);
    }
    __m128 r128 = _mm_add_ps(_mm256_castps256_ps128(r0), _mm256_extractf128_ps(r0, 1));
    r128 = _mm_add_ps(r128, _mm_permute_ps(r128, 78)); 
    r128 = _mm_add_ps(r128, _mm_shuffle_ps(r128, r128, 85));
    return _mm_cvtss_f32(r128);
#endif
}
void set_zero(float *dst, long n){
#ifdef __AVX512F__
    __m512 r;
    r =_mm512_setzero_ps();
    for (long i = 0; i < n; i += 16){
        _mm512_store_ps(dst+i, r);
    }
    return;
#else
    __m256 r;
    r = _mm256_setzero_ps();
    for (long i = 0; i < n; i += 8){
        _mm256_store_ps(dst+i, r);
    }
    return;
#endif
} 
void red(float *dst, float *src, float q, long n){
#ifdef __AVX512F__
    __m512 q1 = _mm512_set1_ps(q);
    __m512 x0;
    for (long i = 0; i < n; i += 16){
        x0 = _mm512_load_ps(dst + i);
        _mm512_store_ps(dst + i, _mm512_fnmadd_ps(_mm512_load_ps(src + i), q1, x0));
    }
    return;
#else
    __m256 q1 = _mm256_set1_ps(q);
    __m256 x0;
    for (long i = 0; i < n; i += 8){
        x0 = _mm256_load_ps(dst+i);
        _mm256_store_ps(dst+i, _mm256_fnmadd_ps(_mm256_load_ps(src+i), q1, x0));
    }
    return;
#endif
}
void copy(float *dst, float *src, long n){
#ifdef __AVX512F__
    __m512 x0;
    for (long i = 0; i < n; i += 16){
        x0 = _mm512_load_ps(src+i);
        _mm512_store_ps(dst + i, x0);
    }
    return;
#else
    __m256 x0;
    for (long i = 0; i < n; i += 8){
        x0 = _mm256_load_ps(src + i);
        _mm256_store_ps(dst + i, x0);
    }
    return;
#endif
}
void mul(float *dst, float q, long n){
#ifdef __AVX512F__
    __m512 q1 = _mm512_set1_ps(q);
    for (long i = 0; i < n; i += 16){
        __m512 x0 = _mm512_load_ps(dst+i);
        _mm512_store_ps(dst+i, _mm512_mul_ps(q1, x0));
    }
    return;
#else
    __m256 q1 = _mm256_set1_ps(q);
    for (long i = 0; i < n; i += 8){
        __m256 x0 = _mm256_load_ps(dst+i);
        _mm256_store_ps(dst+i, _mm256_mul_ps(q1, x0));
    }
    return;
#endif
}
void add(float *dst, float *src, long n){
#ifdef __AVX512F__
    __m512 x0;
    for (long i = 0; i < n; i += 16){
        x0 = _mm512_load_ps(src+i);
        _mm512_store_ps(dst+i, _mm512_add_ps(_mm512_load_ps(dst+i), x0));
    }
    return;
#else
    __m256 x0;
    for (long i = 0; i < n; i += 8){
        x0 = _mm256_load_ps(src+i);
        _mm256_store_ps(dst+i, _mm256_add_ps(_mm256_load_ps(dst+i), x0));
    }
    return;
#endif
}
void sub(float *dst, float *src, long n){
#ifdef __AVX512F__
    __m512 x0;
    for (long i = 0; i < n; i += 16){
        x0 = _mm512_load_ps(src+i);
        _mm512_store_ps(dst+i, _mm512_sub_ps(_mm512_load_ps(dst+i), x0));
    }
    return;
#else
    __m256 x0;
    for (long i = 0; i < n; i += 8){
        x0 = _mm256_load_ps(src+i);
        _mm256_store_ps(dst+i, _mm256_sub_ps(_mm256_load_ps(dst+i), x0));
    }
    return;
#endif
}
void add(float *dst, float *src1, float *src2, long n){
#ifdef __AVX512F__
    __m512 x0;
    for (long i = 0; i < n; i += 16){
        x0 = _mm512_load_ps(src1+i);
        _mm512_store_ps(dst+i, _mm512_add_ps(_mm512_load_ps(src2+i), x0));
    }
    return;
#else
    __m256 x0;
    for (long i = 0; i < n; i += 8){
        x0 = _mm256_load_ps(src1+i);
        _mm256_store_ps(dst+i, _mm256_add_ps(_mm256_load_ps(src2+i), x0));
    }
    return;
#endif
}
void sub(float *dst, float *src1, float *src2, long n){
#ifdef __AVX512F__
    __m512 x0;
    for (long i = 0; i < n; i += 16){
        x0 = _mm512_load_ps(src1+i);
        _mm512_store_ps(dst+i, _mm512_sub_ps(x0, _mm512_load_ps(src2+i)));
    }
    return;
#else
    __m256 x0;
    for (long i = 0; i < n; i += 8){
        x0 = _mm256_load_ps(src1+i);
        _mm256_store_ps(dst+i, _mm256_sub_ps(x0, _mm256_load_ps(src2+i)));
    }
    return;
#endif
}



/* aligned short vec operations, n should be divided by 32. */
void red(short *dst, short *src, short q, long n){
    if (false){
        //todo
        return;
    }
    for (long i = 0; i < n; i++){
        dst[i] -= q *src[i];
    }
    return;
}
void add(short *dst, short *src1, short *src2, long n){
#ifdef __AVX512F__
    __m512i x0;
    for (long i = 0; i < n; i += 32){
        x0 = _mm512_load_epi32(src1 + i);
        _mm512_store_epi32(dst+i, _mm512_add_epi16(x0, _mm512_load_epi32(src2 + i)));
    }
    return;
#else
    for (long i = 0; i < n; i++){
        dst[i] = src1[i] + src2[i];
    }
#endif
}
void sub(short *dst, short *src1, short *src2, long n){
#ifdef __AVX512F__
    __m512i x0;
    for (long i = 0; i < n; i += 32){
        x0 = _mm512_load_epi32(src1 + i);
        _mm512_store_epi32(dst+i, _mm512_sub_epi16(x0, _mm512_load_epi32(src2 + i)));
    }
    return;
#else
    for (long i = 0; i < n; i++){
        dst[i] = src1[i] -src2[i];
    }
#endif
}
void add(short *dst, short *src, long n){
#ifdef __AVX512F__
    __m512i x0;
    for (long i = 0; i < n; i+=32){
        x0 = _mm512_load_epi32(src + i);
        _mm512_store_epi32(dst+i, _mm512_add_epi16(x0, _mm512_load_epi32(dst + i)));
    }
    return;
#else
    for (long i = 0; i < n; i++){
        dst[i] = dst[i] +src[i];
    }
#endif
}
void sub(short *dst, short *src, long n){
#ifdef __AVX512F__
    __m512i x0;
    for (long i = 0; i < n; i+=32){
        x0 = _mm512_load_epi32(src + i);
        _mm512_store_epi32(dst+i, _mm512_sub_epi16(_mm512_load_epi32(dst + i), x0));
    }
    return;
#else
    for (long i = 0; i < n; i++){
        dst[i] = dst[i]-src[i];
    }
#endif
}
void copy(short *dst, short *src, long n){
#ifdef __AVX512F__
    for (long i = 0; i < n; i += 32){
        _mm512_store_epi32(dst + i, _mm512_load_epi32(src + i));
    }
    return;
#else
    
    for (long i = 0; i < n; i++){
        dst[i] = src[i];
    }
#endif
}



/* faster for small n. */
float dot_avx2(float *src1, float *src2, long n){
    __m256 r0 = _mm256_setzero_ps();
    __m256 x0;
    for (long i = 0; i < n; i += 8){
        x0 = _mm256_load_ps(src1+i);
        r0 = _mm256_fmadd_ps(x0, _mm256_load_ps(src2+i), r0);
    }
    __m128 r128 = _mm_add_ps(_mm256_castps256_ps128(r0), _mm256_extractf128_ps(r0, 1));
    r128 = _mm_add_ps(r128, _mm_permute_ps(r128, 78)); 
    r128 = _mm_add_ps(r128, _mm_shuffle_ps(r128, r128, 85));
    return _mm_cvtss_f32(r128);
}
void red_avx2(float *dst, float *src, float q, long n){
    __m256 q1 = _mm256_set1_ps(q);
    __m256 x0;
    for (long i = 0; i < n; i += 8){
        x0 = _mm256_load_ps(dst+i);
        _mm256_store_ps(dst+i, _mm256_fnmadd_ps(_mm256_load_ps(src+i), q1, x0));
    }
    return;
}
void copy_avx2(float *dst, float *src, long n){
    __m256 x0;
    for (long i = 0; i < n; i += 8){
        x0 = _mm256_load_ps(src + i);
        _mm256_store_ps(dst + i, x0);
    }
    return;
}
float dot_sse(float *src1, float *src2, long n){
    __m128 r0 = _mm_setzero_ps();
    __m128 x0;
    for (long i = 0; i < n; i+=4){
        x0 = _mm_load_ps(src1+i);
        r0 = _mm_fmadd_ps(x0, _mm_load_ps(src2+i), r0);
    }
    r0 = _mm_add_ps(r0, _mm_permute_ps(r0, 78)); 
    r0 = _mm_add_ps(r0, _mm_shuffle_ps(r0, r0, 85));
    return _mm_cvtss_f32(r0);
}
double dot_avx2(double *src1, double *src2, long n){
    __m256d r0 = _mm256_setzero_pd();
    for (long i = 0; i < n; i += 4){
        __m256d x0 = _mm256_load_pd(src1 + i);
        r0 = _mm256_fmadd_pd(x0, _mm256_load_pd(src2+i), r0);
    }
    __m128d r128 = _mm_add_pd(_mm256_castpd256_pd128(r0), _mm256_extractf128_pd(r0, 1));
    r128 = _mm_add_pd(r128, _mm_unpackhi_pd(r128, r128));
    return _mm_cvtsd_f64(r128);
}
void mul_avx2(double *dst, double q, long n){
    __m256d q1 = _mm256_set1_pd(q);
    for (long i = 0; i < n; i += 4){
        _mm256_store_pd(dst + i, _mm256_mul_pd(_mm256_load_pd(dst + i), q1));
    }
    return;
}
void copy_avx2(double *dst, double *src, long n){
    for (long i = 0; i < n; i += 4){
        _mm256_store_pd(dst + i, _mm256_load_pd(src + i));
    }
    return;
}
void red_avx2(double *dst, double *src, double q, long n){
    __m256d q1 = _mm256_set1_pd(q);
    for (long i = 0; i < n; i += 4){
        __m256d x1 = _mm256_load_pd(dst + i);
        _mm256_store_pd(dst + i, _mm256_fnmadd_pd(_mm256_load_pd(src + i), q1, x1));
    }
    return;  
}

/* need not to be aligned. */
double tri_dot_slow(double *a, double *b, double *c, long n){
    double ret = 0.0;
    for (long i = 0; i < n; i++){
        ret += a[i] * b[i] * c[i];
    }
    return ret;
}

void vec_collect(int8_t *dst, int8_t **src_list, int n, int CSD, int CSD16) {
    const __mmask64 m0 = CSD > 64 ?  0xffffffffffffffffULL : (1ULL << CSD) - 1;
    const __mmask64 m1 = CSD >= 128 ? 0xffffffffffffffffULL : CSD > 64 ? (1ULL << (CSD - 64)) - 1 : 0;
    const __mmask64 m2 = CSD > 128 ? (1ULL << (CSD - 128)) - 1 : 0;
    
    #if BGJL_HOST_UPK
    const __mmask64 wm0 = CSD16 > 64 ?  0xffffffffffffffffULL : (1ULL << CSD16) - 1;
    const __mmask64 wm1 = CSD16 >= 128 ? 0xffffffffffffffffULL : CSD16 > 64 ? (1ULL << (CSD16 - 64)) - 1 : 0;
    const __mmask64 wm2 = CSD16 > 128 ? (1ULL << (CSD16 - 128)) - 1 : 0;
    #endif

    #pragma unroll
    for (int j = 0; j < n; j++) {
        int8_t *_dst = dst + j * (BGJL_HOST_UPK ? CSD16 : CSD);
        int8_t *_src = src_list[j];
        __m512i v0 = _mm512_maskz_loadu_epi8(m0, _src);
        __m512i v1 = _mm512_maskz_loadu_epi8(m1, _src + 64);
        __m512i v2 = _mm512_maskz_loadu_epi8(m2, _src + 128);
        _mm512_mask_storeu_epi8(_dst, (BGJL_HOST_UPK ? wm0 : m0), v0);
        _mm512_mask_storeu_epi8(_dst + 64, (BGJL_HOST_UPK ? wm1 : m1), v1);
        _mm512_mask_storeu_epi8(_dst + 128, (BGJL_HOST_UPK ? wm2 : m2), v2);
    }
}
                    
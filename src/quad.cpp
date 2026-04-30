#include "../include/quad.h"
#include "../include/vec.h"
#include <immintrin.h>


/* naive vec operations, n can be any integer, data need not to be aligned. */
NTL::quad_float dot(NTL::quad_float *src1, NTL::quad_float *src2, long n){
    NTL::quad_float ret(0.0);
    for (long i = 0; i < n; i++){
        ret += src1[i] * src2[i];
    }
    return ret;
}
void red(NTL::quad_float *dst, NTL::quad_float *src, NTL::quad_float q, long n){
    for (long i = 0; i < n; i++){
        dst[i] -= q * src[i];
    }
}
void copy(NTL::quad_float *dst, NTL::quad_float *src, long n){
    for (long i = 0; i < n; i++){
        dst[i] = src[i];
    }
}
void mul(NTL::quad_float *dst, NTL::quad_float q, long n){
    for (long i = 0; i < n; i++){
        dst[i] *= q;
    }
}
void sub(NTL::quad_float *dst, NTL::quad_float q, long n){
    for (long i = 0; i < n; i++){
        dst[i] -= q;
    }
}


/* aligned vec operations, n should be divided by 8. 
 * currently only the avx-512 version is done. */
NTL::quad_float dot(double *src1_hi, double *src1_lo, double *src2_hi, double *src2_lo, long n){
#if HAVE_AVX512 <= 0
    //todo
    NTL::quad_float ret(0.0);
    for (long i = 0; i < n; i++){
        ret += NTL::quad_float(src1_hi[i], src1_lo[i]) * NTL::quad_float(src2_hi[i], src2_lo[i]);
    }
    return ret;
#else
    register __m512d z7 asm("zmm7") = _mm512_set1_pd(((((double)(1L<<27)))+1.0));
    double dot_hi, dot_lo;
    __asm__ __volatile__(
        "test %%r8, %%r8;"
        "vxorpd %%zmm0, %%zmm0, %%zmm0;"            //store current.hi
        "vxorpd %%zmm1, %%zmm1, %%zmm1;"              //store current.lo
        "jle 1f;"
        "xor %%rax, %%rax;"
        //begin mul
        "2: vmovapd (%%rdx, %%rax, 8), %%zmm2;"			//load y.hi
        "vmovapd (%%rdi, %%rax, 8), %%zmm3;"			//load x.hi
        "vmulpd %%zmm7, %%zmm3, %%zmm5;"
        "vmulpd %%zmm7, %%zmm2, %%zmm4;"
        "vsubpd %%zmm3, %%zmm5, %%zmm6;"
        "vsubpd %%zmm2, %%zmm4, %%zmm8;"
        "vsubpd %%zmm6, %%zmm5, %%zmm5;"
        "vmulpd %%zmm2, %%zmm3, %%zmm6;"
        "vsubpd %%zmm8, %%zmm4, %%zmm4;"
        "vsubpd %%zmm5, %%zmm3, %%zmm9;"
        "vmulpd (%%rcx, %%rax, 8), %%zmm3, %%zmm3;"		//load y.lo
        "vsubpd %%zmm4, %%zmm2, %%zmm10;"		
        "vmulpd %%zmm4, %%zmm5, %%zmm8;"
        "vmulpd (%%rsi, %%rax, 8), %%zmm2, %%zmm2;"		//load x.lo
        "vmulpd %%zmm4, %%zmm9, %%zmm4;"
        "vsubpd %%zmm6, %%zmm8, %%zmm8;"
        "vmulpd %%zmm10, %%zmm5, %%zmm5;"
        "vmulpd %%zmm10, %%zmm9, %%zmm9;"
        "vaddpd %%zmm2, %%zmm3, %%zmm3;"
        "vaddpd %%zmm8, %%zmm5, %%zmm5;"
        "vaddpd %%zmm5, %%zmm4, %%zmm4;"
        "vaddpd %%zmm9, %%zmm4, %%zmm4;"
        "vaddpd %%zmm4, %%zmm3, %%zmm4;"
        "vaddpd %%zmm4, %%zmm6, %%zmm3;"        //z3 = result.hi
        "vsubpd %%zmm3, %%zmm6, %%zmm6;"
        "vaddpd %%zmm6, %%zmm4, %%zmm4;"        //z4 = result.lo

        //begin add, currently z0, z1, z3, z4, z7 are used
        "vaddpd %%zmm3, %%zmm0, %%zmm5;"
        "vaddpd %%zmm4, %%zmm1, %%zmm2;"
        "vsubpd %%zmm0, %%zmm5, %%zmm6;"
        "vsubpd %%zmm1, %%zmm2, %%zmm8;"
        "vsubpd %%zmm6, %%zmm5, %%zmm9;"
        "vsubpd %%zmm6, %%zmm3, %%zmm3;"
        "vsubpd %%zmm8, %%zmm4, %%zmm4;"
        "vsubpd %%zmm9, %%zmm0, %%zmm0;"
        "vaddpd %%zmm3, %%zmm0, %%zmm0;"
        "vsubpd %%zmm8, %%zmm2, %%zmm3;"
        "vaddpd %%zmm0, %%zmm2, %%zmm2;"
        "vsubpd %%zmm3, %%zmm1, %%zmm1;"
        "vaddpd %%zmm2, %%zmm5, %%zmm9;"
        "vaddpd %%zmm4, %%zmm1, %%zmm1;"
        "vsubpd %%zmm9, %%zmm5, %%zmm5;"
        "vaddpd %%zmm5, %%zmm2, %%zmm2;"
        "vaddpd %%zmm2, %%zmm1, %%zmm3;"
        "vaddpd %%zmm3, %%zmm9, %%zmm0;"        //z0 = local_dot.hi
        "vsubpd %%zmm0, %%zmm9, %%zmm10;"
        "vaddpd %%zmm10, %%zmm3, %%zmm1;"       //z1 = loacl_dot.lo
        "add $0x8, %%rax;"
        "cmp %%r8, %%rax;"
        "jl 2b;"
        //now z0 = dot.hi, z1 = dot.lo
        "vextractf64x4 $0x1, %%zmm0, %%ymm3;"
        "vextractf64x4 $0x1, %%zmm1, %%ymm4;"

        "vaddpd %%ymm3,%%ymm0,%%ymm5;"
        "vaddpd %%ymm4,%%ymm1,%%ymm2;"
        "vsubpd %%ymm0,%%ymm5,%%ymm6;"
        "vsubpd %%ymm1,%%ymm2,%%ymm8;"
        "vsubpd %%ymm6,%%ymm5,%%ymm9;"
        "vsubpd %%ymm6,%%ymm3,%%ymm3;"
        "vsubpd %%ymm8,%%ymm4,%%ymm4;"
        "vsubpd %%ymm9,%%ymm0,%%ymm0;"
        "vaddpd %%ymm3,%%ymm0,%%ymm0;"
        "vsubpd %%ymm8,%%ymm2,%%ymm3;"
        "vaddpd %%ymm0,%%ymm2,%%ymm2;"
        "vsubpd %%ymm3,%%ymm1,%%ymm1;"
        "vaddpd %%ymm2,%%ymm5,%%ymm9;"
        "vaddpd %%ymm4,%%ymm1,%%ymm1;"
        "vsubpd %%ymm9,%%ymm5,%%ymm5;"
        "vaddpd %%ymm5,%%ymm2,%%ymm2;"
        "vaddpd %%ymm2,%%ymm1,%%ymm3;"
        "vaddpd %%ymm3,%%ymm9,%%ymm0;"		    //store x.hi
        "vsubpd %%ymm0,%%ymm9,%%ymm10;"
        "vaddpd %%ymm10,%%ymm3,%%ymm1;"		    //store x.lo

        "vextractf64x2 $0x1, %%ymm0, %%xmm3;"
        "vextractf64x2 $0x1, %%ymm1, %%xmm4;"

        "vaddpd %%xmm3,%%xmm0,%%xmm5;"
        "vaddpd %%xmm4,%%xmm1,%%xmm2;"
        "vsubpd %%xmm0,%%xmm5,%%xmm6;"
        "vsubpd %%xmm1,%%xmm2,%%xmm8;"
        "vsubpd %%xmm6,%%xmm5,%%xmm9;"
        "vsubpd %%xmm6,%%xmm3,%%xmm3;"
        "vsubpd %%xmm8,%%xmm4,%%xmm4;"
        "vsubpd %%xmm9,%%xmm0,%%xmm0;"
        "vaddpd %%xmm3,%%xmm0,%%xmm0;"
        "vsubpd %%xmm8,%%xmm2,%%xmm3;"
        "vaddpd %%xmm0,%%xmm2,%%xmm2;"
        "vsubpd %%xmm3,%%xmm1,%%xmm1;"
        "vaddpd %%xmm2,%%xmm5,%%xmm9;"
        "vaddpd %%xmm4,%%xmm1,%%xmm1;"
        "vsubpd %%xmm9,%%xmm5,%%xmm5;"
        "vaddpd %%xmm5,%%xmm2,%%xmm2;"
        "vaddpd %%xmm2,%%xmm1,%%xmm3;"
        "vaddpd %%xmm3,%%xmm9,%%xmm0;"		    //store x.hi
        "vsubpd %%xmm0,%%xmm9,%%xmm10;"
        "vaddpd %%xmm10,%%xmm3,%%xmm1;"		    //store x.lo

        "vunpckhpd %%xmm0, %%xmm0, %%xmm3;"
        "vunpckhpd %%xmm1, %%xmm1, %%xmm4;"

        "vaddpd %%xmm3,%%xmm0,%%xmm5;"
        "vaddpd %%xmm4,%%xmm1,%%xmm2;"
        "vsubpd %%xmm0,%%xmm5,%%xmm6;"
        "vsubpd %%xmm1,%%xmm2,%%xmm8;"
        "vsubpd %%xmm6,%%xmm5,%%xmm9;"
        "vsubpd %%xmm6,%%xmm3,%%xmm3;"
        "vsubpd %%xmm8,%%xmm4,%%xmm4;"
        "vsubpd %%xmm9,%%xmm0,%%xmm0;"
        "vaddpd %%xmm3,%%xmm0,%%xmm0;"
        "vsubpd %%xmm8,%%xmm2,%%xmm3;"
        "vaddpd %%xmm0,%%xmm2,%%xmm2;"
        "vsubpd %%xmm3,%%xmm1,%%xmm1;"
        "vaddpd %%xmm2,%%xmm5,%%xmm9;"
        "vaddpd %%xmm4,%%xmm1,%%xmm1;"
        "vsubpd %%xmm9,%%xmm5,%%xmm5;"
        "vaddpd %%xmm5,%%xmm2,%%xmm2;"
        "vaddpd %%xmm2,%%xmm1,%%xmm3;"
        "vaddpd %%xmm3,%%xmm9,%0;"		    //store x.hi
        "vsubpd %0,%%xmm9,%%xmm10;"
        "vaddpd %%xmm10,%%xmm3,%1;"		    //store x.lo
        "1:"
        : "=x"(dot_hi), "=x"(dot_lo)
        : "x"(z7)
        : "%rax", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm8", "%zmm9", "%zmm10"
    );
    return NTL::quad_float(dot_hi, dot_lo);
#endif
}
void red(double *dst_hi, double *dst_lo, double *src_hi, double *src_lo, NTL::quad_float q, long n){
#if HAVE_AVX512 <= 0
    //todo
    for (long i = 0; i < n; i++){
        NTL::quad_float x(dst_hi[i], dst_lo[i]);
        x -= q * NTL::quad_float(src_hi[i], src_lo[i]);
        dst_hi[i] = x.hi;
        dst_lo[i] = x.lo;
    }
    return;
#else
    register __m512d z7 asm("zmm7") = _mm512_set1_pd(((((double)(1L<<27)))+1.0));
    register __m512d z14 asm("zmm14") = _mm512_set1_pd((-0.0));
    __asm__ __volatile__(
        "test %%r8, %%r8;"
        "vbroadcastsd %%xmm0, %%zmm0;"
        "vbroadcastsd %%xmm1, %%zmm1;"
        "jle 1f;"
        "vmulpd %%zmm0, %%zmm7, %%zmm6;"
        "vsubpd %%zmm0, %%zmm6, %%zmm8;"
        "vsubpd %%zmm8, %%zmm6, %%zmm8;"
        "vsubpd %%zmm8, %%zmm0, %%zmm6;"            //z8 = hy, z6 = ty
        "xor %%rax, %%rax;"
        "2: vmovapd (%%rdx, %%rax, 8), %%zmm2;"     //loop begins
        //compute mul
        "vmulpd %%zmm2, %%zmm7, %%zmm3;"
        "vsubpd %%zmm2, %%zmm3, %%zmm4;"
        "vsubpd %%zmm4, %%zmm3, %%zmm4;"
        "vsubpd %%zmm4, %%zmm2, %%zmm3;"           //z4 = hx, z3 = tx
        "vmulpd %%zmm0, %%zmm2, %%zmm9;"           //z9 = C
        "vmulpd %%zmm4, %%zmm8, %%zmm10;"          //z10 = hx * hy
        "vmulpd %%zmm4, %%zmm6, %%zmm11;"          //z11 = hx * ty
        "vmulpd %%zmm3, %%zmm8, %%zmm12;"          //z12 = tx * hy
        "vmulpd %%zmm3, %%zmm6, %%zmm13;"          //z13 = tx * ty
        "vsubpd %%zmm9, %%zmm10, %%zmm10;"
        "vaddpd %%zmm11, %%zmm10, %%zmm10;"
        "vaddpd %%zmm12, %%zmm10, %%zmm10;"
        "vaddpd %%zmm13, %%zmm10, %%zmm10;"                 //z10 = c
        "vmulpd %%zmm2, %%zmm1, %%zmm3;"                    //z3 = t1
        "vmulpd (%%rcx, %%rax, 8), %%zmm0, %%zmm5;"         //z5 = t2
        "vaddpd %%zmm3, %%zmm5, %%zmm3;"
        "vaddpd %%zmm3, %%zmm10, %%zmm10;"
        "vaddpd %%zmm9, %%zmm10, %%zmm4;"           //z4 = (q * src).hi
        "vsubpd %%zmm4, %%zmm9, %%zmm3;"                
        "vaddpd %%zmm3, %%zmm10, %%zmm3;"           //z3 = (q * src).lo

        //compute sub, currently z3, z4, z0, z1, z6, z7, z8, z14 are used
        "vmovapd (%%rdi, %%rax, 8), %%zmm2;"		//x.hi
        "vsubpd %%zmm4, %%zmm2, %%zmm9;"	
        "vxorpd %%zmm14, %%zmm4, %%zmm4;"
        "vmovapd (%%rsi, %%rax, 8), %%zmm5;"		//x.lo
        "vsubpd %%zmm2, %%zmm9, %%zmm10;"
        "vsubpd %%zmm3, %%zmm5, %%zmm11;"
        "vxorpd %%zmm14, %%zmm3, %%zmm3;"
        "vsubpd %%zmm10, %%zmm9, %%zmm12;"
        "vsubpd %%zmm10, %%zmm4, %%zmm4;"
        "vsubpd %%zmm5, %%zmm11, %%zmm13;"
        "vsubpd %%zmm12, %%zmm2, %%zmm2;"
        "vsubpd %%zmm13, %%zmm11, %%zmm10;"
        "vsubpd %%zmm13, %%zmm3, %%zmm3;"
        "vaddpd %%zmm4, %%zmm2, %%zmm2;"
        "vsubpd %%zmm10, %%zmm5, %%zmm5;"
        "vaddpd %%zmm11, %%zmm2, %%zmm2;"
        "vaddpd %%zmm3, %%zmm5, %%zmm5;"
        "vaddpd %%zmm9, %%zmm2, %%zmm4;"
        "vsubpd %%zmm4, %%zmm9, %%zmm9;"
        "vaddpd %%zmm9, %%zmm2, %%zmm2;"
        "vaddpd %%zmm2, %%zmm5, %%zmm5;"
        "vaddpd %%zmm5, %%zmm4, %%zmm2;"
        "vsubpd %%zmm2, %%zmm4, %%zmm4;"
        "vmovapd %%zmm2, (%%rdi, %%rax, 8);"		//store to x.hi
        "vaddpd %%zmm4, %%zmm5, %%zmm5;"
        "vmovapd %%zmm5, (%%rsi, %%rax, 8);"		//store to x.lo

        "add $0x8, %%rax;"
        "cmp %%r8, %%rax;"
        "jl 2b;"
        "1:"
        :
        : "x"(z7), "x"(z14)
        : "%rax", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13"
    );
#endif
}
void copy(double *dst_hi, double *dst_lo, double *src_hi, double *src_lo, long n){
    copy(dst_hi, src_hi, n);
    copy(dst_lo, src_lo, n);
}
void mul(double *dst_hi, double *dst_lo, NTL::quad_float q, long n){
#if HAVE_AVX512 <= 0
    //todo
    for (long i = 0; i < n; i++){
        NTL::quad_float x(dst_hi[i], dst_lo[i]);
        x *= q;
        dst_hi[i] = x.hi;
        dst_lo[i] = x.lo;
    }
    return;
#else
    register __m512d z7 asm("zmm7") = _mm512_set1_pd(((((double)(1L<<27)))+1.0));
    __asm__ __volatile__(
        "test %%rdx, %%rdx;"
        "vbroadcastsd %%xmm0, %%zmm0;"
        "vbroadcastsd %%xmm1, %%zmm1;"
        "jle 1f;"
        "vmulpd %%zmm0, %%zmm7, %%zmm6;"
        "vsubpd %%zmm0, %%zmm6, %%zmm8;"
        "vsubpd %%zmm8, %%zmm6, %%zmm8;"
        "vsubpd %%zmm8, %%zmm0, %%zmm6;"           //z8 = hy, z6 = ty
        //"sub $0x1, %%rdx;"
        //"and $0xfffffffffffffff8, %%rdx;"
        //"add $0x8, %%rdx;"
        "xor %%rax, %%rax;"
        "2: vmovapd (%%rdi, %%rax, 8), %%zmm2;"
        "vmulpd %%zmm2, %%zmm7, %%zmm3;"
        "vsubpd %%zmm2, %%zmm3, %%zmm4;"
        "vsubpd %%zmm4, %%zmm3, %%zmm4;"
        "vsubpd %%zmm4, %%zmm2, %%zmm3;"           //z4 = hx, z3 = tx

        "vmulpd %%zmm0, %%zmm2, %%zmm9;"           //z9 = C
        "vmulpd %%zmm4, %%zmm8, %%zmm10;"          //z10 = hx * hy
        "vmulpd %%zmm4, %%zmm6, %%zmm11;"          //z11 = hx * ty
        "vmulpd %%zmm3, %%zmm8, %%zmm12;"          //z12 = tx * hy
        "vmulpd %%zmm3, %%zmm6, %%zmm13;"          //z13 = tx * ty
        "vsubpd %%zmm9, %%zmm10, %%zmm10;"
        "vaddpd %%zmm11, %%zmm10, %%zmm10;"
        "vaddpd %%zmm12, %%zmm10, %%zmm10;"
        "vaddpd %%zmm13, %%zmm10, %%zmm10;"                        //z10 = c
        "vmulpd %%zmm2, %%zmm1, %%zmm3;"                           //z3 = t1
        "vmulpd (%%rsi, %%rax, 8), %%zmm0, %%zmm5;"         //z5 = t2
        "vaddpd %%zmm3, %%zmm5, %%zmm3;"
        "vaddpd %%zmm3, %%zmm10, %%zmm10;"
        "vaddpd %%zmm9, %%zmm10, %%zmm4;"              //z4 = hx
        "vsubpd %%zmm4, %%zmm9, %%zmm3;"                
        "vaddpd %%zmm3, %%zmm10, %%zmm3;"              //z3 = tx
        "vmovapd %%zmm4, (%%rdi, %%rax, 8);"
        "vmovapd %%zmm3, (%%rsi, %%rax, 8);"
        "add $0x8, %%rax;"
        "cmp %%rdx, %%rax;"
        "jl 2b;"
        "1:"
        :
        : "x"(z7)
        : "%rax", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13"
    );
#endif
}
void sub(double *dst_hi, double *dst_lo, NTL::quad_float q, long n){
#if HAVE_AVX512 <= 0
    //todo
    for (long i = 0; i < n; i++){
        NTL::quad_float x(dst_hi[i], dst_lo[i]);
        x -= q;
        dst_hi[i] = x.hi;
        dst_lo[i] = x.lo;
    }
    return;
#else
    register __m512d z7 asm("zmm7") = _mm512_set1_pd((-0.0));
    __asm__ __volatile__ (
        "test %%rdx, %%rdx;"
        "vbroadcastsd %%xmm0, %%zmm0;"
        "vbroadcastsd %%xmm1, %%zmm1;"
        "jle 1f;"
        "vxorpd %%zmm7, %%zmm0, %%zmm0;"
        "vxorpd %%zmm7, %%zmm1, %%zmm1;"
        "xor %%rax, %%rax;"
        "2: vmovapd (%%rdi, %%rax, 8), %%zmm2;"		//x.hi
        "vaddpd %%zmm0, %%zmm2, %%zmm4;"
        "vmovapd (%%rsi, %%rax, 8), %%zmm3;"		//x.lo
        "vsubpd %%zmm2, %%zmm4, %%zmm5;"
        "vaddpd %%zmm1, %%zmm3, %%zmm6;"
        "vsubpd %%zmm5, %%zmm4, %%zmm10;"
        "vsubpd %%zmm5, %%zmm0, %%zmm8;"
        "vsubpd %%zmm3, %%zmm6, %%zmm11;"
        "vsubpd %%zmm10, %%zmm2, %%zmm2;"
        "vsubpd %%zmm11, %%zmm6, %%zmm5;"
        "vsubpd %%zmm11, %%zmm1, %%zmm9;"
        "vaddpd %%zmm8, %%zmm2, %%zmm2;"
        "vsubpd %%zmm5, %%zmm3, %%zmm3;"
        "vaddpd %%zmm6, %%zmm2, %%zmm2;"
        "vaddpd %%zmm9, %%zmm3, %%zmm3;"
        "vaddpd %%zmm4, %%zmm2, %%zmm8;"
        "vsubpd %%zmm8, %%zmm4, %%zmm4;"
        "vaddpd %%zmm4, %%zmm2, %%zmm2;"
        "vaddpd %%zmm2, %%zmm3, %%zmm3;"
        "vaddpd %%zmm3, %%zmm8, %%zmm2;"
        "vsubpd %%zmm2, %%zmm8, %%zmm8;"
        "vmovapd %%zmm2, (%%rdi, %%rax, 8);"	//store x.hi
        "vaddpd %%zmm8, %%zmm3, %%zmm3;"
        "vmovapd %%zmm3, (%%rsi, %%rax, 8);"	//store x.lo

        "add $0x8, %%rax;"
        "cmp %%rdx, %%rax;"
        "jl 2b;"
        "1:"
        :
        : "x"(z7)
        : "%rax", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm8", "%zmm9", "%zmm10", "%zmm11"
    );
#endif
}

/* operations on VEC_QP */
NTL::quad_float dot(VEC_QP src1, VEC_QP src2, long n){
    return dot(src1.hi, src1.lo, src2.hi, src2.lo, n);
}
void red(VEC_QP dst, VEC_QP src, NTL::quad_float q, long n){
    red(dst.hi, dst.lo, src.hi, src.lo, q, n);
}
void copy(VEC_QP dst, VEC_QP src, long n){
    copy(dst.hi, dst.lo, src.hi, src.lo, n);
}
void mul(VEC_QP dst, NTL::quad_float q, long n){
    mul(dst.hi, dst.lo, q, n);
}
void sub(VEC_QP dst, NTL::quad_float q, long n){
    sub(dst.hi, dst.lo, q, n);
}
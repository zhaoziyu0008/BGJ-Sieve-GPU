#ifndef __POOL_HD_DEVICE_H
#define __POOL_HD_DEVICE_H

#include "pool_hd.h"
#include "utils.h"
#include "common_device.h"

#include <mma.h>
#include <cuda_runtime.h>

using namespace nvcuda;

struct local_data_t {
    static constexpr unsigned int vec_nbytes    = Pool_hd_t::vec_nbytes;
    static constexpr unsigned int max_boost_dim = boost_data_t::max_boost_dim;

    int8_t b_dual[vec_nbytes * (vec_nbytes + 16) / 2];
    float b_full[vec_nbytes * (max_boost_dim + vec_nbytes)];
    float b_ext_head[max_boost_dim * (max_boost_dim + 2) / 2];
    float igh[max_boost_dim];
    float inorm[max_boost_dim];
    uint64_t uid_coeff[vec_nbytes];

    int32_t CSD;
    int32_t ESD;
    int32_t dhalf;
    int32_t dshift;  
};

constexpr uint32_t Ver_rc = 0;
constexpr uint32_t Ver_el = 1;
constexpr uint32_t Ver_sl = 2;
constexpr uint32_t Ver_tl = 3;
constexpr uint32_t Ver_ml = 4;
constexpr uint32_t Ver_it = 5;
constexpr uint32_t Ver_ds = 6;

template <uint32_t Ver>
struct pdev_traits_t {
    static constexpr unsigned int vec_nbytes    = Pool_hd_t::vec_nbytes;
    static constexpr unsigned int max_boost_dim = boost_data_t::max_boost_dim;
    
    static constexpr unsigned int blockWarps    = 8;
    static constexpr unsigned int blockThreads  = 256;
    static constexpr unsigned int kernelBlocks  = 128;
    static constexpr unsigned int taskChunks    = 8;
    static constexpr unsigned int taskVecs      = taskChunks * Pool_hd_t::chunk_max_nvecs;
    static constexpr unsigned int scoreBias     = taskVecs * (vec_nbytes);
    static constexpr unsigned int normBias      = taskVecs * (vec_nbytes + 2);
    static constexpr unsigned int uBias         = taskVecs * (vec_nbytes + 2 + 4);
    static constexpr unsigned int dynamic_shmem = Ver < Ver_ds ? 68992 : 45056;
    static constexpr unsigned int b_full_active_nbytes = 
        Ver == Ver_rc ? 101376 : Ver == Ver_el ? 111616 : Ver == Ver_sl ? 101376 : 
        Ver == Ver_tl ? 157696 : Ver == Ver_ml ? 101376 : Ver == Ver_it ? 157696 : 0;

    static void (*const kernel)(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data);
    
    static __device__ __forceinline__ void _fmma8x16x8(float *C, float *A, float *B) {
        float A_col[16];
        float B_col[16];

        float4 *A4 = (float4 *)A;
        float4 *B4 = (float4 *)B;
        float4 *A_col4 = (float4 *)A_col;
        float4 *B_col4 = (float4 *)B_col;

        // k = -1 (prefetch only)
        A_col4[0] = A4[0];
        A_col4[1] = A4[1];
        B_col4[0] = B4[0];
        B_col4[1] = B4[1];

        #pragma unroll
        for (int k = 0; k < 16; k++) {
            if (k < 15) {
                A_col4[((k + 1) & 0x1) * 2 + 0] = A4[(k + 1) * 2 + 0];
                A_col4[((k + 1) & 0x1) * 2 + 1] = A4[(k + 1) * 2 + 1];
                B_col4[((k + 1) & 0x1) * 2 + 0] = B4[(k + 1) * 2 + 0];
                B_col4[((k + 1) & 0x1) * 2 + 1] = B4[(k + 1) * 2 + 1];
            }

            for (int i = 0; i < 8; i++) {
                if (i & 0x1) for (int j = 7; j >= 0; j--) C[i * 8 + j] += A_col[(k & 0x1) * 8 + i] * B_col[(k & 0x1) * 8 + j];
                else for (int j = 0; j < 8; j++) C[i * 8 + j] += A_col[(k & 0x1) * 8 + i] * B_col[(k & 0x1) * 8 + j];
            }
        }

        return;
    }
    static __device__ __forceinline__ void _fmma4x16x8(float *C, float *A, float *B) {
        float A_col[8];
        float B_col[16];

        float4 *A4 = (float4 *)A;
        float4 *B4 = (float4 *)B;
        float4 *A_col4 = (float4 *)A_col;
        float4 *B_col4 = (float4 *)B_col;

        // k = -1 (prefetch only)
        A_col4[0] = A4[0];
        B_col4[0] = B4[0];
        B_col4[1] = B4[1];

        #pragma unroll
        for (int k = 0; k < 16; k++) {
            if (k < 15) {
                A_col4[(k + 1) & 0x1] = A4[(k + 1) * 2];
                B_col4[((k + 1) & 0x1) * 2 + 0] = B4[(k + 1) * 2 + 0];
                B_col4[((k + 1) & 0x1) * 2 + 1] = B4[(k + 1) * 2 + 1];
            }

            for (int i = 0; i < 4; i++) {
                if (i & 0x1) for (int j = 7; j >= 0; j--) C[i * 8 + j] += A_col[(k & 0x1) * 4 + i] * B_col[(k & 0x1) * 8 + j];
                else for (int j = 0; j < 8; j++) C[i * 8 + j] += A_col[(k & 0x1) * 4 + i] * B_col[(k & 0x1) * 8 + j];
            }
        }
        
        return;
    }
    static __device__ __forceinline__ void _kernel_init(int8_t *b_dual_tri, float *b_ext_head, 
                                                       uint64_t *uid_coeff, float *inorm, float *igh, 
                                                       local_data_t *ld, uint32_t ESD8, uint32_t tid) {
        for (int i = tid; i < sizeof(local_data_t::b_dual) / sizeof(int4); i += 256) 
            ((int4 *)b_dual_tri)[i] = ((int4 *)ld->b_dual)[i];
        for (int i = tid; i < sizeof(local_data_t::b_ext_head) / sizeof(int4); i += 256) 
            ((int4 *)b_ext_head)[i] = ((int4 *)ld->b_ext_head)[i];
                                                                        
        if (tid < 176) uid_coeff[tid] = ld->uid_coeff[tid];
        if (ESD8 && tid < max_boost_dim) {
            inorm[tid] = ld->inorm[tid];
            igh[tid] = ld->igh[tid];
        }
    }
    static __device__ __forceinline__ void _batch_init(float *frag0, float *frag1, uint16_t sbuf[128], 
                                                       uint32_t size, uint32_t wid, uint32_t lid) {
        if (wid * 16 < size) {
            for (int i = 0; i < 64; i++) frag0[i] = 0.0f;
        }
        if (wid * 16 + 128 < size) {
            for (int i = 0; i < 64; i++) frag1[i] = 0.0f;
        }
        if (wid == 0) ((uint64_t *)sbuf)[lid] = 0xffffffffffffffffULL;
    }
    static __device__ __forceinline__ void _prep_vec(int8_t *dst, int8_t *src, uint32_t tid) {
        for (int i = 0; i < 11; i++) utils_t::_ldgsts_64b_async(dst + 8 * tid + 8 * 256 * i, 
                                                                src + 8 * tid + 8 * 256 * i);
        utils_t::_commit_async_group();
    }
    static __device__ __forceinline__ void _prep_bfull(float *dst, float *src, uint32_t size, uint32_t tid) {
        for (int j = 0; j < size / 64; j++) 
            utils_t::_ldgsts_128b_async(dst + 8 * 132 * j, src + 8 * 128 * j);
        if (tid < ((size * 4) & 0xff)) 
            utils_t::_ldgsts_128b_async(dst + 8 * 132 * (size / 64), src + 8 * 128 * (size / 64));                                  
        utils_t::_commit_async_group();
    }
    static __device__ __forceinline__ void _prep_coeff(int *dst, int8_t *src, int8_t *b_dual, 
                                                       int dhalf, int dshift, int rem, uint64_t u_acc[8], 
                                                       uint64_t *uid_coeff, uint32_t CSD16, uint32_t lid) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> c;

        wmma::fill_fragment(c, dhalf);

        #pragma unroll
        for (int i = rem - 16; i < CSD16; i += 16) {
            wmma::load_matrix_sync(a, b_dual + (i - rem + 16) * 16, 16);
            wmma::load_matrix_sync(b, src + i, vec_nbytes);
            wmma::mma_sync(c, a, b, c);
        }

        for (int j = 0; j < c.num_elements; j++) c.x[j] >>= dshift;
        {   
            uint64_t c0 = uid_coeff[rem - 16 + lid / 4 + 0];
            uint64_t c1 = uid_coeff[rem - 16 + lid / 4 + 8];
            u_acc[0] += c.x[0] * c0 + c.x[2] * c1;
            u_acc[1] += c.x[1] * c0 + c.x[3] * c1;
            u_acc[2] += c.x[4] * c0 + c.x[6] * c1;
            u_acc[3] += c.x[5] * c0 + c.x[7] * c1;
        }
        for (int j = 0; j < c.num_elements; j++) *((float *)&c.x[j]) = __int2float_rn(c.x[j]);
        for (int j = 0; j < c.num_elements; j++) dst[2 * lid + (j & 0x1) + (j & 0x6) * 32 + (j & 0x4)] = c.x[j];
    }
    static __device__ __forceinline__ void _next_coeff(int *dst, int8_t *src, int8_t * b_dual, 
                                                       int dhalf, int dshift, uint64_t u_acc[8], 
                                                       uint64_t *uid_coeff, uint32_t CSD16, uint32_t lid) {
        utils_t::_wait_async_group_exp1();
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> c;
        wmma::fill_fragment(c, dhalf);

        __syncthreads();
        
        wmma::load_matrix_sync(a, b_dual, 16);
        wmma::load_matrix_sync(b, src + CSD16 - 16, vec_nbytes);
        wmma::mma_sync(c, a, b, c);

        for (int j = 0; j < c.num_elements; j++) c.x[j] >>= dshift;
        {   
            uint64_t c0 = uid_coeff[CSD16 - 16 + lid / 4 + 0];
            uint64_t c1 = uid_coeff[CSD16 - 16 + lid / 4 + 8];
            u_acc[0] = c.x[0] * c0 + c.x[2] * c1;
            u_acc[1] = c.x[1] * c0 + c.x[3] * c1;
            u_acc[2] = c.x[4] * c0 + c.x[6] * c1;
            u_acc[3] = c.x[5] * c0 + c.x[7] * c1;
        }
        for (int j = 0; j < c.num_elements; j++) *((float *)&c.x[j]) = __int2float_rn(c.x[j]);
        for (int j = 0; j < c.num_elements; j++) dst[2 * lid + (j & 0x1) + (j & 0x6) * 32 + (j & 0x4)] = c.x[j];

        utils_t::_wait_async_group();
        __syncthreads();
    }
    static __device__ __forceinline__ void _uid_reduce(uint64_t u_acc[8], uint64_t ubuf[128], 
                                                       uint32_t wid, uint32_t lid) {
        for (int w = 16; w >= 4; w >>= 1) {
            u_acc[0] += __shfl_down_sync(0xffffffff, u_acc[0], w);
            u_acc[1] += __shfl_down_sync(0xffffffff, u_acc[1], w);
            u_acc[2] += __shfl_down_sync(0xffffffff, u_acc[2], w);
            u_acc[3] += __shfl_down_sync(0xffffffff, u_acc[3], w);
        }
        if (lid < 4) {
            ubuf[wid * 16 + lid * 2 + 0] = u_acc[0];
            ubuf[wid * 16 + lid * 2 + 1] = u_acc[1];
            ubuf[wid * 16 + lid * 2 + 8] = u_acc[2];
            ubuf[wid * 16 + lid * 2 + 9] = u_acc[3];
        }
    }
    static __device__ __forceinline__ void _vec_stage0(int2 vp_line[16], float n_acc[8], uint32_t &smsk_acc, float *frag0, 
                                                       float *frag1, uint32_t ESD8, uint32_t CSD16, uint32_t wid) {
        for (int i = 0; i < 8; i++) n_acc[i] = 0.0f;
        if (wid >= (ESD8 + 8) / 16 && wid < (ESD8 + 8) / 16 + CSD16 / 16) {
            for (int j = 0; j < 8; j++) {
                for (int i = 0; i < 8; i++) {
                    n_acc[i] += frag0[i * 8 + j] * frag0[i * 8 + j];
                    if (fabsf(frag0[i * 8 + j]) > 127.4f) smsk_acc |= 1 << i;
                }
            }
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) ((int8_t *)&vp_line[i])[j] = 
                                            (int8_t) __float2int_rn(frag0[i * 8 + j]);
            }
        }
        if (wid + 8 < (ESD8 + 8) / 16 + CSD16 / 16) {
            for (int j = 0; j < 8; j++) {
                for (int i = 0; i < 8; i++) {
                    n_acc[i] += frag1[i * 8 + j] * frag1[i * 8 + j];
                    if (fabsf(frag1[i * 8 + j]) > 127.4f) smsk_acc |= 1 << i;
                }
            }
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) ((int8_t *)&vp_line[i + 8])[j] = 
                                            (int8_t) __float2int_rn(frag1[i * 8 + j]);
            }
        }
    }
    static __device__ __forceinline__ void _vec_stage1(float *shbuf, int2 vp_line[16], float n_acc[8], 
                                                       uint32_t &smsk_acc, uint32_t ESD8, uint32_t CSD16, 
                                                       uint32_t wid, uint32_t lid, uint32_t tid) {
        int8_t *v_tmp = (int8_t *)shbuf;
        float *n_tmp = (float *)(v_tmp + 128 * vec_nbytes + 128);
        uint32_t *s_tmp = (uint32_t *)(n_tmp + 1280);
        if (wid >= (ESD8 + 8) / 16 && wid < (ESD8 + 8) / 16 + CSD16 / 16) {
            const int v_bias = (vec_nbytes * 8 + 8) * (tid & 0x0f) + (tid & 0xf0) / 2 - ((ESD8 + 8) & 0xf0);
            for (int i = 0; i < 8; i++) *((int2 *)&v_tmp[v_bias + i * vec_nbytes]) = vp_line[i];
        }
        if (wid + 8 < (ESD8 + 8) / 16 + CSD16 / 16) {
            const int v_bias = (vec_nbytes * 8 + 8) * (tid & 0x0f) + (tid & 0xf0) / 2 + 128 - ((ESD8 + 8) & 0xf0);
            for (int i = 0; i < 8; i++) *((int2 *)&v_tmp[v_bias + i * vec_nbytes]) = vp_line[i+8];
        }
        for (int i = 0; i < 8; i++) {
            for (int j = CSD16; j < vec_nbytes; j += 8) 
                *((int2 *)&v_tmp[(vec_nbytes * 8 + 8) * (tid & 0x0f) + i * vec_nbytes + j]) = {0, 0};
        }

        smsk_acc |= __shfl_xor_sync(0xffffffff, smsk_acc, 16);
        if (lid < 16) s_tmp[wid * 16 + lid] = smsk_acc;

        for (int i = 0; i < 8; i++) n_acc[i] += __shfl_xor_sync(0xffffffff, n_acc[i], 16);
        for (int i = 0; i < 8; i++) {
            if (lid < 16) n_tmp[(lid / 2) * (128 + 2) + (lid & 1) * 16 + (wid & 1) + 2 * i + (wid / 2) * 32] = n_acc[i];
        }
        
        __syncthreads();

        for (int i = 0; i < 11; i++) ((int *)vp_line)[i] = ((int *)v_tmp)[32 * i + lid + wid * (2 * vec_nbytes + 2)];
        for (int i = 0; i < 11; i++) ((int *)vp_line)[i+11] = ((int *)v_tmp)[32 * i + lid + (wid + 8) * (2 * vec_nbytes + 2)];
    }
    static __device__ __forceinline__ void _vec_stage2(float *shbuf, int32_t nbuf[128], uint16_t sbuf[128], 
                                                       int &sn, float &s0, float &s1, uint32_t ESD8,
                                                       uint32_t wid, uint32_t lid, uint32_t tid) {
        float *n_tmp = shbuf + 32 * vec_nbytes + 32;
        uint32_t *s_tmp = (uint32_t *)(n_tmp + 1280);

        if (wid == 7) {
            uint32_t m = s_tmp[lid];
            m |= s_tmp[lid + 32];
            m |= s_tmp[lid + 64];
            m |= s_tmp[lid + 96];
            m |= __shfl_down_sync(0xffffffff, m, 16);
            if (lid < 16 && m) {
                for (int r = 0; r < 8; r++) {
                    if (m & (1 << r)) sbuf[lid * 8 + r] = 0;
                }
            }
        }

        {
            float n_fp = 0.0f;
            for (int i = 0; i < 128; i += 32) n_fp += n_tmp[wid * (128 + 2) + i + lid];
            n_fp += __shfl_xor_sync(0xffffffff, n_fp, 1);
            s0 = __shfl_sync(0xffffffff, n_fp, 0, 4);
            s1 = __shfl_sync(0xffffffff, n_fp, 2, 4);
            int n_epi32 = __float2int_rn(n_fp * 0.5f);
            if (lid & 1) nbuf[tid / 2] = n_epi32;
            if (!ESD8) sn = min(n_epi32 >> 1, 65535);
        }
    }
    static __device__ __forceinline__ void _ext_stage0(float *shbuf, float *ext_frag, float *fp32_frag0, 
                                                       uint32_t ESD8, uint32_t wid, uint32_t tid) {
        if (wid < ESD8 / 16) {
            const int e_bias = ((ESD8 + 4) * 8 + 2) * (tid & 0x0f) + (tid & 0xf0) / 2;
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) shbuf[e_bias + i * (ESD8 + 4) + j] = fp32_frag0[i * 8 + j];
            }
        }
        if (wid == ESD8 / 16 && ESD8 & 0xf) {
            const int e_bias = ((ESD8 + 4) * 8 + 2) * (tid & 0x0f) + ((ESD8 + 4) / 4) * (tid & 0x10) + (tid & 0xe0) / 2;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 8; j++) shbuf[e_bias + i * (ESD8 + 4) + j] = fp32_frag0[i * 8 + j];
            }
        }
        __syncthreads();
        float *src = &shbuf[(tid / 4) * (ESD8 + 4) * 2 + (tid & 0x3) * 2 + (tid / 16) * 2];
        for (int i = 0; i < ESD8; i += 8) {
            ext_frag[i / 2 + 0] = src[i + 0];
            ext_frag[i / 2 + 1] = src[i + 1];
            ext_frag[i / 2 + 2] = src[i + 0 + ESD8 + 4];
            ext_frag[i / 2 + 3] = src[i + 1 + ESD8 + 4];
        }
        __syncthreads();
    }
    static __device__ __forceinline__ void _ext_stage1(float *ext_frag, int &sn, float &s0, float &s1, 
                                                       float *b_ext_head, float *igh, float *inorm, 
                                                       uint32_t ESD8, uint32_t tid) {
        float n0 = s0, n1 = s1;
        float *b_ext_ptr = b_ext_head;
        for (int rem = ESD8 - 2; rem >= 0; rem -= 2) {
            const unsigned int w_tid = (rem / 2) & 0x3;
            const unsigned int w_bias = (rem / 8) * 4;

            float c0, c1, c2, c3;
            if ((tid & 0x3) == w_tid) {
                c1 = __float2int_rn(ext_frag[w_bias + 1] * inorm[rem + 1]);
                c3 = __float2int_rn(ext_frag[w_bias + 3] * inorm[rem + 1]);
                ext_frag[w_bias + 1] -= c1 * b_ext_ptr[rem + 1];
                ext_frag[w_bias + 3] -= c3 * b_ext_ptr[rem + 1];
                ext_frag[w_bias + 0] -= c1 * b_ext_ptr[rem + 0];
                ext_frag[w_bias + 2] -= c3 * b_ext_ptr[rem + 0];
                n0 += ext_frag[w_bias + 1] * ext_frag[w_bias + 1];
                n1 += ext_frag[w_bias + 3] * ext_frag[w_bias + 3];
                s0 = min(s0, n0 * igh[rem + 1]);
                s1 = min(s1, n1 * igh[rem + 1]);
                c0 = __float2int_rn(ext_frag[w_bias + 0] * inorm[rem]);
                c2 = __float2int_rn(ext_frag[w_bias + 2] * inorm[rem]);
                ext_frag[w_bias + 0] -= c0 * b_ext_ptr[2 * rem + 2];
                ext_frag[w_bias + 2] -= c2 * b_ext_ptr[2 * rem + 2];
                n0 += ext_frag[w_bias + 0] * ext_frag[w_bias + 0];
                n1 += ext_frag[w_bias + 2] * ext_frag[w_bias + 2];
                s0 = min(s0, n0 * igh[rem]);
                s1 = min(s1, n1 * igh[rem]);
            }

            n0 = __shfl_sync(0xffffffff, n0, w_tid, 4);
            n1 = __shfl_sync(0xffffffff, n1, w_tid, 4);
            c0 = __shfl_sync(0xffffffff, c0, w_tid, 4);
            c1 = __shfl_sync(0xffffffff, c1, w_tid, 4);
            c2 = __shfl_sync(0xffffffff, c2, w_tid, 4);
            c3 = __shfl_sync(0xffffffff, c3, w_tid, 4);

            #pragma unroll
            for (int i = 0; i < w_bias; i += 4) {
                const int bias = i * 2 + (tid & 0x3) * 2;
                ext_frag[i + 0] -= c0 * b_ext_ptr[rem + 2 + bias + 0] + 
                                c1 * b_ext_ptr[bias + 0];
                ext_frag[i + 1] -= c0 * b_ext_ptr[rem + 2 + bias + 1] + 
                                c1 * b_ext_ptr[bias + 1];
                ext_frag[i + 2] -= c2 * b_ext_ptr[rem + 2 + bias + 0] + 
                                c3 * b_ext_ptr[bias + 0];
                ext_frag[i + 3] -= c2 * b_ext_ptr[rem + 2 + bias + 1] + 
                                c3 * b_ext_ptr[bias + 1];
            }

            if ((tid & 0x3) < w_tid) {
                const int bias = w_bias * 2 + (tid & 0x3) * 2;
                ext_frag[w_bias + 0] -= c0 * b_ext_ptr[rem + 2 + bias + 0] + 
                                        c1 * b_ext_ptr[bias + 0];
                ext_frag[w_bias + 1] -= c0 * b_ext_ptr[rem + 2 + bias + 1] + 
                                        c1 * b_ext_ptr[bias + 1];
                ext_frag[w_bias + 2] -= c2 * b_ext_ptr[rem + 2 + bias + 0] + 
                                        c3 * b_ext_ptr[bias + 0];
                ext_frag[w_bias + 3] -= c2 * b_ext_ptr[rem + 2 + bias + 1] + 
                                        c3 * b_ext_ptr[bias + 1];
            }
            
            b_ext_ptr += 2 * rem + 4;
        }
        s0 = min(s0, __shfl_xor_sync(0xffffffff, s0, 1));
        s1 = min(s1, __shfl_xor_sync(0xffffffff, s1, 1));
        s0 = min(s0, __shfl_xor_sync(0xffffffff, s0, 2));
        s1 = min(s1, __shfl_xor_sync(0xffffffff, s1, 2));
        s0 *= 0.25f;
        s1 *= 0.25f;
        float s = (tid & 2) ? s1 : s0;
        sn = min(__float2int_rn(s), 65535);
    }
    static __device__ __forceinline__ void _write_back(int8_t *data, int ind, int vp_line[22], 
                                                       uint64_t ubuf[128], int32_t nbuf[128], uint16_t sbuf[128], int sn,
                                                       uint32_t wid, uint32_t lid, uint32_t tid) {
        int *glob_vec = (int *)&data[ind * vec_nbytes];
        for (int i = 0; i < 11; i++) glob_vec[32 * i + lid + wid * 2 * vec_nbytes] = vp_line[i];
        for (int i = 0; i < 11; i++) glob_vec[32 * i + lid + (wid + 8) * 2 * vec_nbytes] = vp_line[i+11];
        if (tid < 64) ((int4 *)&data[uBias + ind * sizeof(uint64_t)])[tid] = ((int4 *)ubuf)[tid];
        if (tid < 32) ((int4 *)&data[normBias + ind * sizeof(int32_t)])[tid] = ((int4 *)nbuf)[tid];
        if (tid & 1) ((uint16_t *)&data[scoreBias + ind * sizeof(uint16_t)])[tid/2] = sbuf[tid/2] & sn;
    }

    struct stream_with_l2_holder_t {
        stream_with_l2_holder_t(cudaStream_t &stream, void *base_ptr, int num_bytes) {
            CHECK_CUDA_ERR(cudaStreamCreate(&stream));
            _stream = stream;
            if (base_ptr) {
                stream_attribute.accessPolicyWindow.base_ptr = base_ptr;
                stream_attribute.accessPolicyWindow.num_bytes = num_bytes;
                stream_attribute.accessPolicyWindow.hitRatio = 1.0;
                stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
                stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
                CHECK_CUDA_ERR(cudaStreamSetAttribute(stream, 
                cudaStreamAttributeAccessPolicyWindow, &stream_attribute));
            } else stream_attribute.accessPolicyWindow.base_ptr = NULL;
        }
        ~stream_with_l2_holder_t() {
            if (stream_attribute.accessPolicyWindow.base_ptr) {
                stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
                CHECK_CUDA_ERR(cudaStreamSetAttribute(_stream, 
                        cudaStreamAttributeAccessPolicyWindow, &stream_attribute));
            }
            CHECK_CUDA_ERR(cudaStreamDestroy(_stream));
        }
        cudaStreamAttrValue stream_attribute;
        cudaStream_t _stream;
    };

    struct pool_hd_buffer_holder_t {
        public:
        pool_hd_buffer_holder_t(cudaStream_t &stream, int8_t *&d_buffer, int8_t *&pack_buffer, int8_t *&h_buffer, 
                                                    uint16_t *&h_buffer_score, int32_t *&h_buffer_norm, uint64_t *&h_buffer_u) {
            CHECK_CUDA_ERR(cudaMallocHost(&h_buffer, taskVecs * (vec_nbytes + 2 + 4 + 8)));
            CHECK_CUDA_ERR(cudaMalloc(&d_buffer, taskVecs * (vec_nbytes + 2 + 4 + 8)));
            CHECK_CUDA_ERR(cudaMalloc(&pack_buffer, taskVecs * vec_nbytes));
            CHECK_CUDA_ERR(cudaMemsetAsync(d_buffer, 0, taskVecs * vec_nbytes, stream));
            CHECK_CUDA_ERR(cudaMemsetAsync(pack_buffer, 0, taskVecs * vec_nbytes, stream));
            memset(h_buffer, 0, taskVecs * vec_nbytes);

            this->_d_buffer     = d_buffer;
            this->_h_buffer     = h_buffer;
            this->_pack_buffer  = pack_buffer;
            h_buffer_norm       = (int32_t *)(h_buffer + taskVecs * (vec_nbytes + 2));
            h_buffer_score      = (uint16_t *)(h_buffer + taskVecs * (vec_nbytes));
            h_buffer_u          = (uint64_t *)(h_buffer + taskVecs * (vec_nbytes + 2 + 4));
        }
        ~pool_hd_buffer_holder_t() {
            CHECK_CUDA_ERR(cudaFree(_d_buffer));
            CHECK_CUDA_ERR(cudaFree(_pack_buffer));
            CHECK_CUDA_ERR(cudaFreeHost(_h_buffer));
        }
        // input and output buffer (vec, score, norm, uid in order)
        int8_t *_d_buffer;
        int8_t *_h_buffer;
        int8_t *_pack_buffer;
    };

    static void init_shared_mem_limit() {
        CHECK_CUDA_ERR(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shmem));
        utils_t::init_shared_mem_limit();
    }

    static void prep_device_local_data(local_data_t *&local_data, Pool_hd_t *p) {
        _prep_device_local_data(vec_nbytes, max_boost_dim, local_data, p);
    }

    static void launch(cudaStream_t stream, int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data) {
        kernel<<<kernelBlocks, blockThreads, dynamic_shmem, stream>>>(data, n, local_data);
        CHECK_LAST_ERR;
    }
    
    static MAT_QP b_trans_QP(Pool_hd_t *p) {
        const int CSD = p->CSD;
        const int ESD = p->ESD;
        const int bias = p->index_l - p->ESD;

        MAT_QP ret = NEW_MAT_QP(ESD + CSD, p->basis->NumCols());
        MAT_QP b_QP = p->basis->get_b();
        MAT_QP mu_QP = p->basis->get_miu();
        VEC_QP B_QP = p->basis->get_B();
        
        double **b_normal = (double **) NEW_MAT(CSD + ESD, CSD + ESD, sizeof(double));
        double **b_target = (double **) NEW_MAT(CSD + ESD, CSD + ESD, sizeof(double));
        for (int j = 0; j < CSD + ESD; j++) {
            double x = sqrtl(B_QP.hi[j + bias]) * p->_ratio;
            for (int i = j; i < CSD + ESD; i++) b_normal[i][j] = x * mu_QP.hi[i + bias][j + bias];
        }

        for (int i = 0; i < CSD; i++) {
            for (int j = 0; j < ESD; j++) b_target[ESD + i][j] = p->_boost_data->evec[(ESD + i) * ESD + j];
            for (int j = 0; j < CSD; j++) b_target[ESD + i][ESD + j] = p->_b_local[i][j];
        }
        for (int i = 0; i < ESD; i++) {
            for (int j = 0; j < ESD; j++) b_target[i][j] = p->_boost_data->evec[i * ESD + j];
        }

        double max_fperr = 0.0;

        for (int i = 0; i < ESD + CSD; i++) {
            for (int j = i; j >= 0; j--) {
                double c = roundl(b_target[i][j] / b_normal[j][j]);
                max_fperr = fmax(fabs(b_target[i][j] / b_normal[j][j] - c), max_fperr); 
                for (int l = 0; l < j; l++) b_target[i][l] -= b_normal[j][l] * c;
                red(ret.hi[i], ret.lo[i], b_QP.hi[bias + j], b_QP.lo[bias + j], NTL::quad_float(-c), p->basis->NumCols());
            }
        }
        if (max_fperr > 0.1) {
            fprintf(stderr, "[Error] pdev_traits_t<%u>::b_trans_QP: "
                            "large floating point error(%.2f)\n", Ver, max_fperr);
        } else if (max_fperr > 0.01) {
            fprintf(stderr, "[Warning] pdev_traits_t<%u>::b_trans_QP: "
                            "floating point error(%.2f) warning\n", Ver, max_fperr);
        }

        FREE_MAT(b_normal);
        FREE_MAT(b_target);

        return ret;
    }

    static void _prep_device_local_data(int CSD16, int ESD8, local_data_t *&local_data, Pool_hd_t *p);
};

#if 1
#define pdev_kernel_var_declare()                                                                                   \
    static constexpr unsigned int vec_nbytes = traits::vec_nbytes;                                                  \
    extern __shared__ int8_t dy_buf[];                                                                              \
    int8_t *vec = dy_buf;                                                                                           \
    float *fbuf = (float *)(&dy_buf[22528]);                                                                        \
    float *cbuf = (float *)(&dy_buf[52096]);                                                                        \
    __shared__ uint64_t ubuf[128];                                                                                  \
    __shared__ uint16_t sbuf[128];                                                                                  \
    __shared__ int32_t nbuf[128];                                                                                   \
                                                                                                                    \
    __shared__ int8_t b_dual_tri[176 * (176 + 16) / 2];                                                             \
    __shared__ float b_ext_head[1200];                                                                              \
    __shared__ uint64_t uid_coeff[176];                                                                             \
    __shared__ float inorm[48];                                                                                     \
    __shared__ float igh[48];                                                                                       \
                                                                                                                    \
    uint64_t u_acc[4];                                                                                              \
                                                                                                                    \
    const unsigned int wid = utils_t::_thread_id() / 32;                                                            \
    const unsigned int lid = utils_t::_lane_id();                                                                   \
    const unsigned int tid = utils_t::_thread_id();                                                                 \
                                                                                                                    \
    const int dhalf = local_data->dhalf;                                                                            \
    const int dshift = local_data->dshift;                                                                          \
                                                                                                                    \
    int8_t *const warp_vec = vec + wid * 16 * vec_nbytes;                                                           \
    int *const warp_coeff = (int *)(cbuf + wid * 2 * 132);                                                          \
    float *const lane_bfull = fbuf + tid * 4 + wid * 4;                                                             \
    float *const lane_bfull_glob = local_data->b_full + tid * 4;                                                    \
                                                                                                                    \
    int ind = 128 * blockIdx.x;                                                                                     \
    const int stride = 128 * gridDim.x;                                                                             \
    int8_t *glob_vec = &data[ind * vec_nbytes]                                                                     
#endif

template <uint32_t CSD16, uint32_t ESD8>
__global__ void check_kernel(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data);

template <uint32_t CSD16, uint32_t ESD8>
__global__ void extend_left_kernel(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data);

template <uint32_t CSD16, uint32_t ESD8>
__global__ void min_lift_kernel(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data);

template <uint32_t CSD16, uint32_t ESD8>
__global__ void insert_kernel(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data);

template <uint32_t CSD16>
__global__ void dim_lose_kernel(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data);

template <uint32_t CSD16, uint32_t ESD8>
__global__ void filter_kernel(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data);


typedef pdev_traits_t<Ver_rc> check_traits;
typedef pdev_traits_t<Ver_el> extend_left_traits;
typedef pdev_traits_t<Ver_sl> shrink_left_traits;
typedef pdev_traits_t<Ver_tl> tail_LLL_traits;
typedef pdev_traits_t<Ver_ml> min_lift_traits;
typedef pdev_traits_t<Ver_it> insert_traits;
typedef pdev_traits_t<Ver_ds> dim_lose_traits;

#endif

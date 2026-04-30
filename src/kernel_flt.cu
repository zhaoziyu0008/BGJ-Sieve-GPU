#include "../include/config.h"
#include "../include/pool_hd_device.h"
#include "../include/bgj_hd_device.h"

__device__ __forceinline__ void _fmma8x16x8(float *C, float *A, float *B) {
        __attribute__((aligned(16))) float A_col[16];
        __attribute__((aligned(16))) float B_col[16];

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

__device__ __forceinline__ void _fmma4x16x8(float *C, float *A, float *B) {
    __attribute__((aligned(16))) float A_col[8];
    __attribute__((aligned(16))) float B_col[16];

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

__device__ __forceinline__ void _kernel_init(int8_t *b_dual_tri, float *b_ext_head, 
                                                    uint64_t *uid_coeff, float *inorm, float *igh, 
                                                    local_data_t *ld, uint32_t ESD8, uint32_t tid) {
    for (int i = tid; i < sizeof(local_data_t::b_dual) / sizeof(int4); i += 256) 
        ((int4 *)b_dual_tri)[i] = ((int4 *)ld->b_dual)[i];
    for (int i = tid; i < sizeof(local_data_t::b_ext_head) / sizeof(int4); i += 256) 
        ((int4 *)b_ext_head)[i] = ((int4 *)ld->b_ext_head)[i];
                                                                    
    if (tid < 176) uid_coeff[tid] = ld->uid_coeff[tid];
    if (ESD8 && tid < boost_data_t::max_boost_dim) {
        inorm[tid] = ld->inorm[tid];
        igh[tid] = ld->igh[tid];
    }
}

__device__ __forceinline__ void _batch_init(float *frag0, float *frag1, uint16_t sbuf[128], 
                                                    uint32_t size, uint32_t wid, uint32_t lid) {
    if (wid * 16 < size) {
        for (int i = 0; i < 64; i++) frag0[i] = 0.0f;
    }
    if (wid * 16 + 128 < size) {
        for (int i = 0; i < 64; i++) frag1[i] = 0.0f;
    }
    if (wid == 0) ((uint64_t *)sbuf)[lid] = 0xffffffffffffffffULL;
}

template <uint32_t CSD16>
__device__ __forceinline__ void _prep_vec(int8_t *dst, int8_t *src, uint32_t tid) {
    for (int i = 0; i < CSD16 / 16; i++) utils_t::_ldgsts_64b_async(dst + 8 * tid + 8 * 256 * i, 
                                                            src + 8 * tid + 8 * 256 * i);
    utils_t::_commit_async_group();
}

__device__ __forceinline__ void _prep_bfull(float *dst, float *src, uint32_t size, uint32_t tid) {
    for (int j = 0; j < size / 64; j++) 
        utils_t::_ldgsts_128b_async(dst + 8 * 132 * j, src + 8 * 128 * j);
    if (tid < ((size * 4) & 0xff)) 
        utils_t::_ldgsts_128b_async(dst + 8 * 132 * (size / 64), src + 8 * 128 * (size / 64));                                  
    utils_t::_commit_async_group();
}

__device__ __forceinline__ void _prep_coeff(int *dst, int8_t *src, int8_t *b_dual, 
                                                    int dhalf, int dshift, int rem, uint64_t u_acc[8], 
                                                    uint64_t *uid_coeff, uint32_t CSD16, uint32_t lid) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c;

    wmma::fill_fragment(c, dhalf);

    #pragma unroll
    for (int i = rem - 16; i < CSD16; i += 16) {
        wmma::load_matrix_sync(a, b_dual + (i - rem + 16) * 16, 16);
        wmma::load_matrix_sync(b, src + i, CSD16);
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

__device__ __forceinline__ void _next_coeff(int *dst, int8_t *src, int8_t * b_dual, 
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
    wmma::load_matrix_sync(b, src + CSD16 - 16, CSD16);
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

__device__ __forceinline__ void _uid_reduce(uint64_t u_acc[8], uint64_t ubuf[128], 
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

__device__ __forceinline__ void _vec_stage0(int2 vp_line[16], float n_acc[8], uint32_t &smsk_acc, float *frag0, 
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

__device__ __forceinline__ void _vec_stage1(float *shbuf, int2 vp_line[22], float n_acc[8], 
                                                    uint32_t &smsk_acc, uint32_t ESD8, uint32_t CSD16, 
                                                    uint32_t wid, uint32_t lid, uint32_t tid) {
    int8_t *v_tmp = (int8_t *)shbuf;
    float *n_tmp = (float *)(v_tmp + 128 * CSD16 + 128);
    uint32_t *s_tmp = (uint32_t *)(n_tmp + 1280);
    if (wid >= (ESD8 + 8) / 16 && wid < (ESD8 + 8) / 16 + CSD16 / 16) {
        const int v_bias = (CSD16 * 8 + 8) * (tid & 0x0f) + (tid & 0xf0) / 2 - ((ESD8 + 8) & 0xf0);
        for (int i = 0; i < 8; i++) *((int2 *)&v_tmp[v_bias + i * CSD16]) = vp_line[i];
    }
    if (wid + 8 < (ESD8 + 8) / 16 + CSD16 / 16) {
        const int v_bias = (CSD16 * 8 + 8) * (tid & 0x0f) + (tid & 0xf0) / 2 + 128 - ((ESD8 + 8) & 0xf0);
        for (int i = 0; i < 8; i++) *((int2 *)&v_tmp[v_bias + i * CSD16]) = vp_line[i+8];
    }
    for (int i = 0; i < 8; i++) {
        for (int j = CSD16; j < CSD16; j += 8) 
            *((int2 *)&v_tmp[(CSD16 * 8 + 8) * (tid & 0x0f) + i * CSD16 + j]) = {0, 0};
    }

    smsk_acc |= __shfl_xor_sync(0xffffffff, smsk_acc, 16);
    if (lid < 16) s_tmp[wid * 16 + lid] = smsk_acc;

    for (int i = 0; i < 8; i++) n_acc[i] += __shfl_xor_sync(0xffffffff, n_acc[i], 16);
    for (int i = 0; i < 8; i++) {
        if (lid < 16) n_tmp[(lid / 2) * (128 + 2) + (lid & 1) * 16 + (wid & 1) + 2 * i + (wid / 2) * 32] = n_acc[i];
    }
    
    __syncthreads();

    for (int i = 0; i < CSD16 / 16; i++) ((int *)vp_line)[i] = ((int *)v_tmp)[32 * i + lid + wid * (2 * CSD16 + 2)];
    for (int i = 0; i < CSD16 / 16; i++) ((int *)vp_line)[i+CSD16 / 16] = ((int *)v_tmp)[32 * i + lid + (wid + 8) * (2 * CSD16 + 2)];
}

template<uint32_t CSD16>
__device__ __forceinline__ void _vec_stage2(float *shbuf, int32_t nbuf[128], uint16_t sbuf[128], 
                                                    int &sn, float &s0, float &s1, uint32_t ESD8,
                                                    uint32_t wid, uint32_t lid, uint32_t tid) {
    float *n_tmp = shbuf + 32 * CSD16 + 32;
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

__device__ __forceinline__ void _ext_stage0(float *shbuf, float *ext_frag, float *fp32_frag0, 
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

__device__ __forceinline__ void _ext_stage1(float *ext_frag, int &sn, float &s0, float &s1, 
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

template <uint32_t CSD16>
__device__ __forceinline__ void _filter_write_back(int8_t *data, int ind, int vp_line[22], 
                                            uint64_t ubuf[128], int32_t nbuf[128], uint16_t sbuf[128], int sn,
                                            uint32_t wid, uint32_t lid, uint32_t tid) {
    constexpr uint32_t vec_nbytes = Pool_hd_t::vec_nbytes;
    constexpr uint32_t uBias = filter_taskVecs * (vec_nbytes + 2 + 4);
    constexpr uint32_t normBias = filter_taskVecs * (vec_nbytes + 2);
    constexpr uint32_t scoreBias = filter_taskVecs * vec_nbytes;
    int *glob_vec = (int *)&data[ind * CSD16];
    for (int i = 0; i < CSD16 / 16; i++) glob_vec[32 * i + lid + wid * 2 * CSD16] = vp_line[i];
    for (int i = 0; i < CSD16 / 16; i++) glob_vec[32 * i + lid + (wid + 8) * 2 * CSD16] = vp_line[i+CSD16 / 16];
    if (tid < 64) ((int4 *)&data[uBias + ind * sizeof(uint64_t)])[tid] = ((int4 *)ubuf)[tid];
    if (tid < 32) ((int4 *)&data[normBias + ind * sizeof(int32_t)])[tid] = ((int4 *)nbuf)[tid];
    if (tid & 1) ((uint16_t *)&data[scoreBias + ind * sizeof(uint16_t)])[tid/2] = sbuf[tid/2] & sn;
}


/// almost the same as check_kernel
template <uint32_t CSD16, uint32_t ESD8>
__global__ void filter_kernel(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data) {                                                
    extern __shared__ int8_t dy_buf[];                      
    int8_t *vec = dy_buf;                                   
    float *fbuf = (float *)(&dy_buf[22528]);                
    float *cbuf = (float *)(&dy_buf[52096]);                
    __shared__ uint64_t ubuf[128];                                                                   
    __shared__ uint16_t sbuf[128];                          
    __shared__ int32_t nbuf[128];                           
                                                                                                                    
    __shared__ int8_t b_dual_tri[176 * (176 + 16) / 2];     
    __shared__ float b_ext_head[1200];                      
    __shared__ uint64_t uid_coeff[176];                     
    __shared__ float inorm[48];                             
    __shared__ float igh[48];                               
                                                                                                                    
    uint64_t u_acc[4];                                      
                                                                                                                    
    const unsigned int wid = utils_t::_thread_id() / 32;                                                            
    const unsigned int lid = utils_t::_lane_id();                                                                   
    const unsigned int tid = utils_t::_thread_id();                                                                 
                                                                                                                    
    const int dhalf = local_data->dhalf;                                                                            
    const int dshift = local_data->dshift;                                                                          
                                                                                                                    
    int8_t *const warp_vec = vec + wid * 16 * CSD16;           
    int *const warp_coeff = (int *)(cbuf + wid * 2 * 132);          
    float *const lane_bfull = fbuf + tid * 4 + wid * 4;             
    float *const lane_bfull_glob = local_data->b_full + tid * 4;    
                                                                                                                    
    int ind = 128 * blockIdx.x;                                                                                     
    const int stride = 128 * gridDim.x;                                                                             
    int8_t *glob_vec = &data[ind * CSD16];                                                                    

    /// prepare data ///
    _kernel_init(b_dual_tri, b_ext_head, uid_coeff, inorm, igh, local_data, ESD8, tid);
    
    /// for the first ind ///
    _prep_vec<CSD16>(vec, glob_vec, tid);
    _prep_bfull(lane_bfull, lane_bfull_glob, ESD8 + CSD16, tid);
    _next_coeff(warp_coeff, warp_vec, b_dual_tri, dhalf, dshift, u_acc, uid_coeff, CSD16, lid);

    while (ind < n) {
        float fp32_frag0[64];   // 64R
        float fp32_frag1[64];   // 64R

        _batch_init(fp32_frag0, fp32_frag1, sbuf, ESD8 + CSD16, wid, lid);
        
        #pragma unroll
        for (int rem = CSD16 - 16; rem >= 0; rem -= 16) {
            const int loop_size = (CSD16 - rem) >> 4;
            const int bd_bias = 256 / 2 * loop_size * (loop_size + 1);
            const int bf_bias = (CSD16 + ESD8 + 8 - 8 * loop_size) * loop_size * 16;
            const int c_bias = (loop_size & 0x1) * 16 * 132;
            const int f_bias = (loop_size & 0x1) * 28 * 132;

            if (rem != CSD16 - 16) {
                utils_t::_wait_async_group();
                __syncthreads();
            }

            if (rem) {
                _prep_bfull(lane_bfull + f_bias, lane_bfull_glob + bf_bias, ESD8 + rem, tid);
                _prep_coeff(warp_coeff + c_bias, warp_vec, b_dual_tri + bd_bias, 
                                          dhalf, dshift, rem, u_acc, uid_coeff, CSD16, lid);
            } else if (ind + stride < n) _prep_vec<CSD16>(vec, glob_vec + stride * CSD16, tid);

            if (wid < ESD8 / 16) {
                float *A_8x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + (tid >> 0x4) * 132;
                _fmma8x16x8(fp32_frag0, A_8x16, B_8x16);
            }
            if (wid == ESD8 / 16 && (ESD8 & 0xf)) {
                float *A_4x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132 + (tid & 0x10) / 4;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + wid * 2 * 132;
                _fmma4x16x8(fp32_frag0, A_4x16, B_8x16);
            }
            if (wid >= (ESD8 + 8) / 16 && wid <= (ESD8 + 8) / 16 + rem / 16) {
                float *A_8x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + (tid >> 0x4) * 132 - ((ESD8 & 0x8) / 8) * 132;
                _fmma8x16x8(fp32_frag0, A_8x16, B_8x16);
            }
            if (wid + 8 <= (ESD8 + 8) / 16 + rem / 16) {
                float *A_8x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + (tid >> 0x4) * 132 + 16 * 132 - ((ESD8 & 0x8) / 8) * 132;
                _fmma8x16x8(fp32_frag1, A_8x16, B_8x16);
            }
        }

        __syncthreads();
        if (ind + stride < n) _prep_bfull(lane_bfull, lane_bfull_glob, ESD8 + CSD16, tid);
        
        float n_acc[8];
        int2 vp_line[22];
        uint32_t smsk_acc = 0x00;
        _vec_stage0(vp_line, n_acc, smsk_acc, fp32_frag0, fp32_frag1, ESD8, CSD16, wid);

        _uid_reduce(u_acc, ubuf, wid, lid);

        float ext_frag[ESD8 / 2];
        if (ESD8) _ext_stage0(fbuf + 28 * 132, ext_frag, fp32_frag0, ESD8, wid, tid);

        int sn;
        float s0, s1;
        _vec_stage1(fbuf + 28 * 132, vp_line, n_acc, smsk_acc, ESD8, CSD16, wid, lid, tid);
        _vec_stage2<CSD16>(fbuf + 28 * 132, nbuf, sbuf, sn, s0, s1, ESD8, wid, lid, tid);
        
        if (ESD8) _ext_stage1(ext_frag, sn, s0, s1, b_ext_head, igh, inorm, ESD8, tid);

        if (ind + stride < n) _next_coeff(warp_coeff, warp_vec, b_dual_tri, dhalf,  
                                                        dshift, u_acc, uid_coeff, CSD16, lid);

        _filter_write_back<CSD16>(data, ind, (int *)vp_line, ubuf, nbuf, sbuf, sn, wid, lid, tid);

        ind += stride;
        glob_vec += stride * CSD16;
    }
}

template <uint32_t CSD16>
__global__ void filter_prepare_vec(int8_t *__restrict__ data, const int8_t *__restrict__ vec_pad16, 
                                   int *__restrict__ pairs, int n) {
    constexpr int batchVecs = 16;
    constexpr int blockWarps = 8;

    extern __shared__ int8_t dy_buf[];

    if (n % batchVecs) {
        if (threadIdx.x < batchVecs - (n % batchVecs)) {
            pairs[2 * n + 2 * threadIdx.x + 0] = 0;
            pairs[2 * n + 2 * threadIdx.x + 1] = 0;
        }
    }
    __syncthreads();

    const unsigned int lid = utils_t::_lane_id();
    const unsigned int wid = utils_t::_thread_id() / 32;

    int8_t *wsrc = dy_buf + (3 * wid + 0) * batchVecs * CSD16;
    int8_t *wdst = dy_buf + (3 * wid + 2) * batchVecs * CSD16;

    int *wpair = (int *)(dy_buf + 3 * blockWarps * batchVecs * CSD16) + wid * 2 * batchVecs;

    unsigned int ind = (blockIdx.x * blockWarps + wid) * batchVecs;
    const unsigned int stride = gridDim.x * blockWarps * batchVecs;
    
    while (ind < n) {
        wpair[lid] = pairs[2 * ind + lid];
        __syncwarp();
        int rpairs[16];
        for (int i = 0; i < 16; i++) rpairs[i] = wpair[i + (lid & 16)];
        for (int i = 0; i < 16; i++) {
            int4 *src = (int4 *)(vec_pad16 + rpairs[i] * (uint64_t) CSD16);
            int4 *dst = (int4 *)(wsrc + (i / 2) * CSD16 + (lid & 16) / 2 * CSD16 + (i & 1) * CSD16 * batchVecs);
            if ((lid & 15) < CSD16 / 16) utils_t::_ldgsts_128b_async(dst + (lid & 15), src + (lid & 15));
        }
        utils_t::_commit_async_group();
        utils_t::_wait_async_group();
        __syncwarp();

        int8_t *a = wsrc + lid * (CSD16 / 2);
        int8_t *b = wsrc + lid * (CSD16 / 2) + CSD16 * batchVecs;
        int8_t *c = wdst + lid * (CSD16 / 2);
        int8_t ra[CSD16 / 2];
        int8_t rb[CSD16 / 2];
        for (int i = 0; i < CSD16 / 2; i++) { ra[i] = a[i]; rb[i] = b[i]; }
        for (int i = 0; i < CSD16 / 2; i++) ra[i] = ra[i] - rb[i];
        for (int i = 0; i < CSD16 / 2; i++) c[i] = ra[i];

        __syncwarp();
        int2 *dst = (int2 *)(data + ind * CSD16);
        int2 *src = (int2 *)(wdst);
        for (int i = 0; i < CSD16 / 16; i++) {
            dst[i * 32 + lid] = src[i * 32 + lid];
        }

        ind += stride;
    }
}

template <uint32_t CSD16>
__global__ void filter_collect_sol(int8_t *vec_out, uint16_t *score_out, int32_t *norm_out, uint64_t *u_out, 
                                   int *num_out, int out_max_size, int8_t *__restrict__ data, int n, int goal_score) {
    constexpr int batchVecs = 64;
    constexpr int blockWarps = 8;
    
    uint16_t *glob_score = (uint16_t *)(data + filter_taskVecs * Pool_hd_t::vec_nbytes);
    int32_t  *glob_norm  = (int32_t *) (data + filter_taskVecs * (Pool_hd_t::vec_nbytes + 2));
    uint64_t *glob_u     = (uint64_t *)(data + filter_taskVecs * (Pool_hd_t::vec_nbytes + 2 + 4));

    __shared__ uint16_t sh_score[blockWarps * batchVecs];
    __shared__ int acc_num[blockWarps];
    __shared__ int acc[batchVecs * 2 * blockWarps];
    __shared__ uint16_t acc_score[batchVecs * 2 * blockWarps];

    if (threadIdx.x < blockWarps) acc_num[threadIdx.x] = 0;

    if (n % batchVecs) {
        if (threadIdx.x < batchVecs - (n % batchVecs)) {
            glob_score[n + threadIdx.x] = 0xffff;
        }
    }
    __syncthreads();

    const int lid = utils_t::_lane_id();
    const int wid = utils_t::_thread_id() / 32;

    uint16_t *wscore = &sh_score[wid * batchVecs];
    int *wacc_num = &acc_num[wid];
    int *wacc = &acc[wid * batchVecs * 2];
    uint16_t *wacc_score = &acc_score[wid * batchVecs * 2];
    
    unsigned int ind = (blockIdx.x * blockWarps + wid) * batchVecs;
    const unsigned int stride = gridDim.x * blockWarps * batchVecs;

    while (ind < n) {
        ((int *)wscore)[lid] = ((int *)glob_score)[ind / 2 + lid];
        __syncwarp();
        if (wscore[lid] && wscore[lid] < goal_score) {
            int pos = atomicAdd(wacc_num, 1);
            wacc[pos] = ind + lid;
            wacc_score[pos] = wscore[lid];
        }
        if (wscore[lid + 32] && wscore[lid + 32] < goal_score) {
            int pos = atomicAdd(wacc_num, 1);
            wacc[pos] = ind + lid + 32;
            wacc_score[pos] = wscore[lid + 32];
        }

        __syncwarp();

        int num = *wacc_num;
        if (num >= batchVecs) {
            int rnum = batchVecs;
            int pos;
            if (lid == 0) pos = atomicAdd(num_out, batchVecs);
            pos = __shfl_sync(0xffffffff, pos, 0);
            if (pos + rnum > out_max_size) {
                rnum = out_max_size - pos;
                if (rnum < 0) rnum = 0;
            }

            if (rnum == batchVecs) {
                for (int i = 0; i < batchVecs; i++) {
                    int idx = wacc[i];
                    int4 *src = (int4 *)(data + idx * CSD16);
                    int4 *dst = (int4 *)(vec_out + (pos + i) * CSD16);
                    if (lid < CSD16 / 16) dst[lid] = src[lid];
                }
            } else {
                for (int i = 0; i < rnum; i++) {
                    int idx = wacc[i];
                    int4 *src = (int4 *)(data + idx * CSD16);
                    int4 *dst = (int4 *)(vec_out + (pos + i) * CSD16);
                    if (lid < CSD16 / 16) dst[lid] = src[lid];
                }
            }

            int lidx0 = wacc[lid];
            int lidx1 = wacc[lid + 32];
            if (lid < rnum) score_out[pos + lid] = wacc_score[lid];
            if (lid + 32 < rnum) score_out[pos + lid + 32] = wacc_score[lid + 32];
            if (lid < rnum) u_out[pos + lid] = glob_u[lidx0];
            if (lid + 32 < rnum) u_out[pos + lid + 32] = glob_u[lidx1];
            if (lid < rnum) norm_out[pos + lid] = glob_norm[lidx0];
            if (lid + 32 < rnum) norm_out[pos + lid + 32] = glob_norm[lidx1];

            __syncwarp();

            if (rnum == batchVecs) {
                wacc[lid] = wacc[lid + batchVecs];
                wacc[lid + 32] = wacc[lid + 32 + batchVecs];
                ((int *)wacc_score)[lid] = ((int *)wacc_score)[lid + batchVecs / 2];
            } else {
                uint16_t s0, s1;
                int32_t e0, e1;
                if (rnum + lid < num) s0 = wacc_score[rnum + lid];
                if (rnum + lid + 32 < num) s1 = wacc_score[rnum + lid + 32];
                if (rnum + lid < num) e0 = wacc[rnum + lid];
                if (rnum + lid + 32 < num) e1 = wacc[rnum + lid + 32];
                __syncwarp();
                if (rnum + lid < num) wacc[lid] = e0;
                if (rnum + lid + 32 < num) wacc[lid + 32] = e1;
                if (rnum + lid < num) wacc_score[lid] = s0;
                if (rnum + lid + 32 < num) wacc_score[lid + 32] = s1;
            }
            if (lid == 0) wacc_num[0] = max(0, num - rnum);
        }
    
        ind += stride;
    }

    __syncwarp();
    int num = *wacc_num;
    int pos;
    if (lid == 0) pos = atomicAdd(num_out, num);
    pos = __shfl_sync(0xffffffff, pos, 0);
    if (pos + num > out_max_size) {
        num = out_max_size - pos;
        if (num < 0) num = 0;
    }

    for (int i = 0; i < num; i++) {
        int idx = wacc[i];
        int4 *src = (int4 *)(data + idx * CSD16);
        int4 *dst = (int4 *)(vec_out + (pos + i) * CSD16);
        if (lid < CSD16 / 16) dst[lid] = src[lid];
    }

    int lidx0 = wacc[lid];
    int lidx1 = wacc[lid + 32];
    if (lid < num) score_out[pos + lid] = wacc_score[lid];
    if (lid + 32 < num) score_out[pos + lid + 32] = wacc_score[lid + 32];
    if (lid < num) u_out[pos + lid] = glob_u[lidx0];
    if (lid + 32 < num) u_out[pos + lid + 32] = glob_u[lidx1];
    if (lid < num) norm_out[pos + lid] = glob_norm[lidx0];
    if (lid + 32 < num) norm_out[pos + lid + 32] = glob_norm[lidx1];
}


template __global__ void filter_kernel<176, 48>(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data);
template __global__ void filter_kernel<176, 24>(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data);
template __global__ void filter_prepare_vec<176>(int8_t *__restrict__ data, const int8_t *__restrict__ vec_pad16, 
                                                 int *__restrict__ pairs, int n);
template __global__ void filter_collect_sol<176>(int8_t *vec_out, uint16_t *score_out, int32_t *norm_out, uint64_t *u_out, 
                                                 int *num_out, int out_max_size, int8_t *__restrict__ data, int n, int goal_score);
#if RED_MIN_CSD16 < 176
template __global__ void filter_kernel<160, 48>(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data);
template __global__ void filter_kernel<160, 24>(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data);
template __global__ void filter_prepare_vec<160>(int8_t *__restrict__ data, const int8_t *__restrict__ vec_pad16, 
                                                 int *__restrict__ pairs, int n);
template __global__ void filter_collect_sol<160>(int8_t *vec_out, uint16_t *score_out, int32_t *norm_out, uint64_t *u_out,
                                                int *num_out, int out_max_size, int8_t *__restrict__ data, int n, int goal_score);
#endif
#if RED_MIN_CSD16 < 160
template __global__ void filter_kernel<144, 48>(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data);
template __global__ void filter_kernel<144, 24>(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data);
template __global__ void filter_prepare_vec<144>(int8_t *__restrict__ data, const int8_t *__restrict__ vec_pad16, 
                                                 int *__restrict__ pairs, int n);
template __global__ void filter_collect_sol<144>(int8_t *vec_out, uint16_t *score_out, int32_t *norm_out, uint64_t *u_out,
                                                int *num_out, int out_max_size, int8_t *__restrict__ data, int n, int goal_score);
#endif
#if RED_MIN_CSD16 < 144
template __global__ void filter_kernel<128, 48>(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data);
template __global__ void filter_kernel<128, 24>(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data);
template __global__ void filter_prepare_vec<128>(int8_t *__restrict__ data, const int8_t *__restrict__ vec_pad16, 
                                                 int *__restrict__ pairs, int n);
template __global__ void filter_collect_sol<128>(int8_t *vec_out, uint16_t *score_out, int32_t *norm_out, uint64_t *u_out,
                                                int *num_out, int out_max_size, int8_t *__restrict__ data, int n, int goal_score);
#endif
#include "../include/config.h"
#include "../include/common_device.h"
#include "../include/dh_device.h"

#include <mma.h>

__constant__ float dh_head[1176];
__constant__ float dh_inorm[48];

void set_dh_head(float *dh_head_val, cudaStream_t stream) {
    CHECK_CUDA_ERR(cudaMemcpyToSymbolAsync(dh_head, dh_head_val, 1176 * sizeof(float), 0, cudaMemcpyDefault, stream));       
}
void set_dh_inorm(float *dh_inorm_val, cudaStream_t stream) {
    CHECK_CUDA_ERR(cudaMemcpyToSymbolAsync(dh_inorm, dh_inorm_val, 48 * sizeof(float), 0, cudaMemcpyDefault, stream));
}

__device__ __forceinline__ static void _pfch_32vec(int8_t *dst, int8_t *src, int CSD, int CSD16, int wid, int lid) {
    for (int i = lid; i < CSD; i += 32) {
        utils_t::_ldgsts_32b_async(dst + wid * 4 * CSD + i * 4, src + wid * 4 * CSD + i * 4);
    }
    utils_t::_commit_async_group();
}

__device__ __forceinline__ static void _unpk_32vec(int8_t *dst, int8_t *src, int CSD, int CSD16, int wid, int lid) {
    utils_t::_wait_async_group();
    __syncwarp();
    int *wdst = (int *)(dst + wid * 4 * CSD16);
    int *wsrc = (int *)(src + wid * 4 * CSD);
    for (int i = lid; i < CSD16; i += 32) {
        int vid = i / (CSD16 / 4);
        int pos = i % (CSD16 / 4);
        int msk = 0xffffffff >> max(0, 32 * pos - 8 * CSD + 32);
        int val0 = wsrc[(CSD * vid + 4 * pos) / 4 + 0];
        int val1 = wsrc[(CSD * vid + 4 * pos) / 4 + 1];
        wdst[i] = __funnelshift_r(val0, val1, 8 * (CSD * vid)) & msk;
    }
}


template <uint32_t CSD16, uint32_t ESD8>
__global__ void dh_buc_kernel(uint32_t *__restrict__ out, int out_max_size, float *__restrict__ center, float radius, 
                              int batch, int8_t *__restrict__ in, int n, int CSD, dh_data_t *__restrict__ data) {
    /// constants
    constexpr int batchVecs  = 128;
    constexpr int unit       = 256;
    constexpr int sbucSize   = 19;

    extern __shared__ int8_t dy_buf[];
    int8_t     *vec = (int8_t *) (&dy_buf[0]);                       /*   22 KB */
    float     *head = (float *)  (&dy_buf[0]);                       /*   24 KB */
    int     *signal = (int *)    (&dy_buf[22 * 1024]);               /*    2 KB */
    int8_t     *vpk = (int8_t *) (&dy_buf[24 * 1024]);               /*  5.5 KB */
    int      *coeff = (int *)    (&dy_buf[22 * 1024]);               /*    8 KB */
    int       *sbuc = (int *)    (&dy_buf[30 * 1024]);               /*   19 KB */
    int8_t  *b_dual = (int8_t *) (&dy_buf[99 * 512]);                /* 16.5 KB */
    float   *b_head = (float *)  (&dy_buf[66 * 1024]);               /*   33 KB */

    const unsigned int wid = utils_t::_thread_id() / 32;
    const unsigned int lid = utils_t::_lane_id();
    const unsigned int tid = utils_t::_thread_id();

    const int dhalf = data->dhalf;
    const int dshift = data->dshift;

    const int blockTypes = batch / unit;
    const int blockType  = blockIdx.x % blockTypes;
    const int stride = ((gridDim.x + blockTypes - 1 - blockType) / blockTypes) * batchVecs;
    int ind = (blockIdx.x / blockTypes) * batchVecs;
    if (ind >= n) return;

    float t_center[ESD8];
    uint32_t  t_sbuc_num = 0;
    uint32_t *t_gbuc_num = out + blockType * unit + tid;
    uint32_t *w_out = out + batch + (blockType * unit + wid * 32) * out_max_size;

    /// once
    if (tid < 32) ((int4 *)signal)[tid] = {0, 0, 0, 0};
    for (int i = tid; i < sizeof(dh_data_t::b_dual) / sizeof(int4); i += 256) 
        ((int4 *)b_dual)[i] = ((int4 *)data->b_dual)[i];
    for (int i = tid; i < ESD8 * CSD16 / 4; i += 256) 
        ((int4 *)b_head)[i] = ((int4 *)data->b_head)[i];
    for (int i = tid; i < ESD8 * unit / 4; i += 256)
        ((int4 *)dy_buf)[i] = ((int4 *)(center + unit * blockType * ESD8))[i];
    __syncthreads();
    for (int i = 0; i < ESD8; i++) {
        t_center[i] = head[tid * ESD8 + i]; /* ignore bank confict for simplicity */
    }
    __syncthreads();


    /// for the first iter
    for (int l = 0; l < 4; l++) {
        _pfch_32vec(vpk, in + (ind + l * 32) * (uint64_t) CSD, CSD, CSD16, wid, lid);
        _unpk_32vec(vec + l * 32 * CSD16, vpk, CSD, CSD16, wid, lid);
        __syncwarp();
    }
    
    while (ind < n) {
        float vh_frag[4][ESD8 / 8] = {};

        __syncthreads();

        #pragma unroll
        for (int l = 0; l < CSD16; l += 16) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, int> c;
            wmma::fill_fragment(c, dhalf);

            for (int i = l; i < CSD16; i += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b;

                wmma::load_matrix_sync(a, vec + wid * 16 * CSD16 + i, CSD16);
                wmma::load_matrix_sync(b, b_dual + (l * (CSD16 + 8 - l / 2) + (i - l) * 16), 16);

                wmma::mma_sync(c, a, b, c);
            }

            for (int j = 0; j < c.num_elements; j++) c.x[j] >>= dshift;
            for (int j = 0; j < c.num_elements; j++) *((float *)(&c.x[j])) = __int2float_rn(c.x[j]);
            wmma::store_matrix_sync(coeff + wid * 16 * 16, c, 16, wmma::mem_row_major);

            __syncwarp();

            for (int k4 = 0; k4 < 16; k4 += 4) {
                float vc_frag[4][4];
                float bh_frag[ESD8 / 8][4];

                /* ignore bank confict for simplicity */
                for (int i = 0; i < 4; i++) {
                    for (int k = 0; k < 4; k++) {
                        vc_frag[i][k] = ((float *)coeff)[(wid * 16 + (lid / 8) * 4 + i) * 16 + k4 + k];
                    }
                }

                for (int j = 0; j < ESD8 / 8; j++) {
                    for (int k = 0; k < 4; k++) {
                        bh_frag[j][k] = b_head[((lid & 7) * (ESD8 / 8) + j) * CSD16 + l + k4 + k];
                    }
                }

                for (int k = 0; k < 4; k++) {
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < ESD8 / 8; j++) {
                            vh_frag[i][j] += vc_frag[i][k] * bh_frag[j][k];
                        }
                    }
                }
            }
        }

        /* ignore bank confict for simplicity */
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < ESD8 / 8; j++) {
                head[(wid * 16 + (lid / 8) * 4 + i) * ESD8 + (lid & 7) * (ESD8 / 8) + j] = vh_frag[i][j];
            }
        }

        __syncthreads();

        for (int s = 0; s < 128; s++) {
            if (ind + s >= n) break;

            float h0[ESD8], h1[ESD8];

            for (int i = 0; i < ESD8; i++) h1[i] = head[s * ESD8 + i];
            for (int i = 0; i < ESD8; i++) {
                h0[i] = h1[i] - t_center[i];    /// same direction
                h1[i] = h1[i] + t_center[i];    /// opposite direction
            }

            int bias = 0;
            #pragma unroll
            for (int l = ESD8 - 1; l >= 0; l--) {
                float c0 = roundf(h0[l] * dh_inorm[l]);
                float c1 = roundf(h1[l] * dh_inorm[l]);
                #pragma unroll
                for (int j = 0; j <= l; j++) {
                    h0[j] -= c0 * dh_head[bias + j];
                    h1[j] -= c1 * dh_head[bias + j];
                }
                bias += l + 1;
            }

            float n0 = 0.0f, n1 = 0.0f;

            for (int i = 0; i < ESD8; i++) {
                n0 += h0[i] * h0[i];
                n1 += h1[i] * h1[i];
            }

            if (n0 < radius && t_sbuc_num < sbucSize) {
                sbuc[sbucSize * tid + t_sbuc_num] = 2 * (ind + s);
                t_sbuc_num++;
            }
            if (n1 < radius && t_sbuc_num < sbucSize) {
                sbuc[sbucSize * tid + t_sbuc_num] = 2 * (ind + s) + 1;
                t_sbuc_num++;
            }
        }

        __syncwarp();

        {
            int buc_full = t_sbuc_num >= sbucSize - 1 ? 1 : 0;
            int need_s2g =  __reduce_or_sync(0xffffffff, buc_full);
            if (lid == 0) signal[wid] = need_s2g;
            __syncthreads();
            if (tid == 0) {
                int sig[8];
                for (int i = 0; i < 8; i++) sig[i] = signal[i];
                signal[0] = sig[0] | sig[1] | sig[2] | sig[3] | sig[4] | sig[5] | sig[6] | sig[7];
            }
            __syncthreads();
            need_s2g = signal[0];
            if (need_s2g) {
                int pos = atomicAdd(t_gbuc_num, t_sbuc_num);
                int num = min(t_sbuc_num, out_max_size - pos);
                t_sbuc_num = 0;

                signal[2 * tid + 0] = pos;
                signal[2 * tid + 1] = num;
                __syncwarp();

                for (int i = 0; i < 32; i++) {
                    int sig[2];
                    sig[0] = signal[64 * wid + 2 * i + 0];
                    sig[1] = signal[64 * wid + 2 * i + 1];
                    if ((int)lid < sig[1]) w_out[i * out_max_size + sig[0] + lid] = sbuc[sbucSize * (wid * 32 + i) + lid];
                }
            }
        }

        for (int l = 0; l < 4 && ind + stride < n ; l++) {
            _pfch_32vec(vpk, in + (ind + stride + l * 32) * (uint64_t) CSD, CSD, CSD16, wid, lid);
            _unpk_32vec(vec + l * 32 * CSD16, vpk, CSD, CSD16, wid, lid);
            __syncwarp();
        }

        ind += stride;
    }

    int pos = atomicAdd(t_gbuc_num, t_sbuc_num);
    int num = min(t_sbuc_num, out_max_size - pos);

    signal[2 * tid + 0] = pos;
    signal[2 * tid + 1] = num;
    __syncwarp();

    for (int i = 0; i < 32; i++) {
        int sig[2];
        sig[0] = signal[64 * wid + 2 * i + 0];
        sig[1] = signal[64 * wid + 2 * i + 1];
        if ((int)lid < sig[1]) {
            w_out[i * out_max_size + sig[0] + lid] = sbuc[sbucSize * (wid * 32 + i) + lid];
        }
    }
}


template __global__ void dh_buc_kernel<176, 48>(
    uint32_t *__restrict__ out, int out_max_size, float *__restrict__ center, float radius, 
    int batch, int8_t *__restrict__ in, int n, int CSD, dh_data_t *__restrict__ data
);

template __global__ void dh_buc_kernel<176, 40>(
    uint32_t *__restrict__ out, int out_max_size, float *__restrict__ center, float radius, 
    int batch, int8_t *__restrict__ in, int n, int CSD, dh_data_t *__restrict__ data
);

template __global__ void dh_buc_kernel<176, 32>(
    uint32_t *__restrict__ out, int out_max_size, float *__restrict__ center, float radius, 
    int batch, int8_t *__restrict__ in, int n, int CSD, dh_data_t *__restrict__ data
);
#include "../include/config.h"
#include "../include/common_device.h"
#include "../include/dh_device.h"

#include <mma.h>
#include <cuda_runtime.h>

constexpr int dh_nints     = dh_nbits / 32;

constexpr int batchVecs    = 64;
constexpr int unit         = 1024;
constexpr int C_batch      = unit / dhr_threads * 4;
constexpr int V_batch      = batchVecs / 8;
constexpr int rbuc_size    = 3;
constexpr int sbuc_size    = 512;
constexpr int sbuc_freq    = 16;
constexpr int gbuc_freq    = 256;
constexpr int pfch_vec_xor = batchVecs * dh_nints;

typedef wmma::fragment<wmma::matrix_a, 8, 8, 128, wmma::experimental::precision::b1, wmma::row_major> wma_t;
typedef wmma::fragment<wmma::matrix_b, 8, 8, 128, wmma::experimental::precision::b1, wmma::col_major> wmb_t;
typedef wmma::fragment<wmma::accumulator, 8, 8, 128, int> wacc_t;


static __device__ __forceinline__ void _update_rbuc(int *rbuc) {
    for (int j = 0; j < C_batch; j++) {
        if (rbuc[j * rbuc_size] > rbuc[j * rbuc_size + rbuc_size - 1]) {
            int tmp = rbuc[j * rbuc_size + rbuc_size - 1];
            for (int i = rbuc_size - 1; i > 0; i--) rbuc[j * rbuc_size + i] = rbuc[j * rbuc_size + i - 1];
            rbuc[j * rbuc_size] = tmp;
        }
    }
}

static __device__ __forceinline__ void _pfch_64vec(int *dst, int *src, int tid) {
    for (int i = 0; i < dh_nbits / 256; i++) {
        utils_t::_ldgsts_64b_async(dst + (256 * i + tid) * 2, src + (256 * i + tid) * 2);
    }
    utils_t::_commit_async_group();
}

static __device__ __forceinline__ void _sbuc_2_gbuc(int *out, int *num_out, int out_max_size, 
                                                    int *sbuc,  int *sbuc_num, int wid, int lid) {
    __syncwarp();
    
    int b_size = sbuc_num[wid];
    if (b_size > sbuc_size / 2) b_size = sbuc_size / 2;
    int pos;
    if (lid == 0) {
        pos = atomicAdd(num_out, b_size);
        sbuc_num[wid] = 0;
    }
    pos = __shfl_sync(0xffffffff, pos, 0);
    b_size = min(b_size, out_max_size - pos);

    int *wsbuc = sbuc + wid * sbuc_size;
    if (lid < b_size) ((int2 *)out)[pos + lid] = ((int2 *)wsbuc)[lid];
    if (lid + 32 < b_size) ((int2 *)out)[pos + lid + 32] = ((int2 *)wsbuc)[lid + 32];
    if (lid + 64 < b_size) ((int2 *)out)[pos + lid + 64] = ((int2 *)wsbuc)[lid + 64];
    if (lid + 96 < b_size) ((int2 *)out)[pos + lid + 96] = ((int2 *)wsbuc)[lid + 96];
    if (lid + 128 < b_size) ((int2 *)out)[pos + lid + 128] = ((int2 *)wsbuc)[lid + 128];
    if (lid + 160 < b_size) ((int2 *)out)[pos + lid + 160] = ((int2 *)wsbuc)[lid + 160];
    if (lid + 192 < b_size) ((int2 *)out)[pos + lid + 192] = ((int2 *)wsbuc)[lid + 192];
    if (lid + 224 < b_size) ((int2 *)out)[pos + lid + 224] = ((int2 *)wsbuc)[lid + 224];
}


__global__ void dh_red_kernel(int *__restrict__ out, int *num_out, int out_max_size, int *in, int n, int th) {
    /// registers
    int32_t rbuc[C_batch * rbuc_size];

    /// shared memory
    extern __shared__ int8_t dy_buf[];
    int   *sh_vec = (int *)(&dy_buf[0]);
    int     *sbuc = (int *)(&sh_vec[pfch_vec_xor * 2]);
    int *sbuc_num = (int *)(&sh_vec[pfch_vec_xor * 2 + sbuc_size * 8]);


    /// const variables
    const unsigned int tid = utils_t::_thread_id();
    const unsigned int lid = utils_t::_lane_id();
    const unsigned int wid = utils_t::_thread_id() / 32;
    
    /// for loop control
    int32_t unacc_iters = 0;
    int32_t task = blockIdx.x * unit;
    
    /// clear buckets
    if (tid < 8) sbuc_num[tid] = 0;

    /// main loop
    while (task + unit < n) {
        #if SPLIT_DHR
        if (task >= 262144) break;
        #endif
        wma_t center_frag[C_batch][dh_nbits / 128];

        for (int i = 0; i < C_batch * rbuc_size; i++) rbuc[i] = 0xffffffff;
        
        #pragma unroll
        for (int b = -1; b < C_batch; b++) {
            int curr_vec_bias = (b & 1) * 8 * 8 * dh_nints;
            int pfch_vec_bias = ((b + 1) & 1) * 8 * 8 * dh_nints;
            if (b >= 0) {
                utils_t::_wait_async_group();
                __syncthreads();
            }

            if (b != C_batch - 1) {
                int *dst = sh_vec + pfch_vec_bias + wid * 8 * dh_nints;
                int *src = in + (task + (wid * C_batch + (b + 1)) * 8) * (uint64_t) dh_nints;
                for (int i = 0; i < dh_nbits / 256; i++) {
                    utils_t::_ldgsts_64b_async(dst + (i * 32 + lid) * 2, src + (i * 32 + lid) * 2);
                }
                utils_t::_commit_async_group();
            }

            if (b >= 0) {
                for (int l = 0; l < dh_nbits / 128; l++) {
                    wmma::load_matrix_sync(center_frag[b][l], sh_vec + curr_vec_bias + wid * 8 * dh_nints + l * 4, dh_nbits);
                }
            }

            if (b >= 0) {
                wacc_t dp_frag[C_batch];
                wmb_t vec_frag[dh_nbits / 128];
                
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    for (int l = 0; l < dh_nbits / 128; l++) {
                        wmma::load_matrix_sync(vec_frag[l], sh_vec + curr_vec_bias + j * 8 * dh_nints + l * 4, dh_nbits);
                    }
                    for (int i = 0; i < b + (j > wid ? 1 : 0); i++) {
                        dp_frag[i].x[0] = -1;
                        dp_frag[i].x[1] = -1;
                        for (int l = 0; l < dh_nbits / 128; l++) {
                            wmma::bmma_sync(dp_frag[i], center_frag[i][l], vec_frag[l], dp_frag[i]);
                        }
                        if ((uint32_t)dp_frag[i].x[0] < (uint32_t)th) rbuc[i * rbuc_size] = task + (j * C_batch + b) * 8 + (lid & 3) * 2 + 0;
                        if ((uint32_t)dp_frag[i].x[1] < (uint32_t)th) rbuc[i * rbuc_size] = task + (j * C_batch + b) * 8 + (lid & 3) * 2 + 1;
                    }
                }
            }

            if (b >= 0) _update_rbuc(rbuc);
        }

        unacc_iters += unit / batchVecs;
        
        int ind = task + unit;
        int curr_vec_bias = 0;

        _pfch_64vec(sh_vec, in + ind * dh_nints, tid);

        utils_t::_wait_async_group();
        __syncthreads();

        while (ind < n) {
            if (ind + batchVecs < n) {
                _pfch_64vec(sh_vec + (curr_vec_bias ^ pfch_vec_xor), in + (ind + batchVecs) * dh_nints, tid);
            }

            for (int j = 0; j < V_batch; j++) {
                wacc_t dp_frag[C_batch];
                wmb_t vec_frag[dh_nbits / 128];
                for (int l = 0; l < dh_nbits / 128; l++) {
                    wmma::load_matrix_sync(vec_frag[l], sh_vec + curr_vec_bias + j * 8 * dh_nints + 4 * l, dh_nbits);
                }
                for (int i = 0; i < C_batch; i++) {
                    dp_frag[i].x[0] = -1;
                    dp_frag[i].x[1] = -1;
                    for (int l = 0; l < dh_nbits / 128; l++) {
                        wmma::bmma_sync(dp_frag[i], center_frag[i][l], vec_frag[l], dp_frag[i]);
                    }
                    if ((uint32_t) dp_frag[i].x[0] < (uint32_t) th) rbuc[i * rbuc_size] = ind + j * 8 + (lid & 3) * 2 + 0;
                    if ((uint32_t) dp_frag[i].x[1] < (uint32_t) th) rbuc[i * rbuc_size] = ind + j * 8 + (lid & 3) * 2 + 1;
                }
                if (j == V_batch - 1) {
                    utils_t::_wait_async_group();
                    __syncthreads();
                }
            }

            _update_rbuc(rbuc);

            if (!(++unacc_iters % sbuc_freq) || ind + batchVecs >= n) {
                uint32_t b_size = sbuc_num[wid];

                uint32_t db[C_batch];
                uint32_t tdb = 0;
                for (int i = 0; i < C_batch; i++) {
                    db[i] = rbuc_size;
                    for (int j = 0; j < rbuc_size; j++) db[i] -= ((uint32_t ) rbuc[i * rbuc_size + j]) >> 31;
                    tdb += db[i];
                }

                uint32_t rpos = tdb;
                for (int i = 0; i < 5; i++) {
                    uint64_t tmp = __shfl_up_sync(0xffffffff, rpos, 1 << i);
                    if (lid >= (1 << i)) rpos += tmp;
                }

                if (lid == 31) {
                    sbuc_num[wid] = b_size + rpos;
                }

                rpos -= tdb;

                int *wsbuc = sbuc + wid * sbuc_size;
                int pos = 2 * (b_size + rpos);
                for (int i = 0; i < C_batch; i++) {
                    int ctr_id = task + wid * unit / 8 + i * 8 + (lid / 4);
                    if (rbuc[i * rbuc_size + 1] >= 0 && pos + 1 < sbuc_size) { wsbuc[pos] = ctr_id; wsbuc[pos + 1] = rbuc[i * rbuc_size + 1]; }
                    if (rbuc[i * rbuc_size + 2] >= 0 && pos + 3 < sbuc_size) { wsbuc[pos + 2] = ctr_id; wsbuc[pos + 3] = rbuc[i * rbuc_size + 2]; }
                    if (rbuc[i * rbuc_size + 0] >= 0 && pos + 5 < sbuc_size) { wsbuc[pos + 4] = ctr_id; wsbuc[pos + 5] = rbuc[i * rbuc_size + 0]; }
                    pos += db[i] * 2;
                }

                for (int i = 0; i < C_batch * rbuc_size; i++) rbuc[i] = 0xffffffff;
            }


            if (unacc_iters >= gbuc_freq) {
                unacc_iters -= gbuc_freq;
                _sbuc_2_gbuc(out, num_out, out_max_size, sbuc, sbuc_num, wid, lid);
            }

            ind += batchVecs;
            curr_vec_bias ^= pfch_vec_xor;
        }

        task += unit * gridDim.x;
    }

    _sbuc_2_gbuc(out, num_out, out_max_size, sbuc, sbuc_num, wid, lid);
}

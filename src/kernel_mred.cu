#include "../include/config.h"
#include "../include/common_device.h"
#include "../include/bgj_hd_device.h"

#include <mma.h>

using namespace nvcuda;

#define RECORD_CLK 1

typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> wma_t;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> wmb_t;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> wacc_t;
typedef red_traits_t traits;

#define blockThreads traits::blockThreads
#define blockWarps traits::blockWarps
#define rbuc_size traits::rbuc_size
#define sbuc_size traits::sbuc_size
#define unit traits::red_unit
#define batchVecs traits::red_batchVecs
#define C_batch (unit / blockWarps / 16)
#define V_batch (batchVecs / 16)

template <int32_t layer>
static __device__ __forceinline__ void _tail_mask(int32_t *vnorm, int n, uint32_t tid) {
    if (n % batchVecs) {
        if (tid < (int) batchVecs - (n % batchVecs)) {
            if (layer) ((uint64_t *)vnorm)[n + tid] = 0x3fffffff00000000ULL; 
            else vnorm[n + tid] = 0x3fffffff;
        }
    }
}
template <int32_t layer>
static __device__ __forceinline__ void _update_rbuc(int *rbuc) {
    for (int j = 0; j < C_batch * 2; j++) {
        if (rbuc[j * rbuc_size] > rbuc[j * rbuc_size + rbuc_size - 1]) {
            int tmp = rbuc[j * rbuc_size + rbuc_size - 1];
            for (int i = rbuc_size - 1; i > 0; i--) rbuc[j * rbuc_size + i] = rbuc[j * rbuc_size + i - 1];
            rbuc[j * rbuc_size] = tmp;
        }
    }
}
template <int32_t layer>
static __device__ __forceinline__ void _head_prep(int32_t *ctr_ind, int32_t *ctr_th, int32_t *center_ind, int32_t *center_th,
                                                    int32_t *vnorm, int task, int goal_norm, uint32_t wid, uint32_t lid, uint32_t tid) {
    constexpr int thread_ctrs = unit / blockThreads;

    int ctr_data[2 * thread_ctrs];
    for (int i = 0; i < 2 * thread_ctrs && layer; i++) 
        ctr_data[i] = vnorm[2 * task + tid * 2 * thread_ctrs + i];
    for (int i = 0; i < thread_ctrs; i++) {
        ctr_ind[tid * thread_ctrs + i] = layer ? ctr_data[2 * i] : task + tid * thread_ctrs + i;
        ctr_th[tid * thread_ctrs + i] = goal_norm - (layer ? ctr_data[2 * i + 1] : vnorm[task + tid * thread_ctrs + i]);
    }
    __syncwarp();
    for (int i = 0; i < C_batch; i++) {
        center_th[2 * i + 0]  = ctr_th[wid * C_batch * 16 + i * 16 + lid / 4 + 0];
        center_th[2 * i + 1]  = ctr_th[wid * C_batch * 16 + i * 16 + lid / 4 + 8];
        center_ind[2 * i + 0] = layer ? ctr_ind[wid * C_batch * 16 + i * 16 + lid / 4 + 0] : 
                                            task + wid * C_batch * 16 + i * 16 + lid / 4 + 0;
        center_ind[2 * i + 1] = layer ? ctr_ind[wid * C_batch * 16 + i * 16 + lid / 4 + 8] :
                                            task + wid * C_batch * 16 + i * 16 + lid / 4 + 8;
    }
}
template <int32_t layer>
static __device__ __forceinline__ void _task_head(int32_t *rbuc, wma_t center_frag[][traits::vec_nbytes / 16], int32_t *center_th,
                                                    int8_t *sh_vec, const int8_t *vec_pad16, int32_t *ctr_ind, int32_t *ctr_th, 
                                                    int32_t goal_norm, uint32_t wid, uint32_t lid, uint32_t CSD16) {
    for (int i = 0; i < C_batch * 2 * rbuc_size; i++) rbuc[i] = 0xffffffff;

    int32_t thread_norm[blockWarps * 4];
    int32_t thread_ind[blockWarps * 4];

    #pragma unroll
    for (int b = -1; b < (int) C_batch; b++) {
        int curr_vec_bias = (b & 1) * blockWarps * 16 * CSD16;
        int pfch_vec_bias = ((b + 1) & 1) * blockWarps * 16 * CSD16;

        if (b >= 0) {
            utils_t::_wait_async_group();
            __syncthreads();
        }

        if (b != (int) C_batch - 1) {
            int r[8];
            for (int j = 0; j < 8; j++) r[j] = ctr_ind[(wid * C_batch + (b + 1)) * 16 + (lid & 16) / 2 + j];
            for (int j = 0; j < 8; j++) {
                if ((lid & 15) < CSD16 / 16) {
                    utils_t::_ldgsts_128b_async(sh_vec + (wid * 16 + (lid & 16) / 2 + j) * CSD16 + (lid & 15) * 16 + 
                                                            pfch_vec_bias, vec_pad16 + r[j] * (uint64_t) CSD16 + (lid & 15) * 16);
                }
            }
            utils_t::_commit_async_group();
        }
        
        if (b >= 0) {
            for (int l = 0; l < CSD16; l += 16) {
                wmma::load_matrix_sync(center_frag[b][l / 16], sh_vec + curr_vec_bias + wid * 16 * CSD16 + l, CSD16);
            }
            for (int j = 0; j < blockWarps; j++) {
                thread_norm[j * 4 + 0] = goal_norm - ctr_th[(j * C_batch + b) * 16 + (lid % 4) * 2 + 0];
                thread_norm[j * 4 + 1] = goal_norm - ctr_th[(j * C_batch + b) * 16 + (lid % 4) * 2 + 1];
                thread_norm[j * 4 + 2] = goal_norm - ctr_th[(j * C_batch + b) * 16 + (lid % 4) * 2 + 8];
                thread_norm[j * 4 + 3] = goal_norm - ctr_th[(j * C_batch + b) * 16 + (lid % 4) * 2 + 9];
                thread_ind[j * 4 + 0] = ctr_ind[(j * C_batch + b) * 16 + (lid % 4) * 2 + 0];
                thread_ind[j * 4 + 1] = ctr_ind[(j * C_batch + b) * 16 + (lid % 4) * 2 + 1];
                thread_ind[j * 4 + 2] = ctr_ind[(j * C_batch + b) * 16 + (lid % 4) * 2 + 8];
                thread_ind[j * 4 + 3] = ctr_ind[(j * C_batch + b) * 16 + (lid % 4) * 2 + 9];
            }
        }

        if (b >= 0) {
            wacc_t dp_frag[C_batch];
            wmb_t vec_frag[traits::vec_nbytes / 16];
            
            #pragma unroll
            for (int j = 0; j < blockWarps; j++) {
                for (int l = 0; l < CSD16; l += 16) {
                    wmma::load_matrix_sync(vec_frag[l / 16], sh_vec + curr_vec_bias + j * 16 * CSD16 + l, CSD16);
                }
                for (int i = 0; i < b + (j > wid ? 1 : 0); i++) {
                    for (int k = 0; k < 8; k++) dp_frag[i].x[k] = center_th[i * 2 + (k & 2) / 2];
                    for (int l = 0; l < CSD16; l += 16) {
                        wmma::mma_sync(dp_frag[i], center_frag[i][l / 16], vec_frag[l / 16], dp_frag[i]);
                    }
                    for (int k = 0; k < 8; k++) {
                        #if USE_IF
                        if (dp_frag[i].x[k] >= thread_norm[j * 4 + (k & 1) + (k & 4) / 2]) 
                            rbuc[(i * 2 + (k & 2) / 2) * rbuc_size] = thread_ind[j * 4 + (k & 1) + (k & 4) / 2];
                        #else
                        int32_t cmp = dp_frag[i].x[k] - thread_norm[j * 4 + (k & 1) + (k & 4) / 2];
                        int32_t entry = thread_ind[j * 4 + (k & 1) + (k & 4) / 2];
                        int32_t val = (cmp & 0x80000000) | entry;
                        rbuc[(i * 2 + (k & 2) / 2) * rbuc_size] = max(rbuc[(i * 2 + (k & 2) / 2) * rbuc_size], val);
                        #endif
                    }
                }
            }
        }

        if (b >= 0) _update_rbuc<layer>(rbuc);
    }
}
template <int32_t layer>
static __device__ __forceinline__ void _normth_prep(int32_t *ndst, int32_t *idst, int32_t *src, uint32_t lid) {
    for (int j = 0; j < V_batch; j++) {
        if (layer) {
            idst[j * 4 + 0] = src[j * 32 + (lid % 4) * 4 + 0];
            ndst[j * 4 + 0] = src[j * 32 + (lid % 4) * 4 + 1];
            idst[j * 4 + 1] = src[j * 32 + (lid % 4) * 4 + 2];
            ndst[j * 4 + 1] = src[j * 32 + (lid % 4) * 4 + 3];
            idst[j * 4 + 2] = src[j * 32 + (lid % 4) * 4 + 16];
            ndst[j * 4 + 2] = src[j * 32 + (lid % 4) * 4 + 17];
            idst[j * 4 + 3] = src[j * 32 + (lid % 4) * 4 + 18];
            ndst[j * 4 + 3] = src[j * 32 + (lid % 4) * 4 + 19];
        } else {
            ndst[j * 4 + 0] = src[j * 16 + (lid % 4) * 2 + 0];
            ndst[j * 4 + 1] = src[j * 16 + (lid % 4) * 2 + 1];
            ndst[j * 4 + 2] = src[j * 16 + (lid % 4) * 2 + 8];
            ndst[j * 4 + 3] = src[j * 16 + (lid % 4) * 2 + 9];
        }
    }
}
static __device__ __forceinline__ void _buffer_init(int32_t *sbuc_num, uint32_t tid) {
    if (tid < 8) sbuc_num[tid] = 0;
}
static __device__ __forceinline__ void _l0_vec_pfch(int32_t *ndst, int8_t *vdst, int32_t *nsrc, const int8_t *vsrc, 
                                                    uint32_t tid, uint32_t CSD16) {
    if (tid < batchVecs / 2) utils_t::_ldgsts_64b_async(ndst + tid * 2, nsrc + tid * 2);
    if (tid < batchVecs * CSD16 / 16) utils_t::_ldgsts_128b_async(vdst + tid * 16, vsrc + tid * 16);
    if (tid + blockThreads < batchVecs * CSD16 / 16) 
        utils_t::_ldgsts_128b_async(vdst + (tid + blockThreads) * 16, vsrc + (tid + blockThreads) * 16);
    if (tid + blockThreads * 2 < batchVecs * CSD16 / 16) 
        utils_t::_ldgsts_128b_async(vdst + (tid + blockThreads * 2) * 16, vsrc + (tid + blockThreads * 2) * 16);
    if (tid + blockThreads * 3 < batchVecs * CSD16 / 16 && 3 * blockThreads < batchVecs * CSD16 / 16)
        utils_t::_ldgsts_128b_async(vdst + (tid + blockThreads * 3) * 16, vsrc + (tid + blockThreads * 3) * 16);
    if (tid + blockThreads * 4 < batchVecs * CSD16 / 16 && 4 * blockThreads < batchVecs * CSD16 / 16)
        utils_t::_ldgsts_128b_async(vdst + (tid + blockThreads * 4) * 16, vsrc + (tid + blockThreads * 4) * 16);
    if (tid + blockThreads * 5 < batchVecs * CSD16 / 16 && 5 * blockThreads < batchVecs * CSD16 / 16)
        utils_t::_ldgsts_128b_async(vdst + (tid + blockThreads * 5) * 16, vsrc + (tid + blockThreads * 5) * 16);
    utils_t::_commit_async_group();
}
static __device__ __forceinline__ void _l1_vid_pfch(int *dst, int *src, uint32_t tid) {
    if (tid < batchVecs / 2) {
        utils_t::_ldgsts_128b_async(dst + tid * 4, src + tid * 4);
        utils_t::_commit_async_group();
    }
}
static __device__ __forceinline__ void _l1_vec_pfch(int8_t *vdst, const int8_t *vec_pad16, int32_t *vid, 
                                                    uint32_t tid, uint32_t CSD16) {
    constexpr int b = batchVecs / blockWarps / 2;
    int r[b];
    for (int j = 0; j < b; j++) r[j] = vid[(tid / 16) * b * 2 + j * 2];
    for (int j = 0; j < b; j++) {
        if ((tid & 15) < CSD16 / 16) {
            utils_t::_ldgsts_128b_async(vdst + ((tid / 16) * b + j) * CSD16 + (tid & 15) * 16,
                                        vec_pad16 + r[j] * (uint64_t) CSD16 + (tid & 15) * 16);
        }
    }
    utils_t::_commit_async_group();
}
template <int32_t layer>
static __device__ __forceinline__ void _rbuc_2_sbuc(int32_t *sbuc_num, uint32_t *sbuc, int32_t *rbuc, 
                                                    int32_t *center_ind, uint32_t wid, uint32_t lid) {
    uint32_t b_size = sbuc_num[wid];

    uint32_t db[2 * C_batch];
    uint32_t tdb = 0;
    for (int i = 0; i < 2 * C_batch; i++) {
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

    uint32_t *wsbuc = sbuc + wid * sbuc_size;
    int pos = 2 * (b_size + rpos);
    for (int i = 0; i < 2 * C_batch; i++) {
        if (rbuc[i * rbuc_size + 1] >= 0 && pos + 1 < sbuc_size) { wsbuc[pos] = center_ind[i]; wsbuc[pos + 1] = rbuc[i * rbuc_size + 1]; }
        if (rbuc[i * rbuc_size + 2] >= 0 && pos + 3 < sbuc_size) { wsbuc[pos + 2] = center_ind[i]; wsbuc[pos + 3] = rbuc[i * rbuc_size + 2]; }
        if (rbuc[i * rbuc_size + 0] >= 0 && pos + 5 < sbuc_size) { wsbuc[pos + 4] = center_ind[i]; wsbuc[pos + 5] = rbuc[i * rbuc_size + 0]; }
        pos += db[i] * 2;
    }

    for (int i = 0; i < 2 * C_batch * rbuc_size; i++) rbuc[i] = 0xffffffff;
}
static __device__ __forceinline__ void _sbuc_2_gbuc(int *out, int *num_out, int out_max_size, uint32_t *sbuc, 
                                                    int *sbuc_num, uint32_t wid, int32_t lid) {
    __syncwarp();
    
    int b_size = sbuc_num[wid];
    if (b_size > sbuc_size / 2) b_size = sbuc_size / 2;
    int pos;
    if (lid == 0) {
        pos = atomicAdd(num_out, b_size);
        sbuc_num[wid] = 0;
        if (pos > out_max_size + 1073741824) num_out[0] = out_max_size + 1073741824;
    }
    pos = __shfl_sync(0xffffffff, pos, 0);
    b_size = min(b_size, out_max_size - pos);

    uint32_t *wsbuc = sbuc + wid * sbuc_size;
    if (lid < b_size) ((int2 *)out)[pos + lid] = ((int2 *)wsbuc)[lid];
    if (lid + 32 < b_size) ((int2 *)out)[pos + lid + 32] = ((int2 *)wsbuc)[lid + 32];
    if (lid + 64 < b_size) ((int2 *)out)[pos + lid + 64] = ((int2 *)wsbuc)[lid + 64];
    if (lid + 96 < b_size) ((int2 *)out)[pos + lid + 96] = ((int2 *)wsbuc)[lid + 96];
    if (lid + 128 < b_size) ((int2 *)out)[pos + lid + 128] = ((int2 *)wsbuc)[lid + 128];
    if (lid + 160 < b_size) ((int2 *)out)[pos + lid + 160] = ((int2 *)wsbuc)[lid + 160];
    if (lid + 192 < b_size) ((int2 *)out)[pos + lid + 192] = ((int2 *)wsbuc)[lid + 192];
    if (lid + 224 < b_size) ((int2 *)out)[pos + lid + 224] = ((int2 *)wsbuc)[lid + 224];
}


#undef blockThreads
#undef blockWarps
#undef rbuc_size
#undef sbuc_size
#undef unit
#undef batchVecs
#undef C_batch
#undef V_batch


__global__ void _multi_reduce_prepare(int32_t *vids, int *n_ptr, int max_n) {
    __shared__ int sh_n;

    constexpr int layer = 1;
    int32_t *vnorm = vids + blockIdx.x * 2 * max_n;

    if (threadIdx.x == 0) {
        sh_n = n_ptr[blockIdx.x];
    }
    __syncthreads();
    int old_n = sh_n;
    
    int n = old_n;
    if (n > max_n) n = max_n;
    if (n <= 512 && n) {
        if (n + threadIdx.x < 513) {
            if (layer) ((uint64_t *)vnorm)[n + threadIdx.x] = 0x3fffffff00000000ULL; 
            else vnorm[n + threadIdx.x] = 0x3fffffff;
        }
        n = 513;
    }
    if (threadIdx.x == 0) {
        if (n != old_n) n_ptr[blockIdx.x] = n;
    }
}

template <int32_t CSD16, uint32_t sbuc_freq, uint32_t gbuc_freq>
__global__ void _multi_reduce_kernel(int *out, int *num_out, int out_max_size, const int8_t *__restrict__ vec_pad16, 
                                    int32_t *__restrict__ vids, int *n_ptr, int buc_max_size, int32_t goal_norm) {
    typedef red_traits_t traits;
    typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> wma_t;
    typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> wmb_t;
    typedef wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> wacc_t;

    uint32_t n = n_ptr[blockIdx.x];
    int32_t *vnorm = vids + blockIdx.x * 2 * buc_max_size;

    #if RECORD_CLK
    uint32_t start_clk = utils_t::_clock() >> 32;
    #endif

    /// constants
    constexpr unsigned int layer        = 1;
    constexpr unsigned int vec_nbytes   = traits::vec_nbytes;
    constexpr unsigned int batchVecs    = traits::red_batchVecs;
    constexpr unsigned int unit         = traits::red_unit;
    constexpr unsigned int C_batch      = unit / traits::blockWarps / 16;
    constexpr unsigned int V_batch      = batchVecs / 16;

    constexpr unsigned int rbuc_size    = traits::rbuc_size;

    /// registers
    int32_t center_th[C_batch * 2];
    int32_t center_ind[C_batch * 2];
    int32_t thread_norm[V_batch * 4];
    int32_t thread_ind[V_batch * 4];
    int32_t rbuc[C_batch * 2 * rbuc_size];

    /// memory
    extern __shared__ int8_t dy_buf[];
    constexpr unsigned sh_vec_size = vec_nbytes * 2 * (traits::blockWarps * 16 > batchVecs ? traits::blockWarps * 16 : batchVecs);
    int32_t  *ctr_ind  = (int32_t *)  &dy_buf[0];
    int32_t  *ctr_th   = (int32_t *)  &dy_buf[unit * 4];
    int32_t  *sh_norm  = (int32_t *)  &dy_buf[unit * 8];
    int8_t   *sh_vec   = (int8_t *)   &dy_buf[unit * 8 + batchVecs * 4 * (layer ? 6 : 2)];
    int32_t  *sbuc_num = (int32_t *)  &sh_vec[sh_vec_size];
    uint32_t *sbuc     = (uint32_t *) &sh_vec[sh_vec_size + traits::blockWarps * 4];

    /// const variables
    const unsigned int tid = utils_t::_thread_id();
    const unsigned int lid = utils_t::_lane_id();
    const unsigned int wid = utils_t::_thread_id() / 32;

    /// tail mask
    _tail_mask<layer>(vnorm, n, tid);

    /// clear buckets
    _buffer_init(sbuc_num, tid);
    
    __syncthreads();
    
    /// for loop control
    int32_t unacc_iters = 0;
    int32_t task = 0;
    constexpr uint32_t pfch_vec_xor = batchVecs * vec_nbytes;
    constexpr uint32_t pfch_norm_xor = batchVecs * (layer ? 2 : 1);
    #define pfch_vec_bias (curr_vec_bias ^ pfch_vec_xor)
    #define pfch_norm_bias (layer ? ((curr_norm_bias + 2 * pfch_norm_xor) % (3 * pfch_norm_xor)) : (curr_norm_bias ^ pfch_norm_xor))
    #define next_norm_bias (layer ? ((curr_norm_bias + pfch_norm_xor) % (3 * pfch_norm_xor)) : (curr_norm_bias ^ pfch_norm_xor))

    /// main loop
    while (task + unit < n) {
        wma_t center_frag[C_batch][vec_nbytes / 16];

        _head_prep<layer>(ctr_ind, ctr_th, center_ind, center_th, vnorm, task, goal_norm, wid, lid, tid);
        _task_head<layer>(rbuc, center_frag, center_th, sh_vec, vec_pad16, ctr_ind, ctr_th, goal_norm, wid, lid, CSD16);
        unacc_iters += unit / batchVecs;

        uint32_t ind = task + unit;
        uint32_t curr_vec_bias = 0;
        uint32_t curr_norm_bias = 0;

        if (layer) {
            _l1_vid_pfch(sh_norm, vnorm + 2 * ind, tid);
            utils_t::_wait_async_group();
            __syncthreads();
            _l1_vec_pfch(sh_vec, vec_pad16, sh_norm, tid, CSD16);
            _l1_vid_pfch(sh_norm + pfch_norm_xor, vnorm + 2 * ind + batchVecs * 2, tid);
        } else {
            _l0_vec_pfch(sh_norm, sh_vec, vnorm + ind, vec_pad16 + ind * CSD16, tid, CSD16);
        }
        utils_t::_wait_async_group();
        __syncthreads();

        _normth_prep<layer>(thread_norm, thread_ind, sh_norm, lid);

        while (ind < n) {
            if (layer) {
                if (ind + batchVecs * 2 < n) {
                    _l1_vid_pfch(sh_norm + pfch_norm_bias, vnorm + 2 * ind + batchVecs * 4, tid);
                }
                if (ind + batchVecs < n) {
                    _l1_vec_pfch(sh_vec + pfch_vec_bias, vec_pad16, sh_norm + next_norm_bias, tid, CSD16);
                }
            } else if (ind + batchVecs < n) {
                _l0_vec_pfch(sh_norm + pfch_norm_bias, sh_vec + pfch_vec_bias, vnorm + ind + batchVecs, 
                                     vec_pad16 + (ind + batchVecs) * CSD16, tid, CSD16); 
            }

            for (int j = 0; j < V_batch; j++) {
                wacc_t dp_frag[C_batch];
                wmb_t vec_frag[CSD16 / 16];
                for (int l = 0; l < CSD16; l += 16) {
                    wmma::load_matrix_sync(vec_frag[l / 16], sh_vec + curr_vec_bias + CSD16 * 16 * j + l, CSD16);
                }
                for (int i = 0; i < C_batch; i++) {
                    for (int k = 0; k < 8; k++) dp_frag[i].x[k] = center_th[i * 2 + (k & 2) / 2];
                    for (int l = 0; l < CSD16; l += 16) {
                        wmma::mma_sync(dp_frag[i], center_frag[i][l / 16], vec_frag[l / 16], dp_frag[i]);
                    }
                    for (int k = 0; k < 8; k++) {
                        #if USE_IF
                        if (dp_frag[i].x[k] >= thread_norm[j * 4 + (k & 1) + (k & 4) / 2]) 
                            rbuc[(i * 2 + (k & 2) / 2) * rbuc_size] = layer ? thread_ind[j * 4 + (k & 1) + (k & 4) / 2] : 
                                                                 (ind + j * 16 + (k & 4) * 2 + (k & 1) + (lid & 3) * 2);
                        #else
                        int32_t cmp = dp_frag[i].x[k] - thread_norm[j * 4 + (k & 1) + (k & 4) / 2];
                        int32_t entry = layer ? thread_ind[j * 4 + (k & 1) + (k & 4) / 2] : 
                                                (ind + j * 16 + (k & 4) * 2 + (k & 1) + (lid & 3) * 2);
                        int32_t val = (cmp & 0x80000000) | entry;
                        rbuc[(i * 2 + (k & 2) / 2) * rbuc_size] = max(rbuc[(i * 2 + (k & 2) / 2) * rbuc_size], val);
                        #endif
                    }
                }
                if (j == 2) {
                    utils_t::_wait_async_group();
                    __syncthreads();
                }
            }

            _update_rbuc<layer>(rbuc);

            _normth_prep<layer>(thread_norm, thread_ind, sh_norm + next_norm_bias, lid);
            
            if (!(++unacc_iters % sbuc_freq) || ind + batchVecs >= n) {
                _rbuc_2_sbuc<layer>(sbuc_num, sbuc, rbuc, center_ind, wid, lid);
            }

            if (unacc_iters >= gbuc_freq) {
                unacc_iters -= gbuc_freq;
                _sbuc_2_gbuc(out, num_out, out_max_size, sbuc, sbuc_num, wid, lid);
            }

            ind += batchVecs;
            curr_vec_bias = pfch_vec_bias;
            curr_norm_bias = next_norm_bias;
        }

        task += unit;
    }

    _sbuc_2_gbuc(out, num_out, out_max_size, sbuc, sbuc_num, wid, lid);
    
    #if RECORD_CLK
    uint32_t end_clk = utils_t::_clock() >> 32;
    if (end_clk - start_clk == 0xffffffff) {
        ((uint64_t *)&num_out[2])[blockIdx.x * blockDim.x + threadIdx.x] = start_clk;
        ((uint64_t *)&num_out[2])[blockIdx.x * blockDim.x + threadIdx.x + blockDim.x * gridDim.x] = end_clk;
    }
    #endif
}




#define INSTANTIATE(CSD16, sbuc_freq, gbuc_freq) \
    template __global__ void _multi_reduce_kernel<CSD16, sbuc_freq, gbuc_freq>(int *out, int *num_out, int out_max_size, \
                 const int8_t *__restrict__ vec_pad16, int32_t *__restrict__ vids, int *n_ptr, int buc_max_size, int32_t goal_norm)

INSTANTIATE(176, 64, 512);
INSTANTIATE(176, 16, 128);
INSTANTIATE(176, 1, 2);
#if RED_MIN_CSD16 < 176
INSTANTIATE(160, 64, 512);
INSTANTIATE(160, 16, 128);
INSTANTIATE(160, 1, 2);
#endif
#if RED_MIN_CSD16 < 160
INSTANTIATE(144, 64, 512);
INSTANTIATE(144, 16, 128);
INSTANTIATE(144, 1, 2);
#endif
#if RED_MIN_CSD16 < 144
INSTANTIATE(128, 64, 512);
INSTANTIATE(128, 16, 128);
INSTANTIATE(128, 1, 2);
#endif
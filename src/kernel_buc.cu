#include "../include/config.h"
#include "../include/common_device.h"
#include "../include/bgj_hd_device.h"

#include <mma.h>

using namespace nvcuda;

#define RECORD_CLK 1
#define UNPACK_INCLUDED 1

typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> wma_t;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> wmb_t;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> wacc_t;
typedef buc_traits_t traits;


#define blockThreads traits::blockThreads
#define blockWarps traits::blockWarps
#define blockCenters (traits::buc_unit < batchsize ? traits::buc_unit : batchsize)
#define unit traits::buc_unit
#define sbucSize (layer == 0 ? traits::l0_sbucSize : layer == 1 ? traits::l1_sbucSize : traits::l2_sbucSize)
#define batchVecs traits::buc_batchVecs
#define C_batch ((blockCenters * 2 / blockThreads) > 0 ? (blockCenters * 2 / blockThreads) : 1)
#define L12_USE_IF (layer == 1 ? traits::l1_use_if : traits::l2_use_if)

template <int32_t layer>
static __device__ __forceinline__ void _tail_mask(int32_t *vnorm, int n, uint32_t tid) {
    if (n % batchVecs) {
        if (tid < (int) batchVecs - (n % batchVecs)) {
            if (layer == 2) ((uint64_t *)vnorm)[n + tid] = 0x3fffffff00000000ULL; 
            else vnorm[n + tid] = 0x3fffffff; 
        }
        __syncthreads();
    }
}
template <int32_t layer>
static __device__ __forceinline__ void _center_load(wma_t center_frag[][traits::vec_nbytes / 16], const int8_t *warp_center, 
                                                    uint32_t wid, uint32_t lid, uint32_t batchsize, uint32_t CSD16) {
    for (int b = 0; b < C_batch * 16; b += 16) {
        for (int i = 0; i < CSD16; i += 16) 
            wmma::load_matrix_sync(center_frag[b / 16][i / 16], warp_center + b * CSD16 + i, CSD16);
    }
    if (blockCenters < blockThreads / 2) {
        if (wid * 16 >= blockCenters) 
            for (int i = 0; i < CSD16; i += 16) wmma::fill_fragment(center_frag[0][i / 16], 0);
        if (blockCenters < 16 && wid == 0) {
            for (int i = 0; i < CSD16; i += 16) {
                for (int j = 4; j < 8; j++) center_frag[0][i / 16].x[j] = 0;
                if (lid >= blockCenters * 4) {
                    for (int j = 0; j < 4; j++) center_frag[0][i / 16].x[j] = 0;
                }
            }
        }
    }
}
template <int32_t layer>
static __device__ __forceinline__ void _buffer_init(int *sbuc_num, int8_t *sh_vec, uint32_t tid, uint32_t batchsize) {
    for (int i = 0; i < unit / 8; i += blockThreads) {
        ((int2 *)sbuc_num)[i + tid] = {0, 0};
    }
    for (int i = 0; i < 2 * batchVecs * traits::vec_nbytes / 8; i += blockThreads) {
        ((int2 *)sh_vec)[i + tid] = {0, 0};
    }
}
template <int32_t layer>
static __device__ __forceinline__ void _nextnv_pfch(int32_t *ndst, int32_t *nsrc, int2 *vdst, int2 *vsrc, 
                                                    uint32_t int2_per_warp, uint32_t lid, uint32_t tid) {
    if (tid < batchVecs / 4) utils_t::_ldgsts_128b_async(((int4 *)ndst) + tid, ((int4 *)nsrc) + tid);
    if (lid < int2_per_warp) utils_t::_ldgsts_64b_async(vdst + lid, vsrc + lid);
    if (lid + 32 < int2_per_warp) utils_t::_ldgsts_64b_async(vdst + lid + 32, vsrc + lid + 32);
    if (lid + 64 < int2_per_warp) utils_t::_ldgsts_64b_async(vdst + lid + 64, vsrc + lid + 64);
    if (lid + 96 < int2_per_warp) utils_t::_ldgsts_64b_async(vdst + lid + 96, vsrc + lid + 96);
    if (lid + 128 < int2_per_warp) utils_t::_ldgsts_64b_async(vdst + lid + 128, vsrc + lid + 128);
    if (lid + 160 < int2_per_warp) utils_t::_ldgsts_64b_async(vdst + lid + 160, vsrc + lid + 160);
    utils_t::_commit_async_group();
}
template <int32_t layer>
static __device__ __forceinline__ void _normth_prep(int *ndst, int *tdst, int *sh_norm, uint32_t alpha_u32, uint32_t lid) {
    for (int j = 0; j < batchVecs / 16; j++) {
        if (L12_USE_IF || layer == 0) {
            if (layer == 2) {
                ndst[j * 8 + 0] = sh_norm[j * 32 + (lid % 4) * 4 + 0];
                ndst[j * 8 + 1] = sh_norm[j * 32 + (lid % 4) * 4 + 1];
                ndst[j * 8 + 2] = sh_norm[j * 32 + (lid % 4) * 4 + 2];
                ndst[j * 8 + 3] = sh_norm[j * 32 + (lid % 4) * 4 + 3];
                ndst[j * 8 + 4] = sh_norm[j * 32 + (lid % 4) * 4 + 16];
                ndst[j * 8 + 5] = sh_norm[j * 32 + (lid % 4) * 4 + 17];
                ndst[j * 8 + 6] = sh_norm[j * 32 + (lid % 4) * 4 + 18];
                ndst[j * 8 + 7] = sh_norm[j * 32 + (lid % 4) * 4 + 19];
                tdst[j * 4 + 0] = __umulhi(alpha_u32, ndst[j * 8 + 1]);
                tdst[j * 4 + 1] = __umulhi(alpha_u32, ndst[j * 8 + 3]);
                tdst[j * 4 + 2] = __umulhi(alpha_u32, ndst[j * 8 + 5]);
                tdst[j * 4 + 3] = __umulhi(alpha_u32, ndst[j * 8 + 7]);
            } else {
                ndst[j * 4 + 0] = sh_norm[j * 16 + (lid % 4) * 2 + 0];
                ndst[j * 4 + 1] = sh_norm[j * 16 + (lid % 4) * 2 + 1];
                ndst[j * 4 + 2] = sh_norm[j * 16 + (lid % 4) * 2 + 8];
                ndst[j * 4 + 3] = sh_norm[j * 16 + (lid % 4) * 2 + 9];
                tdst[j * 4 + 0] = __umulhi(alpha_u32, ndst[j * 4 + 0]);
                tdst[j * 4 + 1] = __umulhi(alpha_u32, ndst[j * 4 + 1]);
                tdst[j * 4 + 2] = __umulhi(alpha_u32, ndst[j * 4 + 2]);
                tdst[j * 4 + 3] = __umulhi(alpha_u32, ndst[j * 4 + 3]);
            }
        } else {
            constexpr int b = layer == 2 ? 2 : 1;
            int n0 = sh_norm[(j * 16 + (lid % 4) * 2 + 0) * b + b / 2];
            int n1 = sh_norm[(j * 16 + (lid % 4) * 2 + 1) * b + b / 2];
            int n2 = sh_norm[(j * 16 + (lid % 4) * 2 + 8) * b + b / 2];
            int n3 = sh_norm[(j * 16 + (lid % 4) * 2 + 9) * b + b / 2];
            tdst[j * 4 + 0] = __umulhi(alpha_u32, n0);
            tdst[j * 4 + 1] = __umulhi(alpha_u32, n1);
            tdst[j * 4 + 2] = __umulhi(alpha_u32, n2);
            tdst[j * 4 + 3] = __umulhi(alpha_u32, n3);
        }
    }
}
template <int32_t layer>
static __device__ __forceinline__ void _pfched_unpk(void *_wdst, void *_wsrc, uint32_t CSD, uint32_t lid, uint32_t CSD16) {
    int *wdst = (int *)_wdst;
    int *wsrc = (int *)_wsrc;
    for (int i = 0; i < CSD16 / 16; i++) {
        int job = i * 32 + lid;
        int vid = job / (CSD16 / 4);
        int pos = job % (CSD16 / 4);
        int msk = 0xffffffff >> max(0, 32 * pos - 8 * (int)CSD + 32);
        int val0 = wsrc[(CSD * vid + 4 * pos) / 4 + 0];
        int val1 = wsrc[(CSD * vid + 4 * pos) / 4 + 1];
        wdst[job] = __funnelshift_r(val0, val1, 8 * (CSD * vid)) & msk;
    }
}
template <int32_t layer>
static __device__ __forceinline__ void _rbuc_2_sbuc(int *warp_sbuc_num, uint32_t *warp_sbuc, uint32_t *rbuc, 
                                                    int32_t *sh_norm, int ind, uint32_t lid, uint32_t batchsize) {        
    if (batchsize >= blockThreads) {
        for (int c = 0; c < C_batch / 2; c++) {
            if (!L12_USE_IF) {
                if (layer) {
                    for (int i = 0; i < 8; i++) {
                        uint32_t e = rbuc[c * 16 + i * 2 + 0];
                        if ((e >> 30) == 0) {
                            if (layer == 2) {
                                rbuc[c * 16 + i * 2 + 0] = sh_norm[2 * e + (lid % 4) * 4 + 0] + 0x40000000;
                                rbuc[c * 16 + i * 2 + 1] = sh_norm[2 * e + (lid % 4) * 4 + 1];
                            } else {
                                rbuc[c * 16 + i * 2 + 0] += ind + 0x40000000;
                                rbuc[c * 16 + i * 2 + 1] = sh_norm[e + (lid % 4) * 2];
                            }
                        }
                    }
                }
            }

            uint32_t b_size = warp_sbuc_num[c * 8 + lid / 4];
            uint32_t db;
            if (layer == 0) db = ((1 - (rbuc[c * 8 + 0] >> 31)) << 0)  |
                                    ((1 - (rbuc[c * 8 + 2] >> 31)) << 8)  |
                                    ((1 - (rbuc[c * 8 + 4] >> 31)) << 16) |
                                    ((1 - (rbuc[c * 8 + 6] >> 31)) << 24);
            else db = ((1 - (rbuc[c * 16 +  0] >> 31)) << 1)  |
                        ((1 - (rbuc[c * 16 +  4] >> 31)) << 9)  |
                        ((1 - (rbuc[c * 16 +  8] >> 31)) << 17) |
                        ((1 - (rbuc[c * 16 + 12] >> 31)) << 25);
            uint32_t rpos = db;
            uint32_t tmp = __shfl_up_sync(0xffffffff, rpos, 1, 4);
            if (lid & 3) rpos += tmp;
            tmp = __shfl_up_sync(0xffffffff, rpos, 2, 4);
            if (lid & 2) rpos += tmp;
            rpos += b_size;
            
            if ((lid & 3) == 3) warp_sbuc_num[c * 8 + lid / 4] = rpos & 0x7f7f7f7f;

            rpos -= db;

            for (int i = 0; i < 4; i++) {
                int num = db & (0xff << (i * 8));
                int pos = (rpos >> (i * 8)) & 0xff;
                if (layer == 0) {   
                    if (num && pos < sbucSize) 
                        warp_sbuc[(c * 32 + i * 8 + lid / 4) * sbucSize + pos] = rbuc[c * 8 + 2 * i];
                    rbuc[c * 8 + 2 * i + 0] = rbuc[c * 8 + 2 * i + 1];
                    rbuc[c * 8 + 2 * i + 1] = 0xffffffff;
                } else {
                    if (num && pos + 1 < sbucSize) {
                        warp_sbuc[(c * 32 + i * 8 + lid / 4) * sbucSize + pos + 0] = 
                                (layer == 1 ? (lid & 3) * 2 : 0) + rbuc[c * 16 + 4 * i + 0];
                        warp_sbuc[(c * 32 + i * 8 + lid / 4) * sbucSize + pos + 1] = rbuc[c * 16 + 4 * i + 1];
                    }
                    rbuc[c * 16 + 4 * i + 0] = rbuc[c * 16 + 4 * i + 2];
                    rbuc[c * 16 + 4 * i + 1] = rbuc[c * 16 + 4 * i + 3];
                    rbuc[c * 16 + 4 * i + 2] = 0xffffffff;
                    rbuc[c * 16 + 4 * i + 3] = 0xffffffff;
                }
            }
        }
    } else {
        if (!L12_USE_IF) {
            if (layer) {
                for (int i = 0; i < 4; i++) {
                    uint32_t e = rbuc[i * 2 + 0];
                    if ((e >> 30) == 0) {
                        if (layer == 2) {
                            rbuc[i * 2 + 0] = sh_norm[2 * e + (lid % 4) * 4 + 0] + 0x40000000;
                            rbuc[i * 2 + 1] = sh_norm[2 * e + (lid % 4) * 4 + 1];
                        } else {
                            rbuc[i * 2 + 0] += ind + 0x40000000;
                            rbuc[i * 2 + 1] = sh_norm[e + (lid % 4) * 2];
                        }
                    }
                }
            }
        }

        uint16_t b_size = ((uint16_t *)warp_sbuc_num)[lid / 4];
        uint16_t db;
        if (layer == 0) {
            db = ((1 - (rbuc[0] >> 31)) << 0) | ((1 - (rbuc[2] >> 31)) << 8);
        } else {
            db = ((1 - (rbuc[0] >> 31)) << 1) | ((1 - (rbuc[4] >> 31)) << 9);
        }
        uint16_t rpos = db;
        uint16_t tmp = __shfl_up_sync(0xffffffff, rpos, 1, 4);
        if (lid & 3) rpos += tmp;
        tmp = __shfl_up_sync(0xffffffff, rpos, 2, 4);
        if (lid & 2) rpos += tmp;
        rpos += b_size;

        if ((lid & 3) == 3) ((uint16_t *)warp_sbuc_num)[lid / 4] = rpos & 0x7f7f;

        rpos -= db;

        for (int i = 0; i < 2; i++) {
            int num = db & (0xff << (i * 8));
            int pos = (rpos >> (i * 8)) & 0xff;
            if (layer == 0) {
                if (num && pos < sbucSize) 
                    warp_sbuc[(i * 8 + lid / 4) * sbucSize + pos] = rbuc[2 * i];
                rbuc[2 * i + 0] = rbuc[2 * i + 1];
                rbuc[2 * i + 1] = 0xffffffff;
            } else {
                if (num && pos + 1 < sbucSize) {
                    warp_sbuc[(i * 8 + lid / 4) * sbucSize + pos + 0] = 
                        (layer == 1 ? (lid & 3) * 2 : 0) + rbuc[4 * i + 0];
                    warp_sbuc[(i * 8 + lid / 4) * sbucSize + pos + 1] = rbuc[4 * i + 1];
                }
                rbuc[4 * i + 0] = rbuc[4 * i + 2];
                rbuc[4 * i + 1] = rbuc[4 * i + 3];
                rbuc[4 * i + 2] = 0xffffffff;
                rbuc[4 * i + 3] = 0xffffffff;
            }
        }
    }
}
template <int32_t layer>
static __device__ __forceinline__ void _sbuc_2_gbuc(uint32_t *bgbuc_num, uint32_t *bgbuc, int out_max_size, int *sbuc_num, 
                                                        uint32_t *sbuc, uint32_t wid, uint32_t lid, uint32_t batchsize) {
    int *sbuc_tmp = (int *)sbuc + sbucSize * blockCenters;
    
    if (batchsize >= blockThreads) {
        for (int c = 0; c < C_batch / 2; c++) {
            int bias = wid * C_batch * 4 + c * 8;
            int32_t b_size = min((sbuc_num[bias + lid / 4] >> ((lid & 3) * 8)) & 0xff, sbucSize);
            if (layer != 0) b_size /= 2;
            int pos = atomicAdd(&bgbuc_num[bias * 4 + (lid & 3) * 8 + lid / 4], b_size);
            b_size = max(min(out_max_size - pos, b_size), 0);
            sbuc_tmp[bias * 4 + (lid & 3) * 8 + lid / 4] = pos | (b_size << 26);
            if ((lid & 3) == 0) sbuc_num[bias + lid / 4] = 0;
        }
        __syncwarp();
        for (int i = wid * C_batch * 16; i < (wid + 1) * C_batch * 16; i++) {
            int pos = sbuc_tmp[i];
            int num = pos >> 26;
            pos &= 0x03ffffff;
            if (layer == 0) {
                if ((int)lid < num) bgbuc[i * out_max_size + pos + lid] = sbuc[i * sbucSize + lid];
            } else {
                if ((int)lid < num * 2) bgbuc[i * 2 * out_max_size + 2 * pos + lid] = (layer && !L12_USE_IF) ? 
                                    (sbuc[i * sbucSize + lid] & 0xbfffffff) : sbuc[i * sbucSize + lid];
            }
        }
    } else {
        int bias = wid * 16;
        if (lid < 16) {
            int32_t b_size = min(((uint8_t *)sbuc_num)[bias + lid], sbucSize);
            if (layer != 0) b_size /= 2;
            int pos = atomicAdd(&bgbuc_num[bias + (lid & 1) * 8 + lid / 2], b_size);
            b_size = max(min(out_max_size - pos, b_size), 0);
            sbuc_tmp[bias + (lid & 1) * 8 + lid / 2] = pos | (b_size << 26);
        }
        if (lid < 4) sbuc_num[bias / 4 + lid] = 0;
        __syncwarp();
        for (int i = 0; i < 16; i++) {
            int pos = sbuc_tmp[bias + i];
            int num = pos >> 26;
            pos &= 0x03ffffff;
            if (layer == 0) {
                if ((int)lid < num) bgbuc[(bias + i) * out_max_size + pos + lid] = sbuc[(bias + i) * sbucSize + lid];
            } else {
                if ((int)lid < num * 2) bgbuc[(bias + i) * 2 * out_max_size + 2 * pos + lid] = (layer && !L12_USE_IF) ? 
                                    (sbuc[(bias + i) * sbucSize + lid] & 0xbfffffff) : sbuc[(bias + i) * sbucSize + lid];
            }
        }
    }
}
static __device__ __forceinline__ void _store_pad16(int8_t *wdst, int8_t *wsrc, uint32_t lid, uint32_t CSD16) {
    int int4_per_warp = batchVecs * CSD16 / blockWarps / sizeof(int4);
    if (lid < int4_per_warp) ((int4 *)wdst)[lid] = ((int4 *)wsrc)[lid];
    if (lid + 32 < int4_per_warp) ((int4 *)wdst)[lid + 32] = ((int4 *)wsrc)[lid + 32];
    if (lid + 64 < int4_per_warp) ((int4 *)wdst)[lid + 64] = ((int4 *)wsrc)[lid + 64];
}
static __device__ __forceinline__ void _l2_vid_pfch(int *dst, int *src, uint32_t wid, uint32_t lid, uint32_t batchsize) {
    int int2_per_warp = batchVecs / blockWarps;

    if (lid < int2_per_warp) {
        utils_t::_ldgsts_64b_async(dst + wid * int2_per_warp * 2 + lid * 2, src + wid * int2_per_warp * 2 + lid * 2);
    }

    utils_t::_commit_async_group();
}
static __device__ __forceinline__ void _l2_vec_pfch(int8_t *vdst, int8_t *vec16, int32_t *sh_norm, uint32_t wid, 
                                                    uint32_t lid, uint32_t batchsize, uint32_t CSD16) {
    int bias = wid * (batchVecs / blockWarps) + (lid & 16) / 4;
    int  *lnid  = sh_norm + bias * 2;
    int4 *lvdst = (int4 *)(vdst + CSD16 * bias);
    
    int r[4];
    for (int i = 0; i < 4; i++) r[i] = lnid[2 * i];

    for (int i = 0; i < 4; i++) {
        if ((lid & 15) < CSD16 / 16) {
            utils_t::_ldgsts_128b_async(lvdst + i * CSD16 / 16 + (lid & 15), vec16 + r[i] * (uint64_t) CSD16 + (lid & 15) * 16);
        }
    }
    utils_t::_commit_async_group();
}


#undef blockThreads
#undef blockWarps
#undef blockCenters
#undef unit
#undef sbucSize
#undef batchVecs
#undef C_batch
#undef L12_USE_IF

template <int32_t layer, int32_t batchsize, int32_t CSD16>
__global__ void _bucket_kernel(uint32_t *__restrict__ out, int out_max_size, int8_t *__restrict__ vec_pad16,
                                int32_t *__restrict__ vnorm, const int8_t *__restrict__ center, uint32_t in_max_size,
                                const int8_t *__restrict__ vec, int *n_ptr, float alpha, int CSD, int gbuc_freq, int ind_bias) {
    typedef buc_traits_t traits;
    typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> wma_t;
    typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> wmb_t;
    typedef wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> wacc_t;

    uint32_t n = *n_ptr;
    if (n > in_max_size && layer == 2) n = in_max_size;

    #if RECORD_CLK
    uint32_t start_clk = utils_t::_clock() >> 32;
    #endif

    /// constants
    constexpr unsigned int vec_nbytes   = traits::vec_nbytes;
    constexpr unsigned int blockCenters = traits::buc_unit < batchsize ? traits::buc_unit : batchsize;
    constexpr unsigned int blockTypes   = batchsize / blockCenters;
    constexpr unsigned int batchVecs    = traits::buc_batchVecs;
    constexpr unsigned int sbucSize     = layer == 0 ? traits::l0_sbucSize :
                                          layer == 1 ? traits::l1_sbucSize :
                                                       traits::l2_sbucSize;
    constexpr unsigned int C_batch      = (blockCenters * 2 / traits::blockThreads) > 0 ? 
                                          (blockCenters * 2 / traits::blockThreads) : 1;
    constexpr unsigned int V_batch      = batchVecs / 16;
    constexpr unsigned int use_if       = layer == 1 ? traits::l1_use_if : traits::l2_use_if;

    /// registers
    int32_t thread_th[V_batch * 4];
    int32_t thread_norm[V_batch * 4 * ((layer == 2 && use_if) ? 2 : 1)];
    wma_t center_frag[C_batch][vec_nbytes / 16];
    uint32_t rbuc[C_batch * 2 * 2 * (layer ? 2 : 1)];

    /// memory
    extern __shared__ int8_t dy_buf[];
    constexpr unsigned int sh_norm_nbytes = 4 * batchVecs * (layer == 2 ? 6 : (layer == 1 && !use_if) ? 3 : 2);
    int8_t   *unpack    = (int8_t *)   &dy_buf[0];
    int8_t   *sh_vec    = (int8_t *)   &dy_buf[layer == 2 ? 0 : vec_nbytes * batchVecs];
    int32_t  *sh_norm   = (int32_t *)  &sh_vec[vec_nbytes * batchVecs * 2];
    int32_t  *sbuc_num  = (int32_t *)  &sh_vec[vec_nbytes * batchVecs * 2 + sh_norm_nbytes];
    uint32_t *sbuc      = (uint32_t *) &sh_vec[vec_nbytes * batchVecs * 2 + sh_norm_nbytes + traits::buc_unit];

    uint32_t *bgbuc_num = (uint32_t *) &out[(blockIdx.x % blockTypes) * blockCenters];
    uint32_t *bgbuc     = (uint32_t *) &out[batchsize + (blockIdx.x % blockTypes) * blockCenters * out_max_size * (layer ? 2 : 1)];

    /// const variables
    const unsigned int tid = utils_t::_thread_id();
    const unsigned int lid = utils_t::_lane_id();
    const unsigned int wid = utils_t::_thread_id() / 32;
    const unsigned int alpha_u32 = __float2uint_rn(2.0f * alpha * 0x1p32f);
    
    /// for loop control
    uint32_t unacc_iters   = 0;
    uint32_t ind           = (blockIdx.x / blockTypes) * batchVecs;
    const uint32_t stride  = ((gridDim.x + blockTypes - 1 - (blockIdx.x % blockTypes)) / blockTypes) * batchVecs;
    constexpr uint32_t pfch_vec_xor = batchVecs * vec_nbytes;
    constexpr uint32_t pfch_norm_xor = batchVecs * (layer == 2 ? 2 : 1);
    uint32_t pfch_vec_bias = pfch_vec_xor;
    uint32_t pfch_norm_bias = pfch_norm_xor * (layer == 2 ? 2 : 1);
    #define curr_vec_bias (pfch_vec_bias ^ pfch_vec_xor)
    #define curr_norm_bias (layer + (use_if ^ 1) > 1 ? ((pfch_norm_bias + 2 * pfch_norm_xor) % (3 * pfch_norm_xor)) : (pfch_norm_bias ^ pfch_norm_xor))
    #define last_norm_bias (layer + (use_if ^ 1) > 1 ? ((pfch_norm_bias + 1 * pfch_norm_xor) % (3 * pfch_norm_xor)) : (pfch_norm_bias ^ pfch_norm_xor))

    /// tail mask
    _tail_mask<layer>(layer == 2 ? (int *) vec : vnorm, n, tid);
    
    /// for global vec loading
    const unsigned int int2_per_warp = CSD * batchVecs / 8 / traits::ldWarps;
    int2 *wgvld_src = (int2 *)(&vec[ind * CSD]) + wid * int2_per_warp;
    int2 *wgvld_dst = UNPACK_INCLUDED ? (int2 *)unpack + wid * (vec_nbytes * batchVecs / 8 / traits::ldWarps) :
                                        (int2 *)sh_vec + wid * (vec_nbytes * batchVecs / 8 / traits::ldWarps);
    int *gnidld_src = ((int *)vec) + ind * 2;

    /// load center vectors
    const int8_t *warp_center = center + ((blockIdx.x % blockTypes) * blockCenters + (C_batch * wid) * 16) * CSD16;
    _center_load<layer>(center_frag, warp_center, wid, lid, batchsize, CSD16);

    /// clear buckets
    _buffer_init<layer>(sbuc_num, sh_vec, tid, batchsize);
    
    /// for the first iter
    if (layer == 2) {
        _l2_vid_pfch(sh_norm, gnidld_src, wid, lid, batchsize);
        gnidld_src += 2 * stride;
    } else if (wid < traits::ldWarps) {
        _nextnv_pfch<layer>(sh_norm, vnorm + ind, wgvld_dst, wgvld_src, int2_per_warp, lid, tid);
    }
    wgvld_src += CSD * stride / 8;
    for (int i = 0; i < sizeof(rbuc) / 4; i++) rbuc[i] = 0xffffffff;
    utils_t::_wait_async_group();
    __syncwarp();
    if (layer == 2) {
        _l2_vid_pfch(sh_norm + pfch_norm_xor, gnidld_src, wid, lid, batchsize);
        gnidld_src += 2 * stride;
        _l2_vec_pfch(sh_vec, vec_pad16, sh_norm, wid, lid, batchsize, CSD16);
        utils_t::_wait_async_group();
    } else if (wid < traits::ldWarps) {
        #if UNPACK_INCLUDED
        _pfched_unpk<layer>(sh_vec + CSD16 * (batchVecs / traits::ldWarps) * wid, wgvld_dst, CSD, lid, CSD16);
        #endif
    }
    __syncthreads();

    _normth_prep<layer>(thread_norm, thread_th, sh_norm + (layer == 1 ? curr_norm_bias : last_norm_bias), alpha_u32, lid);

    while (ind < n) {
        if (layer == 2) {
            if (ind + stride * 2 < n) {
                _l2_vid_pfch(sh_norm + pfch_norm_bias, gnidld_src, wid, lid, batchsize);
            } 
            if (ind + stride < n) {
                _l2_vec_pfch(sh_vec + pfch_vec_bias, vec_pad16, sh_norm + curr_norm_bias, 
                                    wid, lid, batchsize, CSD16);
            }                
        } else if (ind + stride < n && wid < traits::ldWarps) {
            #if UNPACK_INCLUDED
            _nextnv_pfch<layer>(sh_norm + pfch_norm_bias, vnorm + ind + stride, 
                                        wgvld_dst, wgvld_src, int2_per_warp, lid, tid);
            #else
            _nextnv_pfch<layer>(sh_norm + pfch_norm_bias, vnorm + ind + stride, 
                                    wgvld_dst + pfch_vec_bias / 8, wgvld_src, int2_per_warp, lid, tid);
            #endif
        }

        for (int j = 0; j < V_batch; j++) {
            wacc_t dp_frag[C_batch];
            wmb_t vec_frag[CSD16 / 16];
            for (int l = 0; l < CSD16; l += 16) {
                wmma::load_matrix_sync(vec_frag[l / 16], sh_vec + curr_vec_bias + CSD16 * 16 * j + l, CSD16);
            }
            for (int i = 0; i < C_batch; i++) {
                wmma::fill_fragment(dp_frag[i], 0);
                for (int l = 0; l < CSD16; l += 16) {
                    wmma::mma_sync(dp_frag[i], center_frag[i][l / 16], vec_frag[l / 16], dp_frag[i]);
                }
                for (int k = 0; k < 8; k++) {
                    if (layer == 0) {
                        uint32_t cmp = ((abs(dp_frag[i].x[k]) - thread_th[j * 4 + (k & 1) + (k & 4) / 2])) & 0x80000000;
                        uint32_t entry = ((uint32_t)dp_frag[i].x[k] >> 31) + 2 * (ind + j * 16 + (lid & 3) * 2 + (k & 1) + (k & 4) * 2);
                        int32_t val = entry | cmp;
                        int32_t tmp = min((int)rbuc[i * 2 * 2 + (k & 2) + 0], val);
                        rbuc[i * 2 * 2 + (k & 2) + 0] = max((int)rbuc[i * 2 * 2 + (k & 2) + 0], val);
                        rbuc[i * 2 * 2 + (k & 2) + 1] = max((int)rbuc[i * 2 * 2 + (k & 2) + 1], tmp);
                    } else {
                        if (use_if) {
                            if (dp_frag[i].x[k] > thread_th[j * 4 + (k & 1) + (k & 4) / 2]) {
                                uint32_t entry, norm;
                                if (layer == 1) {
                                    entry = ind + j * 16 + (k & 1) + (k & 4) * 2;
                                    norm = thread_norm[j * 4 + (k & 1) + (k & 4) / 2];
                                } else {
                                    entry = thread_norm[j * 8 + (k & 1) * 2 + (k & 4) + 0];
                                    norm  = thread_norm[j * 8 + (k & 1) * 2 + (k & 4) + 1];
                                }
                                if (rbuc[(i * 2 + (k & 2) / 2) * 4 + 0] == 0xffffffff) {
                                    rbuc[(i * 2 + (k & 2) / 2) * 4 + 0] = entry;
                                    rbuc[(i * 2 + (k & 2) / 2) * 4 + 1] = norm;
                                } else {
                                    rbuc[(i * 2 + (k & 2) / 2) * 4 + 2] = entry;
                                    rbuc[(i * 2 + (k & 2) / 2) * 4 + 3] = norm;
                                }
                            }
                        } else {
                            uint32_t cmp = (dp_frag[i].x[k] - thread_th[j * 4 + (k & 1) + (k & 4) / 2]) & 0x80000000;
                            uint32_t entry = j * 16 + (k & 1) + (k & 4) * 2;
                            int32_t val = entry | cmp;
                            int32_t tmp = min((int)rbuc[(i * 2 + (k & 2) / 2) * 4 + 0], val);
                            rbuc[(i * 2 + (k & 2) / 2) * 4 + 0] = max((int)rbuc[(i * 2 + (k & 2) / 2) * 4 + 0], val);
                            rbuc[(i * 2 + (k & 2) / 2) * 4 + 2] = max((int)rbuc[(i * 2 + (k & 2) / 2) * 4 + 2], tmp);
                        }
                    }
                }
            }
            if (j == 1) {
                utils_t::_wait_async_group();
                if (layer != 2) {
                    #if UNPACK_INCLUDED
                    __syncwarp();
                    if (wid < traits::ldWarps) _pfched_unpk<layer>(sh_vec + CSD16 * (batchVecs / traits::ldWarps) * 
                                                                        wid + pfch_vec_bias, wgvld_dst, CSD, lid, CSD16);
                    #endif
                }
                __syncthreads();
            }
        }

        _normth_prep<layer>(thread_norm, thread_th, sh_norm + (layer == 2 ? curr_norm_bias : pfch_norm_bias), alpha_u32, lid);

        if (wid * 16 < batchsize || batchsize >= traits::blockThreads)
            _rbuc_2_sbuc<layer>(&sbuc_num[wid * C_batch * 4], &sbuc[sbucSize * (wid * C_batch * 16)], rbuc, 
                                        sh_norm + (layer == 1 ? curr_norm_bias : last_norm_bias), 
                                        ind + (layer == 1 ? ind_bias : 0), lid, batchsize);
        __syncwarp();

        if (unacc_iters++ > gbuc_freq || ind + stride >= n) {
            unacc_iters = 0;
            if (wid * 16 < batchsize || batchsize >= traits::blockThreads)
                _sbuc_2_gbuc<layer>(bgbuc_num, bgbuc, out_max_size, sbuc_num, sbuc, wid, lid, batchsize);
        }

        if (layer == 1) _store_pad16(vec_pad16 + CSD16 * (ind + batchVecs / traits::ldWarps * wid),
                        sh_vec + CSD16 * (batchVecs / traits::ldWarps) * wid + curr_vec_bias, lid, CSD16);

        ind            += stride;
        wgvld_src      += CSD * stride / 8;
        pfch_vec_bias  ^= pfch_vec_xor;
        pfch_norm_bias  = last_norm_bias;
        gnidld_src     += 2 * stride;
    }

    #if RECORD_CLK
    uint32_t end_clk = utils_t::_clock() >> 32;
    if (end_clk - start_clk == 0xffffffff) {
        ((uint64_t *)&out[batchsize * (2 * out_max_size + 1)])[blockIdx.x * blockDim.x + threadIdx.x] = start_clk;
        ((uint64_t *)&out[batchsize * (2 * out_max_size + 1)])[blockIdx.x * blockDim.x + threadIdx.x + traits::blockThreads * traits::kernelBlocks] = end_clk;
    }
    #endif

    #undef curr_vec_bias
    #undef curr_norm_bias
    #undef curr_nid_bias
    #undef last_nid_bias
}

#define INSTANTIATE(layer, batchsize, CSD16) \
    template __global__ void _bucket_kernel<layer, batchsize, CSD16>(uint32_t *__restrict__ out, int out_max_size, int8_t *__restrict__ vec_pad16, \
                                                                    int32_t *__restrict__ vnorm, const int8_t *__restrict__ center, uint32_t in_max_size, \
                                                                    const int8_t *__restrict__ vec, int *n_ptr, float alpha, int CSD, int gbuc_freq, int ind_bias)

INSTANTIATE(0, 16, 176);
INSTANTIATE(0, 32, 176);
INSTANTIATE(0, 64, 176);
INSTANTIATE(0, 128, 176);
INSTANTIATE(0, 256, 176);
INSTANTIATE(0, 512, 176);
INSTANTIATE(0, 1024, 176);
INSTANTIATE(0, 2048, 176);
INSTANTIATE(2, 64, 176);
INSTANTIATE(1, 64, 176);
INSTANTIATE(1, 128, 176);
INSTANTIATE(1, 256, 176);
INSTANTIATE(1, 512, 176);
INSTANTIATE(1, 1024, 176);
INSTANTIATE(1, 2048, 176);
INSTANTIATE(1, 4096, 176);
INSTANTIATE(2, 128, 176);
INSTANTIATE(2, 256, 176);
INSTANTIATE(2, 512, 176);
INSTANTIATE(2, 1024, 176);
INSTANTIATE(2, 2048, 176);
INSTANTIATE(2, 16, 176);
INSTANTIATE(2, 32, 176);

#if BUC_MIN_CSD16 < 176
INSTANTIATE(0, 16, 160);
INSTANTIATE(0, 32, 160);
INSTANTIATE(0, 64, 160);
INSTANTIATE(0, 128, 160);
INSTANTIATE(0, 256, 160);
INSTANTIATE(0, 512, 160);
INSTANTIATE(0, 1024, 160);
INSTANTIATE(0, 2048, 160);
INSTANTIATE(1, 64, 160);
INSTANTIATE(1, 128, 160);
INSTANTIATE(1, 256, 160);
INSTANTIATE(1, 512, 160);
INSTANTIATE(1, 1024, 160);
INSTANTIATE(1, 2048, 160);
INSTANTIATE(1, 4096, 160);
INSTANTIATE(2, 64, 160);
INSTANTIATE(2, 128, 160);
INSTANTIATE(2, 256, 160);
INSTANTIATE(2, 512, 160);
INSTANTIATE(2, 1024, 160);
INSTANTIATE(2, 2048, 160);
INSTANTIATE(2, 16, 160);
INSTANTIATE(2, 32, 160);
#endif
#if BUC_MIN_CSD16 < 160
INSTANTIATE(0, 16, 144);
INSTANTIATE(0, 32, 144);
INSTANTIATE(0, 64, 144);
INSTANTIATE(0, 128, 144);
INSTANTIATE(0, 256, 144);
INSTANTIATE(0, 512, 144);
INSTANTIATE(0, 1024, 144);
INSTANTIATE(0, 2048, 144);
INSTANTIATE(1, 64, 144);
INSTANTIATE(1, 128, 144);
INSTANTIATE(1, 256, 144);
INSTANTIATE(1, 512, 144);
INSTANTIATE(1, 1024, 144);
INSTANTIATE(1, 2048, 144);
INSTANTIATE(1, 4096, 144);
INSTANTIATE(2, 64, 144);
INSTANTIATE(2, 128, 144);
INSTANTIATE(2, 256, 144);
INSTANTIATE(2, 512, 144);
INSTANTIATE(2, 1024, 144);
INSTANTIATE(2, 2048, 144);
INSTANTIATE(2, 16, 144);
INSTANTIATE(2, 32, 144);
#endif
#if BUC_MIN_CSD16 < 144
INSTANTIATE(0, 16, 128);
INSTANTIATE(0, 32, 128);
INSTANTIATE(0, 64, 128);
INSTANTIATE(0, 128, 128);
INSTANTIATE(0, 256, 128);
INSTANTIATE(0, 512, 128);
INSTANTIATE(0, 1024, 128);
INSTANTIATE(0, 2048, 128);
INSTANTIATE(1, 64, 128);
INSTANTIATE(1, 128, 128);
INSTANTIATE(1, 256, 128);
INSTANTIATE(1, 512, 128);
INSTANTIATE(1, 1024, 128);
INSTANTIATE(1, 2048, 128);
INSTANTIATE(1, 4096, 128);
INSTANTIATE(2, 64, 128);
INSTANTIATE(2, 128, 128);
INSTANTIATE(2, 256, 128);
INSTANTIATE(2, 512, 128);
INSTANTIATE(2, 1024, 128);
INSTANTIATE(2, 2048, 128);
INSTANTIATE(2, 16, 128);
INSTANTIATE(2, 32, 128);
#endif
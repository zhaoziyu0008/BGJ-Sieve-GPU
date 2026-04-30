#include "../include/config.h"
#include "../include/common_device.h"
#include "../include/pool_hd_device.h"


template <uint32_t CSD16, uint32_t ESD8>
__global__ void check_kernel(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data) {
    typedef check_traits traits;
    pdev_kernel_var_declare();

    /// prepare data ///
    traits::_kernel_init(b_dual_tri, b_ext_head, uid_coeff, inorm, igh, local_data, ESD8, tid);
    
    /// for the first ind ///
    traits::_prep_vec(vec, glob_vec, tid);
    traits::_prep_bfull(lane_bfull, lane_bfull_glob, ESD8 + CSD16, tid);
    traits::_next_coeff(warp_coeff, warp_vec, b_dual_tri, dhalf, dshift, u_acc, uid_coeff, CSD16, lid);

    while (ind < n) {
        float fp32_frag0[64];   // 64R
        float fp32_frag1[64];   // 64R

        traits::_batch_init(fp32_frag0, fp32_frag1, sbuf, ESD8 + CSD16, wid, lid);
        
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
                traits::_prep_bfull(lane_bfull + f_bias, lane_bfull_glob + bf_bias, ESD8 + rem, tid);
                traits::_prep_coeff(warp_coeff + c_bias, warp_vec, b_dual_tri + bd_bias, 
                                          dhalf, dshift, rem, u_acc, uid_coeff, CSD16, lid);
            } else if (ind + stride < n) traits::_prep_vec(vec, glob_vec + stride * vec_nbytes, tid);

            if (wid < ESD8 / 16) {
                float *A_8x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + (tid >> 0x4) * 132;
                traits::_fmma8x16x8(fp32_frag0, A_8x16, B_8x16);
            }
            if (wid == ESD8 / 16 && (ESD8 & 0xf)) {
                float *A_4x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132 + (tid & 0x10) / 4;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + wid * 2 * 132;
                traits::_fmma4x16x8(fp32_frag0, A_4x16, B_8x16);
            }
            if (wid >= (ESD8 + 8) / 16 && wid <= (ESD8 + 8) / 16 + rem / 16) {
                float *A_8x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + (tid >> 0x4) * 132 - ((ESD8 & 0x8) / 8) * 132;
                traits::_fmma8x16x8(fp32_frag0, A_8x16, B_8x16);
            }
            if (wid + 8 <= (ESD8 + 8) / 16 + rem / 16) {
                float *A_8x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + (tid >> 0x4) * 132 + 16 * 132 - ((ESD8 & 0x8) / 8) * 132;
                traits::_fmma8x16x8(fp32_frag1, A_8x16, B_8x16);
            }
        }

        __syncthreads();
        if (ind + stride < n) traits::_prep_bfull(lane_bfull, lane_bfull_glob, ESD8 + CSD16, tid);
        
        float n_acc[8];
        int2 vp_line[16];
        uint32_t smsk_acc = 0x00;
        traits::_vec_stage0(vp_line, n_acc, smsk_acc, fp32_frag0, fp32_frag1, ESD8, CSD16, wid);

        traits::_uid_reduce(u_acc, ubuf, wid, lid);

        float ext_frag[ESD8 / 2];
        if (ESD8) traits::_ext_stage0(fbuf + 28 * 132, ext_frag, fp32_frag0, ESD8, wid, tid);

        int sn;
        float s0, s1;
        traits::_vec_stage1(fbuf + 28 * 132, vp_line, n_acc, smsk_acc, ESD8, CSD16, wid, lid, tid);
        traits::_vec_stage2(fbuf + 28 * 132, nbuf, sbuf, sn, s0, s1, ESD8, wid, lid, tid);
        
        if (ESD8) traits::_ext_stage1(ext_frag, sn, s0, s1, b_ext_head, igh, inorm, ESD8, tid);

        if (ind + stride < n) traits::_next_coeff(warp_coeff, warp_vec, b_dual_tri, dhalf,  
                                                        dshift, u_acc, uid_coeff, CSD16, lid);

        traits::_write_back(data, ind, (int *)vp_line, ubuf, nbuf, sbuf, sn, wid, lid, tid);

        ind += stride;
        glob_vec += stride * vec_nbytes;
    }
}

template <uint32_t CSD16, uint32_t ESD8>
__global__ void extend_left_kernel(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data) {
    typedef extend_left_traits traits;
    pdev_kernel_var_declare();

    __shared__ int c0[128];
    __shared__ float e0[48];

    /// prepare data ///
    traits::_kernel_init(b_dual_tri, b_ext_head, uid_coeff, inorm, igh, local_data, ESD8, tid);
    
    /// for the first ind ///
    traits::_prep_vec(vec, glob_vec, tid);
    traits::_prep_bfull(lane_bfull, lane_bfull_glob, ESD8 + CSD16, tid);
    traits::_next_coeff(warp_coeff, warp_vec, b_dual_tri, dhalf, dshift, u_acc, uid_coeff, CSD16, lid);

    const float b00 = local_data->b_full[ESD8 * CSD16 + (CSD16 / 2 + 8) * CSD16 + 16 * (CSD16 - 16) + ESD8];
    const float ib00 = __frcp_rn(b00);
    if (tid < ESD8) e0[tid] = local_data->b_full[ESD8 * CSD16 + (CSD16 / 2 + 8) * CSD16 + 16 * (CSD16 - 16) + tid];

    while (ind < n) {
        float fp32_frag0[64];   // 64R
        float fp32_frag1[64];   // 64R

        traits::_batch_init(fp32_frag0, fp32_frag1, sbuf, ESD8 + CSD16, wid, lid);
        
        #pragma unroll
        for (int rem = CSD16 - 16; rem >= 0; rem -= 16) {
            const int loop_size = (CSD16 - rem) >> 4;
            const int bd_bias = 256 / 2 * loop_size * (loop_size + 1);
            const int bf_bias = (CSD16 + ESD8 + 16 + 8 - 8 * loop_size) * loop_size * 16 - 256;
            const int c_bias = (loop_size & 0x1) * 16 * 132;
            const int f_bias = (loop_size & 0x1) * 28 * 132;

            if (rem != CSD16 - 16) {
                utils_t::_wait_async_group();
                __syncthreads();
            }

            if (rem) {
                traits::_prep_bfull(lane_bfull + f_bias, lane_bfull_glob + bf_bias, ESD8 + rem + 16, tid);
                traits::_prep_coeff(warp_coeff + c_bias, warp_vec, b_dual_tri + bd_bias, 
                                          dhalf, dshift, rem, u_acc, uid_coeff, CSD16, lid);
            } else if (ind + stride < n) traits::_prep_vec(vec, glob_vec + stride * vec_nbytes, tid);

            if (wid < ESD8 / 16) {
                float *A_8x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + (tid >> 0x4) * 132;
                traits::_fmma8x16x8(fp32_frag0, A_8x16, B_8x16);
            }
            if (wid == ESD8 / 16 && (ESD8 & 0xf)) {
                float *A_4x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132 + (tid & 0x10) / 4;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + wid * 2 * 132;
                traits::_fmma4x16x8(fp32_frag0, A_4x16, B_8x16);
            }
            if (wid >= (ESD8 + 8) / 16 && wid <= (ESD8 + 8) / 16 + rem / 16 + ((rem == CSD16 - 16) ? 0 : 1)) {
                float *A_8x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + (tid >> 0x4) * 132 - ((ESD8 & 0x8) / 8) * 132;
                traits::_fmma8x16x8(fp32_frag0, A_8x16, B_8x16);
            }
            if (wid + 8 <= (ESD8 + 8) / 16 + rem / 16 + ((rem == CSD16 - 16) ? 0 : 1)) {
                float *A_8x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + (tid >> 0x4) * 132 + 16 * 132 - ((ESD8 & 0x8) / 8) * 132;
                traits::_fmma8x16x8(fp32_frag1, A_8x16, B_8x16);
            }
        }

        if (wid == (ESD8 + 8) / 16 && lid < 16) {
            for (int i = 0; i < 8; i++) {
                float y = roundf(-fp32_frag0[8 * i] * ib00);
                fp32_frag0[8 * i] += y * b00;
                c0[8 * lid + i] = __float2int_rn(y);
            }
        }

        __syncthreads();
        if (ind + stride < n) traits::_prep_bfull(lane_bfull, lane_bfull_glob, ESD8 + CSD16, tid);
        
        float n_acc[8];
        int2 vp_line[16];
        uint32_t smsk_acc = 0x00;
        traits::_vec_stage0(vp_line, n_acc, smsk_acc, fp32_frag0, fp32_frag1, ESD8, CSD16, wid);

        if (wid < ESD8 / 16) {
            float c0_fp32[8], *e0_fp32 = e0 + 8 * (tid / 16);
            for (int i = 0; i < 8; i++) c0_fp32[i] = __int2float_rn(c0[8 * (tid & 15) + i]);
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) fp32_frag0[i * 8 + j] += c0_fp32[i] * e0_fp32[j];
            }
        }
        if (wid == ESD8 / 16 && (ESD8 & 0xf)) {
            float c0_fp32[4], *e0_fp32 = e0 + ESD8 - 8;
            for (int i = 0; i < 4; i++) c0_fp32[i] = __int2float_rn(c0[8 * (tid & 15) + (tid & 16) / 4 + i]);
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 8; j++) fp32_frag0[i * 8 + j] += c0_fp32[i] * e0_fp32[j];
            }
        }

        traits::_uid_reduce(u_acc, ubuf, wid, lid);

        float ext_frag[ESD8 / 2];
        if (ESD8) traits::_ext_stage0(fbuf + 28 * 132, ext_frag, fp32_frag0, ESD8, wid, tid);

        int sn;
        float s0, s1;
        traits::_vec_stage1(fbuf + 28 * 132, vp_line, n_acc, smsk_acc, ESD8, CSD16, wid, lid, tid);
        traits::_vec_stage2(fbuf + 28 * 132, nbuf, sbuf, sn, s0, s1, ESD8, wid, lid, tid);

        if (ESD8) traits::_ext_stage1(ext_frag, sn, s0, s1, b_ext_head, igh, inorm, ESD8, tid);

        if (tid < 128) ubuf[tid] += c0[tid] * uid_coeff[CSD16 - 1];
        __syncthreads();

        if (ind + stride < n) traits::_next_coeff(warp_coeff, warp_vec, b_dual_tri, dhalf,  
                                                        dshift, u_acc, uid_coeff, CSD16, lid);

        traits::_write_back(data, ind, (int *)vp_line, ubuf, nbuf, sbuf, sn, wid, lid, tid);

        ind += stride;
        glob_vec += stride * vec_nbytes;
    }
}

template <uint32_t CSD16, uint32_t ESD8>
__global__ void min_lift_kernel(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data) {
    typedef min_lift_traits traits;
    pdev_kernel_var_declare();

    (void) nbuf;
    (void) ubuf;
    const int force_one = local_data->uid_coeff[0];

    __shared__ float min_norm[48];
    float *min_norm_glob = (float *)(local_data + 1);
    int *min_lock = (int *)(min_norm_glob + ESD8);
    int *min_coeff = min_lock + ESD8;

    for (int i = n; i < ((n + 127) & ~127); i++) {
        if (tid < 11) ((int4 *)&data[i * vec_nbytes])[tid] = {0, 0, 0, 0};
    }
    if (tid < 48) min_norm[tid] = min_norm_glob[tid];
    if (n < 128) __syncthreads();

    /// prepare data ///
    traits::_kernel_init(b_dual_tri, b_ext_head, uid_coeff, inorm, igh, local_data, ESD8, tid);
    
    /// for the first ind ///
    traits::_prep_vec(vec, glob_vec, tid);
    traits::_prep_bfull(lane_bfull, lane_bfull_glob, ESD8 + CSD16, tid);
    traits::_next_coeff(warp_coeff, warp_vec, b_dual_tri, dhalf, dshift, u_acc, uid_coeff, CSD16, lid);

    while (ind < n) {
        float fp32_frag0[64];   // 64R
        float fp32_frag1[64];   // 64R

        traits::_batch_init(fp32_frag0, fp32_frag1, sbuf, ESD8 + CSD16, wid, lid);
        
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
                traits::_prep_bfull(lane_bfull + f_bias, lane_bfull_glob + bf_bias, ESD8 + rem, tid);
                traits::_prep_coeff(warp_coeff + c_bias, warp_vec, b_dual_tri + bd_bias, 
                                          dhalf, dshift, rem, u_acc, uid_coeff, CSD16, lid);
            } else if (ind + stride < n) traits::_prep_vec(vec, glob_vec + stride * vec_nbytes, tid);

            if (wid < ESD8 / 16) {
                float *A_8x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + (tid >> 0x4) * 132;
                traits::_fmma8x16x8(fp32_frag0, A_8x16, B_8x16);
            }
            if (wid == ESD8 / 16 && (ESD8 & 0xf)) {
                // 2 threads process 1 8x8 block together
                float *A_4x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132 + (tid & 0x10) / 4;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + wid * 2 * 132;
                traits::_fmma4x16x8(fp32_frag0, A_4x16, B_8x16);
            }
            if (wid >= (ESD8 + 8) / 16 && wid <= (ESD8 + 8) / 16 + rem / 16) {
                float *A_8x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + (tid >> 0x4) * 132 - ((ESD8 & 0x8) / 8) * 132;
                traits::_fmma8x16x8(fp32_frag0, A_8x16, B_8x16);
            }
            if (wid + 8 <= (ESD8 + 8) / 16 + rem / 16) {
                float *A_8x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + (tid >> 0x4) * 132 + 16 * 132 - ((ESD8 & 0x8) / 8) * 132;
                traits::_fmma8x16x8(fp32_frag1, A_8x16, B_8x16);
            }
        }

        __syncthreads();
        if (ind + stride < n) traits::_prep_bfull(lane_bfull, lane_bfull_glob, ESD8 + CSD16, tid);
        
        float n_acc[8];
        {
            for (int i = 0; i < 8; i++) n_acc[i] = 0.0f;
            if (wid >= (ESD8 + 8) / 16 && wid < (ESD8 + 8) / 16 + CSD16 / 16) {
                for (int j = 0; j < 8; j++) {
                    for (int i = 0; i < 8; i++) n_acc[i] += fp32_frag0[i * 8 + j] * fp32_frag0[i * 8 + j];
                }
            }
            if (wid + 8 < (ESD8 + 8) / 16 + CSD16 / 16) {
                for (int j = 0; j < 8; j++) {
                    for (int i = 0; i < 8; i++) n_acc[i] += fp32_frag1[i * 8 + j] * fp32_frag1[i * 8 + j];
                }
            }
        }
        

        float ext_frag[ESD8 / 2];
        if (ESD8) traits::_ext_stage0(fbuf + 28 * 132, ext_frag, fp32_frag0, ESD8, wid, tid);

        float n0, n1;
        {
            float *n_tmp = fbuf + 28 * 132;

            for (int i = 0; i < 8; i++) n_acc[i] += __shfl_xor_sync(0xffffffff, n_acc[i], 16);
            for (int i = 0; i < 8; i++) {
                if (lid < 16) n_tmp[(lid / 2) * (128 + 2) + (lid & 1) * 16 + (wid & 1) + 2 * i + (wid / 2) * 32] = n_acc[i];
            }
            __syncthreads();

            float n_fp = 0.0f;
            for (int i = 0; i < 128; i += 32) n_fp += n_tmp[wid * (128 + 2) + i + lid];
            n_fp += __shfl_xor_sync(0xffffffff, n_fp, 1);
            n0 = __shfl_sync(0xffffffff, n_fp, 0, 4);
            n1 = __shfl_sync(0xffffffff, n_fp, 2, 4);
        }
        

        int ext_coeff[ESD8];
        float *ext_norm = fbuf + 28 * 132 + 1280;
        int *updated = (int *)(ext_norm + (ESD8 + 1) * 128);
        if (tid == 0) *updated = 0;
        if (ESD8) {
            float *lane_ext_norm = ext_norm + ((lid / 4) * 2 + wid * 16) * (ESD8 + 1);
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
                    lane_ext_norm[0 * (ESD8 + 1) + rem + 1] = n0;
                    lane_ext_norm[1 * (ESD8 + 1) + rem + 1] = n1;
                    c0 = __float2int_rn(ext_frag[w_bias + 0] * inorm[rem]);
                    c2 = __float2int_rn(ext_frag[w_bias + 2] * inorm[rem]);
                    ext_frag[w_bias + 0] -= c0 * b_ext_ptr[2 * rem + 2];
                    ext_frag[w_bias + 2] -= c2 * b_ext_ptr[2 * rem + 2];
                    n0 += ext_frag[w_bias + 0] * ext_frag[w_bias + 0];
                    n1 += ext_frag[w_bias + 2] * ext_frag[w_bias + 2];
                    lane_ext_norm[0 * (ESD8 + 1) + rem] = n0;
                    lane_ext_norm[1 * (ESD8 + 1) + rem] = n1;
                }

                n0 = __shfl_sync(0xffffffff, n0, w_tid, 4);
                n1 = __shfl_sync(0xffffffff, n1, w_tid, 4);
                c0 = __shfl_sync(0xffffffff, c0, w_tid, 4);
                c1 = __shfl_sync(0xffffffff, c1, w_tid, 4);
                c2 = __shfl_sync(0xffffffff, c2, w_tid, 4);
                c3 = __shfl_sync(0xffffffff, c3, w_tid, 4);
                ext_coeff[rem + 0] = (tid & 0x2) ? c2 : c0;
                ext_coeff[rem + 1] = (tid & 0x2) ? c3 : c1;

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
        }
        
        __syncthreads();
        int4 v_tmp[11];
        int c[16];
        if ((tid & 0x1) == 0) {
            float *norm = &ext_norm[(tid / 2) * (ESD8 + 1)];
            for (int i = 0; i < ESD8; i++) {
                if (norm[i] < min_norm[i] && norm[i] > 0x1p-20f) {
                    utils_t::_spin_lock_device(&min_lock[i]);
                    if (norm[i] < ((volatile float *)min_norm_glob)[i]) {
                        for (int j = 0; j < 11; j++) v_tmp[j] = ((int4 *)(glob_vec + (tid / 2) * 176))[j];
                        int has_one = force_one ? 1 : 0;
                        if (has_one == 0) {
                            for (int rem = CSD16 - 16; rem >= 0; rem -= 16) {
                                int8_t *b_dual = b_dual_tri + (CSD16 - rem) * (CSD16 - rem - 16) / 2;
                                for (int j = 0; j < 16; j++) {
                                    c[j] = dhalf;
                                    for (int k = 0; k < CSD16 - rem; k++) c[j] += (int) ((int8_t *)v_tmp)[rem + k] * 
                                                                  (int) b_dual[(k / 16) * 256 + j * 16 + (k & 15)];
                                    c[j] >>= dshift;
                                    if (c[j] == -1 || c[j] == 1) {
                                        has_one = 1;
                                        break;
                                    }
                                }
                                if (has_one) break;
                            }
                        }
                        if (has_one) {
                            min_norm_glob[i] = norm[i];
                            for (int j = 0; j < ESD8; j++) min_coeff[i * 256 + j] = -ext_coeff[j];
                            for (int rem = CSD16 - 16; rem >= 0; rem -= 16) {
                                int8_t *b_dual = b_dual_tri + (CSD16 - rem) * (CSD16 - rem - 16) / 2;
                                for (int j = 0; j < 16; j++) c[j] = dhalf;
                                for (int j = 0; j < 16; j++) {
                                    for (int k = 0; k < CSD16 - rem; k++) c[j] += (int) ((int8_t *)v_tmp)[rem + k] * 
                                                                  (int) b_dual[(k / 16) * 256 + j * 16 + (k & 15)];
                                }
                                for (int j = 0; j < 16; j++) min_coeff[i * 256 + ESD8 + rem + j] = c[j] >> dshift;
                            }
                            *updated = 1;
                            __threadfence();
                        }
                    }
                    utils_t::_spin_unlock_device(&min_lock[i]);
                }
            }
        }
        __syncthreads();
        if (*updated && tid < ESD8) min_norm[tid] = min_norm_glob[tid];

        if (ind + stride < n) traits::_next_coeff(warp_coeff, warp_vec, b_dual_tri, dhalf,  
                                                        dshift, u_acc, uid_coeff, CSD16, lid);

        ind += stride;
        glob_vec += stride * vec_nbytes;
    }
}

template <uint32_t CSD16, uint32_t ESD8>
__global__ void insert_kernel(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data) {
    typedef insert_traits traits;
    pdev_kernel_var_declare();

    /// prepare data ///
    traits::_kernel_init(b_dual_tri, b_ext_head, uid_coeff, inorm, igh, local_data, ESD8, tid);
    
    /// for the first ind ///
    traits::_prep_vec(vec, glob_vec, tid);
    traits::_prep_bfull(lane_bfull, lane_bfull_glob, ESD8 + CSD16, tid);
    traits::_next_coeff(warp_coeff, warp_vec, b_dual_tri, dhalf, dshift, u_acc, uid_coeff, CSD16, lid);

    while (ind < n) {
        float fp32_frag0[64];   // 64R
        float fp32_frag1[64];   // 64R

        traits::_batch_init(fp32_frag0, fp32_frag1, sbuf, ESD8 + CSD16, wid, lid);
        
        #pragma unroll
        for (int rem = CSD16 - 16; rem >= 0; rem -= 16) {
            const int loop_size = (CSD16 - rem) >> 4;
            const int bd_bias = 256 / 2 * loop_size * (loop_size + 1);
            const int bf_bias = (CSD16 + ESD8) * loop_size * 16;
            const int c_bias = (loop_size & 0x1) * 16 * 132;
            const int f_bias = (loop_size & 0x1) * 28 * 132;

            if (rem != CSD16 - 16) {
                utils_t::_wait_async_group();
                __syncthreads();
            }

            if (rem) {
                traits::_prep_bfull(lane_bfull + f_bias, lane_bfull_glob + bf_bias, ESD8 + CSD16, tid);
                traits::_prep_coeff(warp_coeff + c_bias, warp_vec, b_dual_tri + bd_bias, 
                                          dhalf, dshift, rem, u_acc, uid_coeff, CSD16, lid);
            } else if (ind + stride < n) traits::_prep_vec(vec, glob_vec + stride * vec_nbytes, tid);


            if (wid < ESD8 / 16) {
                float *A_8x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + (tid >> 0x4) * 132;
                traits::_fmma8x16x8(fp32_frag0, A_8x16, B_8x16);
            }
            if (wid == ESD8 / 16 && (ESD8 & 0xf)) {
                float *A_4x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132 + (tid & 0x10) / 4;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + wid * 2 * 132;
                traits::_fmma4x16x8(fp32_frag0, A_4x16, B_8x16);
            }
            if (wid >= (ESD8 + 8) / 16 && wid <= (ESD8 + 8) / 16 + CSD16 / 16) {
                float *A_8x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + (tid >> 0x4) * 132 - ((ESD8 & 0x8) / 8) * 132;
                traits::_fmma8x16x8(fp32_frag0, A_8x16, B_8x16);
            }
            if (wid + 8 <= (ESD8 + 8) / 16 + CSD16 / 16) {
                float *A_8x16 = cbuf + (c_bias ^ (16 * 132)) + (tid & 0x0f) * 132;
                float *B_8x16 = fbuf + (f_bias ^ (28 * 132)) + (tid >> 0x4) * 132 + 16 * 132 - ((ESD8 & 0x8) / 8) * 132;
                traits::_fmma8x16x8(fp32_frag1, A_8x16, B_8x16);
            }
        }

        __syncthreads();
        if (ind + stride < n) traits::_prep_bfull(lane_bfull, lane_bfull_glob, ESD8 + CSD16, tid);
        
        float n_acc[8];
        int2 vp_line[16];
        uint32_t smsk_acc = 0x00;
        traits::_vec_stage0(vp_line, n_acc, smsk_acc, fp32_frag0, fp32_frag1, ESD8, CSD16, wid);

        traits::_uid_reduce(u_acc, ubuf, wid, lid);

        float ext_frag[ESD8 / 2];
        if (ESD8) traits::_ext_stage0(fbuf + 28 * 132, ext_frag, fp32_frag0, ESD8, wid, tid);

        int sn;
        float s0, s1;
        traits::_vec_stage1(fbuf + 28 * 132, vp_line, n_acc, smsk_acc, ESD8, CSD16, wid, lid, tid);
        traits::_vec_stage2(fbuf + 28 * 132, nbuf, sbuf, sn, s0, s1, ESD8, wid, lid, tid);
        
        if (ESD8) traits::_ext_stage1(ext_frag, sn, s0, s1, b_ext_head, igh, inorm, ESD8, tid);

        if (ind + stride < n) traits::_next_coeff(warp_coeff, warp_vec, b_dual_tri, dhalf,  
                                                        dshift, u_acc, uid_coeff, CSD16, lid);

        traits::_write_back(data, ind, (int *)vp_line, ubuf, nbuf, sbuf, sn, wid, lid, tid);

        ind += stride;
        glob_vec += stride * vec_nbytes;
    }
}

template <uint32_t CSD16>
__global__ void dim_lose_kernel(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data) {
    typedef dim_lose_traits traits;

    static constexpr unsigned int vec_nbytes = traits::vec_nbytes;

    extern __shared__ int8_t vec[];                         
    __shared__ int8_t b_dual_tri[176 * (176 + 16) / 2];     
    __shared__ uint64_t acc[176];                           

    const unsigned int wid = utils_t::_warp_id();
    const unsigned int lid = utils_t::_lane_id();
    const unsigned int tid = wid * warpSize + lid;

    const int dhalf = local_data->dhalf;
    const int dshift = local_data->dshift;

    int8_t *const warp_vec = vec + wid * 16 * vec_nbytes;
    int vec_bias = 0;

    int ind = 128 * blockIdx.x;
    const int stride = 128 * gridDim.x;
    int8_t *warp_glob_vec = &data[(ind + wid * 16) * vec_nbytes];

    for (int i = tid; i < sizeof(local_data_t::b_dual) / sizeof(int4); i += 256) 
        ((int4 *)b_dual_tri)[i] = ((int4 *)local_data->b_dual)[i];
    for (int i = n; i < ((n + 127) & ~127); i++) {
        if (tid < 11) ((int4 *)&data[i * vec_nbytes])[tid] = {0, 0, 0, 0};
    }
    if (n < 128) __syncthreads();
    if (tid < 176) acc[tid] = 0ULL;

    for (int i = 0; i < 11; i++) utils_t::_ldgsts_64b_async(warp_vec + 8 * lid + 8 * 32 * i, 
                                                       warp_glob_vec + 8 * lid + 8 * 32 * i);
    warp_glob_vec += stride * vec_nbytes;
    utils_t::_commit_async_group();
    utils_t::_wait_async_group();
    __syncthreads();

    uint32_t acc0[11] = {};
    uint32_t acc1[11] = {};

    while (ind < n) {
        utils_t::_wait_async_group();
        __syncwarp();

        if (ind + stride < n) {
            for (int i = 0; i < 11; i++) utils_t::_ldgsts_64b_async(warp_vec + (vec_bias ^ 22528) + 8 * lid + 8 * 32 * i, 
                                                                                    warp_glob_vec + 8 * lid + 8 * 32 * i);
            utils_t::_commit_async_group();
        }

        #pragma unroll
        for (int rem = CSD16 - 16; rem >= 0; rem -= 16) {
            const int loop_size = (CSD16 - rem) >> 4;
            const int bd_bias = 256 / 2 * loop_size * (loop_size - 1);

            wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, int> c;

            wmma::fill_fragment(c, dhalf);

            #pragma unroll
            for (int i = rem; i < CSD16; i += 16) {
                wmma::load_matrix_sync(a, b_dual_tri + bd_bias + (i - rem) * 16, 16);
                wmma::load_matrix_sync(b, warp_vec + vec_bias + i, vec_nbytes);
                wmma::mma_sync(c, a, b, c);
            }

            for (int j = 0; j < c.num_elements; j++) c.x[j] >>= dshift;
            for (int j = 0; j < c.num_elements; j++) c.x[j] = c.x[j] ? 1 : 0;
            int tmp0 = c.x[0] + c.x[1] + c.x[4] + c.x[5];
            int tmp1 = c.x[2] + c.x[3] + c.x[6] + c.x[7];
            tmp0 += __shfl_xor_sync(0xffffffff, tmp0, 1);
            tmp1 += __shfl_xor_sync(0xffffffff, tmp1, 1);
            tmp0 += __shfl_xor_sync(0xffffffff, tmp0, 2);
            tmp1 += __shfl_xor_sync(0xffffffff, tmp1, 2);
            if (lid % 4 == 0) acc0[rem / 16] += tmp0;
            if (lid % 4 == 0) acc1[rem / 16] += tmp1;
        }
        
        ind += stride;
        warp_glob_vec += stride * vec_nbytes;
        vec_bias ^= 128 * vec_nbytes;
    }
    
    if (lid % 4 == 0) {
        for (int i = 0; i < 11; i++) atomicAdd_block((unsigned long long int *)&acc[i * 16 + lid / 4 + 0], 
                                                     (unsigned long long int)acc0[i]);
        for (int i = 0; i < 11; i++) atomicAdd_block((unsigned long long int *)&acc[i * 16 + lid / 4 + 8], 
                                                     (unsigned long long int)acc1[i]);
    }
    __syncthreads();

    if (tid < 176) atomicAdd((unsigned long long int *)&local_data->uid_coeff[tid], 
                             (unsigned long long int)acc[tid]);
}


template __global__ void check_kernel<check_traits::vec_nbytes, check_traits::max_boost_dim>(
    int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data
);

template __global__ void extend_left_kernel<extend_left_traits::vec_nbytes, extend_left_traits::max_boost_dim>(
    int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data
);

template __global__ void min_lift_kernel<min_lift_traits::vec_nbytes, min_lift_traits::max_boost_dim>(
    int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data
);

template __global__ void min_lift_kernel<min_lift_traits::vec_nbytes, 40>(
    int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data
);

template __global__ void min_lift_kernel<min_lift_traits::vec_nbytes, 32>(
    int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data
);

template __global__ void insert_kernel<insert_traits::vec_nbytes, insert_traits::max_boost_dim>(
    int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data
);

template __global__ void dim_lose_kernel<dim_lose_traits::vec_nbytes>(
    int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data
);
#include "../include/config.h"
#include "../include/common_device.h"
#include "../include/pool_hd.h"


__global__ void _device_unpackf(int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec) {
    constexpr int batchVecs = utils_t::packBatch;
    constexpr int vec_nbytes = Pool_hd_t::vec_nbytes;

    extern __shared__ int8_t dy_buf[];

    for (long i = 0; i < 22; i++) {
        ((int4 *)dy_buf)[256 * i + threadIdx.x] = {0, 0, 0, 0};
    }
    __syncthreads();

    const unsigned int lid = utils_t::_lane_id();
    const unsigned int wid = utils_t::_thread_id() / 32;

    constexpr unsigned int wxor = batchVecs * vec_nbytes;
    const unsigned int int4_per_batch = batchVecs / 16 * CSD;
    int8_t *wdst = &dy_buf[wxor * (2 * wid + 0) * 2];
    int8_t *wsrc = &dy_buf[wxor * (2 * wid + 1) * 2];
    unsigned int wbias = 0;
    #define wpfch (wbias ^ wxor)

    int ind = batchVecs * (wid + utils_t::packThreads / 32 * blockIdx.x);
    const int stride = utils_t::packThreads / 32 * gridDim.x * batchVecs; 
    
    if (ind < num_vec) {
        if (lid < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid, ((int4 *)(src + ind * CSD)) + lid);
        if (lid + 32 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 32, ((int4 *)(src + ind * CSD)) + lid + 32);
        if (lid + 64 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 64, ((int4 *)(src + ind * CSD)) + lid + 64);
        if (lid + 96 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 96, ((int4 *)(src + ind * CSD)) + lid + 96);
        if (lid + 128 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 128, ((int4 *)(src + ind * CSD)) + lid + 128);
        if (lid + 160 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 160, ((int4 *)(src + ind * CSD)) + lid + 160);
        utils_t::_commit_async_group();
        utils_t::_wait_async_group();
        __syncwarp();
    }

    while (ind < num_vec) {
        if (ind + stride < num_vec) {
            if (lid < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid, ((int4 *)(src + (ind + stride) * CSD)) + lid);
            if (lid + 32 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 32, ((int4 *)(src + (ind + stride) * CSD)) + lid + 32);
            if (lid + 64 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 64, ((int4 *)(src + (ind + stride) * CSD)) + lid + 64);
            if (lid + 96 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 96, ((int4 *)(src + (ind + stride) * CSD)) + lid + 96);
            if (lid + 128 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 128, ((int4 *)(src + (ind + stride) * CSD)) + lid + 128);
            if (lid + 160 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 160, ((int4 *)(src + (ind + stride) * CSD)) + lid + 160);
            utils_t::_commit_async_group();
        }

        for (int i = 0; i < batchVecs; i++) {
            for (int j = lid; j < CSD; j += 32) {
                (wdst + wbias)[i * vec_nbytes + j] = (wsrc + wbias)[i * CSD + j];
            }
        }

        utils_t::_wait_async_group();
        __syncwarp();

        if (lid < wxor / 16) ((int4 *)(dst + ind * vec_nbytes))[lid] = ((int4 *)(wdst + wbias))[lid];
        if (lid + 32 < wxor / 16) ((int4 *)(dst + ind * vec_nbytes))[lid + 32] = ((int4 *)(wdst + wbias))[lid + 32];
        if (lid + 64 < wxor / 16) ((int4 *)(dst + ind * vec_nbytes))[lid + 64] = ((int4 *)(wdst + wbias))[lid + 64];
        if (lid + 96 < wxor / 16) ((int4 *)(dst + ind * vec_nbytes))[lid + 96] = ((int4 *)(wdst + wbias))[lid + 96];
        if (lid + 128 < wxor / 16) ((int4 *)(dst + ind * vec_nbytes))[lid + 128] = ((int4 *)(wdst + wbias))[lid + 128];
        if (lid + 160 < wxor / 16) ((int4 *)(dst + ind * vec_nbytes))[lid + 160] = ((int4 *)(wdst + wbias))[lid + 160];
        
        wbias = wpfch;
        ind += stride;
    }

    #undef wpfch
}

__global__ void _device_packf(int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec) {
    constexpr int batchVecs = utils_t::packBatch;
    constexpr int vec_nbytes = Pool_hd_t::vec_nbytes;
    extern __shared__ int8_t dy_buf[];

    const unsigned int lid = utils_t::_lane_id();
    const unsigned int wid = utils_t::_thread_id() / 32;

    constexpr unsigned int wxor = batchVecs * vec_nbytes;
    const unsigned int int4_per_batch = batchVecs / 16 * CSD;
    int8_t *wdst = &dy_buf[wxor * (2 * wid + 0) * 2];
    int8_t *wsrc = &dy_buf[wxor * (2 * wid + 1) * 2];
    unsigned int wbias = 0;
    #define wpfch (wbias ^ wxor)

    int ind = batchVecs * (wid + utils_t::packThreads / 32 * blockIdx.x);
    const int stride = utils_t::packThreads / 32 * gridDim.x * batchVecs; 
    
    if (ind < num_vec) {
        if (lid < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid, ((int4 *)(src + ind * vec_nbytes)) + lid);
        if (lid + 32 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 32, ((int4 *)(src + ind * vec_nbytes)) + lid + 32);
        if (lid + 64 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 64, ((int4 *)(src + ind * vec_nbytes)) + lid + 64);
        if (lid + 96 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 96, ((int4 *)(src + ind * vec_nbytes)) + lid + 96);
        if (lid + 128 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 128, ((int4 *)(src + ind * vec_nbytes)) + lid + 128);
        if (lid + 160 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 160, ((int4 *)(src + ind * vec_nbytes)) + lid + 160);
        utils_t::_commit_async_group();
        utils_t::_wait_async_group();
        __syncwarp();
    }

    while (ind < num_vec) {
        if (ind + stride < num_vec) {
            if (lid < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid, ((int4 *)(src + (ind + stride) * vec_nbytes)) + lid);
            if (lid + 32 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 32, ((int4 *)(src + (ind + stride) * vec_nbytes)) + lid + 32);
            if (lid + 64 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 64, ((int4 *)(src + (ind + stride) * vec_nbytes)) + lid + 64);
            if (lid + 96 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 96, ((int4 *)(src + (ind + stride) * vec_nbytes)) + lid + 96);
            if (lid + 128 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 128, ((int4 *)(src + (ind + stride) * vec_nbytes)) + lid + 128);
            if (lid + 160 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 160, ((int4 *)(src + (ind + stride) * vec_nbytes)) + lid + 160);
            utils_t::_commit_async_group();
        }

        for (int i = 0; i < batchVecs; i++) {
            for (int j = lid; j < CSD; j += 32) {
                (wdst + wbias)[i * CSD + j] = (wsrc + wbias)[i * vec_nbytes + j];
            }
        }

        utils_t::_wait_async_group();
        __syncwarp();

        if (lid < int4_per_batch) ((int4 *)(dst + ind * CSD))[lid] = ((int4 *)(wdst + wbias))[lid];
        if (lid + 32 < int4_per_batch) ((int4 *)(dst + ind * CSD))[lid + 32] = ((int4 *)(wdst + wbias))[lid + 32];
        if (lid + 64 < int4_per_batch) ((int4 *)(dst + ind * CSD))[lid + 64] = ((int4 *)(wdst + wbias))[lid + 64];
        if (lid + 96 < int4_per_batch) ((int4 *)(dst + ind * CSD))[lid + 96] = ((int4 *)(wdst + wbias))[lid + 96];
        if (lid + 128 < int4_per_batch) ((int4 *)(dst + ind * CSD))[lid + 128] = ((int4 *)(wdst + wbias))[lid + 128];
        if (lid + 160 < int4_per_batch) ((int4 *)(dst + ind * CSD))[lid + 160] = ((int4 *)(wdst + wbias))[lid + 160];
        
        wbias = wpfch;
        ind += stride;
    }

    #undef wpfch
}

template <int32_t CSD16>
__global__ void _device_unpack(int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec) {
    constexpr int batchVecs = utils_t::packBatch;
    constexpr int vec_nbytes = CSD16;

    extern __shared__ int8_t dy_buf[];

    for (long i = 0; i < 22; i++) {
        ((int4 *)dy_buf)[256 * i + threadIdx.x] = {0, 0, 0, 0};
    }
    __syncthreads();

    const unsigned int lid = utils_t::_lane_id();
    const unsigned int wid = utils_t::_thread_id() / 32;

    constexpr unsigned int wxor = batchVecs * vec_nbytes;
    const unsigned int int4_per_batch = batchVecs / 16 * CSD;
    int8_t *wdst = &dy_buf[wxor * (2 * wid + 0) * 2];
    int8_t *wsrc = &dy_buf[wxor * (2 * wid + 1) * 2];
    unsigned int wbias = 0;
    #define wpfch (wbias ^ wxor)

    int ind = batchVecs * (wid + utils_t::packThreads / 32 * blockIdx.x);
    const int stride = utils_t::packThreads / 32 * gridDim.x * batchVecs; 
    
    if (ind < num_vec) {
        if (lid < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid, ((int4 *)(src + ind * CSD)) + lid);
        if (lid + 32 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 32, ((int4 *)(src + ind * CSD)) + lid + 32);
        if (lid + 64 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 64, ((int4 *)(src + ind * CSD)) + lid + 64);
        if (lid + 96 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 96, ((int4 *)(src + ind * CSD)) + lid + 96);
        if (lid + 128 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 128, ((int4 *)(src + ind * CSD)) + lid + 128);
        if (lid + 160 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 160, ((int4 *)(src + ind * CSD)) + lid + 160);
        utils_t::_commit_async_group();
        utils_t::_wait_async_group();
        __syncwarp();
    }

    while (ind < num_vec) {
        if (ind + stride < num_vec) {
            if (lid < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid, ((int4 *)(src + (ind + stride) * CSD)) + lid);
            if (lid + 32 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 32, ((int4 *)(src + (ind + stride) * CSD)) + lid + 32);
            if (lid + 64 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 64, ((int4 *)(src + (ind + stride) * CSD)) + lid + 64);
            if (lid + 96 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 96, ((int4 *)(src + (ind + stride) * CSD)) + lid + 96);
            if (lid + 128 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 128, ((int4 *)(src + (ind + stride) * CSD)) + lid + 128);
            if (lid + 160 < int4_per_batch) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 160, ((int4 *)(src + (ind + stride) * CSD)) + lid + 160);
            utils_t::_commit_async_group();
        }

        for (int i = 0; i < batchVecs; i++) {
            for (int j = lid; j < CSD; j += 32) {
                (wdst + wbias)[i * vec_nbytes + j] = (wsrc + wbias)[i * CSD + j];
            }
        }

        utils_t::_wait_async_group();
        __syncwarp();

        if (lid < wxor / 16) ((int4 *)(dst + ind * vec_nbytes))[lid] = ((int4 *)(wdst + wbias))[lid];
        if (lid + 32 < wxor / 16) ((int4 *)(dst + ind * vec_nbytes))[lid + 32] = ((int4 *)(wdst + wbias))[lid + 32];
        if (lid + 64 < wxor / 16) ((int4 *)(dst + ind * vec_nbytes))[lid + 64] = ((int4 *)(wdst + wbias))[lid + 64];
        if (lid + 96 < wxor / 16) ((int4 *)(dst + ind * vec_nbytes))[lid + 96] = ((int4 *)(wdst + wbias))[lid + 96];
        if (lid + 128 < wxor / 16) ((int4 *)(dst + ind * vec_nbytes))[lid + 128] = ((int4 *)(wdst + wbias))[lid + 128];
        if (lid + 160 < wxor / 16) ((int4 *)(dst + ind * vec_nbytes))[lid + 160] = ((int4 *)(wdst + wbias))[lid + 160];
        
        wbias = wpfch;
        ind += stride;
    }

    #undef wpfch
}

template <int32_t CSD16>
__global__ void _device_pack(int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec) {
    constexpr int batchVecs = utils_t::packBatch;
    constexpr int vec_nbytes = CSD16;
    extern __shared__ int8_t dy_buf[];

    const unsigned int lid = utils_t::_lane_id();
    const unsigned int wid = utils_t::_thread_id() / 32;

    constexpr unsigned int wxor = batchVecs * vec_nbytes;
    const unsigned int int4_per_batch = batchVecs / 16 * CSD;
    int8_t *wdst = &dy_buf[wxor * (2 * wid + 0) * 2];
    int8_t *wsrc = &dy_buf[wxor * (2 * wid + 1) * 2];
    unsigned int wbias = 0;
    #define wpfch (wbias ^ wxor)

    int ind = batchVecs * (wid + utils_t::packThreads / 32 * blockIdx.x);
    const int stride = utils_t::packThreads / 32 * gridDim.x * batchVecs; 
    
    if (ind < num_vec) {
        if (lid < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid, ((int4 *)(src + ind * vec_nbytes)) + lid);
        if (lid + 32 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 32, ((int4 *)(src + ind * vec_nbytes)) + lid + 32);
        if (lid + 64 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 64, ((int4 *)(src + ind * vec_nbytes)) + lid + 64);
        if (lid + 96 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 96, ((int4 *)(src + ind * vec_nbytes)) + lid + 96);
        if (lid + 128 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 128, ((int4 *)(src + ind * vec_nbytes)) + lid + 128);
        if (lid + 160 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)wsrc) + lid + 160, ((int4 *)(src + ind * vec_nbytes)) + lid + 160);
        utils_t::_commit_async_group();
        utils_t::_wait_async_group();
        __syncwarp();
    }

    while (ind < num_vec) {
        if (ind + stride < num_vec) {
            if (lid < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid, ((int4 *)(src + (ind + stride) * vec_nbytes)) + lid);
            if (lid + 32 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 32, ((int4 *)(src + (ind + stride) * vec_nbytes)) + lid + 32);
            if (lid + 64 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 64, ((int4 *)(src + (ind + stride) * vec_nbytes)) + lid + 64);
            if (lid + 96 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 96, ((int4 *)(src + (ind + stride) * vec_nbytes)) + lid + 96);
            if (lid + 128 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 128, ((int4 *)(src + (ind + stride) * vec_nbytes)) + lid + 128);
            if (lid + 160 < wxor / 16) utils_t::_ldgsts_128b_async(((int4 *)(wsrc + wpfch)) + lid + 160, ((int4 *)(src + (ind + stride) * vec_nbytes)) + lid + 160);
            utils_t::_commit_async_group();
        }

        for (int i = 0; i < batchVecs; i++) {
            for (int j = lid; j < CSD; j += 32) {
                (wdst + wbias)[i * CSD + j] = (wsrc + wbias)[i * vec_nbytes + j];
            }
        }

        utils_t::_wait_async_group();
        __syncwarp();

        if (lid < int4_per_batch) ((int4 *)(dst + ind * CSD))[lid] = ((int4 *)(wdst + wbias))[lid];
        if (lid + 32 < int4_per_batch) ((int4 *)(dst + ind * CSD))[lid + 32] = ((int4 *)(wdst + wbias))[lid + 32];
        if (lid + 64 < int4_per_batch) ((int4 *)(dst + ind * CSD))[lid + 64] = ((int4 *)(wdst + wbias))[lid + 64];
        if (lid + 96 < int4_per_batch) ((int4 *)(dst + ind * CSD))[lid + 96] = ((int4 *)(wdst + wbias))[lid + 96];
        if (lid + 128 < int4_per_batch) ((int4 *)(dst + ind * CSD))[lid + 128] = ((int4 *)(wdst + wbias))[lid + 128];
        if (lid + 160 < int4_per_batch) ((int4 *)(dst + ind * CSD))[lid + 160] = ((int4 *)(wdst + wbias))[lid + 160];
        
        wbias = wpfch;
        ind += stride;
    }

    #undef wpfch
}


void utils_t::init_shared_mem_limit() {
    CHECK_CUDA_ERR(cudaFuncSetAttribute(_device_unpackf, cudaFuncAttributeMaxDynamicSharedMemorySize, utils_t::packshmem));
    CHECK_CUDA_ERR(cudaFuncSetAttribute(_device_packf, cudaFuncAttributeMaxDynamicSharedMemorySize, utils_t::packshmem));
}
void utils_t::device_unpackf(cudaStream_t stream, int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec) {
    _device_unpackf<<<packBlocks, packThreads, packshmem, stream>>>(dst, src, CSD, num_vec);
    CHECK_LAST_ERR;
}
void utils_t::device_packf(cudaStream_t stream, int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec) {
    _device_packf<<<packBlocks, packThreads, packshmem, stream>>>(dst, src, CSD, num_vec);
    CHECK_LAST_ERR;
}

template __global__ void _device_unpack<176>(
    int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec
);

template __global__ void _device_pack<176>(
    int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec
);


template __global__ void _device_unpack<160>(
    int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec
);

template __global__ void _device_pack<160>(
    int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec
);

template __global__ void _device_unpack<144>(
    int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec
);

template __global__ void _device_pack<144>(
    int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec
);

template __global__ void _device_unpack<128>(
    int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec
);

template __global__ void _device_pack<128>(
    int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec
);
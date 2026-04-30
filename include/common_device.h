#ifndef __COMMON_DEVICE_H
#define __COMMON_DEVICE_H

#include <stdio.h>

#define CHECK_CUDA_ERR(val) do {                                        \
    if (val) {                                                          \
        fprintf(stderr, "CUDA error at %s:%d \"%s\", code = %d(%s)\n",  \
        __FILE__, __LINE__, #val, val, cudaGetErrorString(val));        \
        cudaGetLastError();                                             \
        sleep(100000);                                                  \
    }                                                                   \
} while (0)

#define CHECK_LAST_ERR do {                                             \
    cudaError_t __err = cudaGetLastError();                             \
    if (__err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error found at %s:%d, code = %d(%s)\n",   \
        __FILE__, __LINE__, __err, cudaGetErrorString(__err));          \
        sleep(100000);                                                  \
    }                                                                   \
} while (0)

struct utils_t {
    static __device__ __forceinline__ unsigned int _lane_id() {
        unsigned int ret; 
        asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
        return ret;
    }
    static __device__ __forceinline__ unsigned int _warp_id() {
        unsigned int ret; 
        asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
        return ret;
    }
    static __device__ __forceinline__ unsigned int _thread_id() {
        unsigned int ret; 
        asm volatile ("mov.u32 %0, %tid.x;" : "=r"(ret));
        return ret;
    }
    static __device__ __forceinline__ void _ldgsts_128b_async(void *dst, const void *src) {
        int dst_sh_addr = __cvta_generic_to_shared(dst);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16, 16;\n"
            :
            : "r"(dst_sh_addr), "l"(src)
        );
    }
    static __device__ __forceinline__ void _ldgsts_64b_async(void *dst, const void *src) {
        int dst_sh_addr = __cvta_generic_to_shared(dst);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 8, 8;\n"
            :
            : "r"(dst_sh_addr), "l"(src)
        );
    }
    static __device__ __forceinline__ void _ldgsts_32b_async(void *dst, const void *src) {
        int dst_sh_addr = __cvta_generic_to_shared(dst);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4, 4;\n"
            :
            : "r"(dst_sh_addr), "l"(src)
        );
    }
    static __device__ __forceinline__ void _commit_async_group() {
        asm volatile(
            "cp.async.commit_group;\n"
            :
            :
        );
    }
    static __device__ __forceinline__ void _wait_async_group() {
        asm volatile(
            "cp.async.wait_group 0;\n"
            :
            :
        );
    }
    static __device__ __forceinline__ void _wait_async_group_exp1() {
        asm volatile(
            "cp.async.wait_group 1;\n"
            :
            :
        );
    }
    static __device__ __forceinline__ unsigned long long _clock() {
        unsigned long long ret;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(ret)::"memory");
        return ret;
    }

    static __device__ __forceinline__ void _spin_lock_device(int *_lock) {
        while (atomicCAS(_lock, 0, 1) != 0) {}
    }
    static __device__ __forceinline__ void _spin_unlock_device(int *_lock) {
        atomicExch(_lock, 0);
    }

    static constexpr int packBatch   = 16;
    static constexpr int packBlocks  = 64;
    static constexpr int packThreads = 256;
    static constexpr int packshmem   = 90112;
    static void init_shared_mem_limit();
    static void device_unpackf(cudaStream_t stream, int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec);
    static void device_packf(cudaStream_t stream, int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec);
};


__global__ void _device_unpackf(int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec);
__global__ void _device_packf(int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec);
template <int32_t CSD16>
__global__ void _device_unpack(int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec);
template <int32_t CSD16>
__global__ void _device_pack(int8_t *__restrict__ dst, int8_t *__restrict__ src, int CSD, int num_vec);

#endif
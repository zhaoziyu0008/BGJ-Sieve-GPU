#ifndef __RANDOM_DEVICE_H
#define __RANDOM_DEVICE_H

#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int buccg_blocks = 1;
constexpr int buccg_threads = 256;
constexpr int buccg_shmem = 90112;

struct vec_t {
    int8_t v[176];
};


__global__ void init_curand(curandState *state, unsigned long long seed);

__global__ void buc_ctr_gen(int8_t *dst, int32_t goal_norm, curandState *state, int CSD, int CSD16, int batch0);

__global__ void bgj2_ctr_gen(int8_t *c1, __grid_constant__ const vec_t c0, float a0, 
                             int32_t goal_norm, curandState *state, int CSD, int CSD16, int batch1);

__global__ void bgj34_ctr_gen(int8_t *c2, int8_t *vec_pad16, int *buc, int *in_size, uint32_t in_max_size,
                              int32_t goal_norm, curandState *state, int CSD, int CSD16, int batch2);


#endif

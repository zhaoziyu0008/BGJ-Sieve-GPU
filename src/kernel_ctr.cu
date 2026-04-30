#include "../include/random_device.h"


__global__ void init_curand(curandState *state, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void buc_ctr_gen(int8_t *dst, int32_t goal_norm, curandState *state, int CSD, int CSD16, int batch0) {
    int tid = threadIdx.x;

    extern __shared__ int4 shmem[];

    int sh_bias = 0;

    const int target_norm = goal_norm * 2;
    const float target_std = sqrtf((float) target_norm / (float) CSD);

    int4 v4[11] = {};
    int8_t *v = (int8_t *) v4;

    for (int i = 0; i < batch0; i += buccg_threads) {
        int rem = min(buccg_threads, batch0 - i);

        int4 *t_sh_v4 = shmem + tid * CSD16 / 16 + sh_bias;
        int4 *sh_v4 = shmem + sh_bias;
        sh_bias ^= buccg_threads * CSD16 / 16;

        int rn = 0;

        #pragma unroll
        for (int j = 0; j < CSD; j++) {
            v[j] = __float2int_rn(curand_normal(&state[tid]) * target_std);
        }

        /// while (abs(v[0]) > target_std) {
        ///     v[0] = __float2int_rn(curand_normal(&state[tid]) * target_std);
        /// }
        v[0] = 0.0f;


        #pragma unroll
        for (int j = 0; j < CSD; j++) {
            rn += (int) v[j] * (int) v[j];
        }

        float ratio = sqrtf((float) target_norm / (float) rn);

        #pragma unroll
        for (int j = 0; j < CSD; j++) {
            v[j] = __float2int_rn((float) v[j] * ratio);
        }

        rn = 0;

        #pragma unroll
        for (int j = 0; j < CSD; j++) {
            rn += (int) v[j] * (int) v[j];
        }

        rn -= target_norm;

        for (int j = 0; j < CSD; j++) {
            if (abs(rn) < 50) break;
            if (abs(v[j]) * 2 < abs(rn)) {
                if (v[j] < 0 && rn > 0) {
                    v[j]++;
                    rn += 2 * v[j] - 1;
                }
                if (v[j] < 0 && rn < 0) {
                    if (v[j] != -128) {
                        v[j]--;
                        rn -= 2 * v[j] + 1;
                    }
                }
                if (v[j] > 0 && rn > 0) {
                    if (v[j] != 127) {
                        v[j]--;
                        rn -= 2 * v[j] + 1;
                    }
                }
                if (v[j] > 0 && rn < 0) {
                    v[j]++;
                    rn += 2 * v[j] - 1;
                }
            }
        }
        
        for (int j = 0; j < CSD16 / 16; j++) {
            t_sh_v4[j] = v4[j];
        }

        __syncthreads();

        for (int j = tid; j < CSD16 / 16 * rem; j += buccg_threads) {
            *((int4 *)(dst + i * CSD16 + j * 16)) = *((int4 *)(sh_v4 + j));
        }
    }
}

__global__ void bgj2_ctr_gen(int8_t *c1, __grid_constant__ const vec_t c0, float a0, 
                             int32_t goal_norm, curandState *state, int CSD, int CSD16, int batch1) {
    int tid = threadIdx.x;

    extern __shared__ int4 shmem[];

    int sh_bias = 0;

    const int target_norm = goal_norm * 2;
    const float target_std = sqrtf((float) target_norm / (float) CSD);
    const float lambda = a0 / sqrtf(1.0f - a0 * a0);
    
    float icn = 0.0f;
    for (int i = 0; i < CSD; i++) icn += c0.v[i] * c0.v[i];
    icn = 1.0f / icn;

    float vf[176] = {};
    int4  v4[11] = {};
    int8_t *v = (int8_t *) v4;

    for (int i = 0; i < batch1; i += buccg_threads) {
        int rem = min(buccg_threads, batch1 - i);

        int4 *t_sh_v4 = shmem + tid * CSD16 / 16 + sh_bias;
        int4 *sh_v4 = shmem + sh_bias;
        sh_bias ^= buccg_threads * CSD16 / 16;

        #pragma unroll
        for (int j = 0; j < CSD; j++) vf[j] = curand_normal(&state[tid]) * target_std;

        /// while (fabs(vf[0]) > target_std) vf[0] = curand_normal(&state[tid]) * target_std;
        vf[0] = 0.0f;

        float dp = 0.0f;
        for (int j = 0; j < CSD; j++) dp += vf[j] * c0.v[j];

        for (int j = 0; j < CSD; j++) vf[j] -= dp * c0.v[j] * icn;

        float rnf = 0.0f;

        #pragma unroll
        for (int j = 0; j < 176; j++) rnf += vf[j] * vf[j];

        float ratio = sqrtf((float) target_norm / rnf);

        #pragma unroll
        for (int j = 0; j < 176; j++) vf[j] = vf[j] * ratio + lambda * c0.v[j];

        rnf = 0.0f;
        for (int j = 0; j < 176; j++) rnf += vf[j] * vf[j];

        ratio = sqrtf((float) target_norm / rnf);

        #pragma unroll
        for (int j = 0; j < 176; j++) v[j] = __float2int_rn(vf[j] * ratio);

        int rn = 0;

        #pragma unroll
        for (int j = 0; j < CSD; j++) rn += (int) v[j] * (int) v[j];

        rn -= target_norm;

        for (int j = 0; j < CSD; j++) {
            if (abs(rn) < 50) break;
            if (abs(v[j]) * 2 < abs(rn)) {
                if (v[j] < 0 && rn > 0) {
                    v[j]++;
                    rn += 2 * v[j] - 1;
                }
                if (v[j] < 0 && rn < 0) {
                    if (v[j] != -128) {
                        v[j]--;
                        rn -= 2 * v[j] + 1;
                    }
                }
                if (v[j] > 0 && rn > 0) {
                    if (v[j] != 127) {
                        v[j]--;
                        rn -= 2 * v[j] + 1;
                    }
                }
                if (v[j] > 0 && rn < 0) {
                    v[j]++;
                    rn += 2 * v[j] - 1;
                }
            }
        }
        
        for (int j = 0; j < CSD16 / 16; j++) t_sh_v4[j] = v4[j];

        __syncthreads();

        for (int j = tid; j < CSD16 / 16 * rem; j += buccg_threads) {
            *((int4 *)(c1 + i * CSD16 + j * 16)) = *((int4 *)(sh_v4 + j));
        }
    }
}

__global__ void bgj34_ctr_gen(int8_t *c2, int8_t *vec_pad16, int *buc, int *in_size, uint32_t in_max_size,
                              int32_t goal_norm, curandState *state, int CSD, int CSD16, int batch2) {
    int tid = threadIdx.x;
    int lid = tid % 32;

    extern __shared__ int4 shmem[];

    __shared__ int sh_idx[buccg_threads];

    int sh_bias = 0;

    const int target_norm = goal_norm * 2;
    const int size = in_size[0] > in_max_size ? in_max_size : in_size[0];
    const int mod = (size - tid) / buccg_threads;

    if (size == 0) return;

    for (int i = 0; i < batch2; i += buccg_threads) {
        int rem = min(buccg_threads, batch2 - i);
        
        int4 *t_sh_v4 = shmem + tid * CSD16 / 16 + sh_bias;
        int4 *sh_v4 = shmem + sh_bias;
        sh_bias ^= buccg_threads * CSD16 / 16;
        
        int idx = mod >= 1 ? buccg_threads * (curand(&state[tid]) % mod) + tid : curand(&state[tid]) % size;
        sh_idx[tid] = buc[2 * idx];

        __syncwarp();

        for (int j = (tid / 32) * 32; j < (tid / 32) * 32 + 32; j++) {
            int64_t id = sh_idx[j];
            if (lid < CSD16 / 16) {
                int4 *v4 = (int4 *)(vec_pad16 + id * CSD16 + lid * 16);
                sh_v4[j * CSD16 / 16 + lid] = *v4;
            }
        }

        __syncwarp();

        int4 v4[11];
        int8_t *v = (int8_t *) v4;

        for (int j = 0; j < CSD16 / 16; j++) {
            v4[j] = t_sh_v4[j];
        }
        v[0] = 0;

        int rn = 0;

        #pragma unroll
        for (int j = 0; j < CSD; j++) {
            rn += (int) v[j] * (int) v[j];
        }

        float ratio = sqrtf((float) target_norm / (float) rn);

        #pragma unroll
        for (int j = 0; j < CSD; j++) {
            v[j] = __float2int_rn((float) v[j] * ratio);
        }

        rn = 0;

        #pragma unroll
        for (int j = 0; j < CSD; j++) {
            rn += (int) v[j] * (int) v[j];
        }

        rn -= target_norm;

        for (int j = 0; j < CSD; j++) {
            if (abs(rn) < 50) break;
            if (abs(v[j]) * 2 < abs(rn)) {
                if (v[j] < 0 && rn > 0) {
                    v[j]++;
                    rn += 2 * v[j] - 1;
                }
                if (v[j] < 0 && rn < 0) {
                    if (v[j] != -128) {
                        v[j]--;
                        rn -= 2 * v[j] + 1;
                    }
                }
                if (v[j] > 0 && rn > 0) {
                    if (v[j] != 127) {
                        v[j]--;
                        rn -= 2 * v[j] + 1;
                    }
                }
                if (v[j] > 0 && rn < 0) {
                    v[j]++;
                    rn += 2 * v[j] - 1;
                }
            }
        }

        for (int j = 0; j < CSD16 / 16; j++) {
            t_sh_v4[j] = v4[j];
        }

        __syncthreads();

        for (int j = tid; j < CSD16 / 16 * rem; j += buccg_threads) {
            *((int4 *)(c2 + i * CSD16 + j * 16)) = *((int4 *)(sh_v4 + j));
        }
    }
}
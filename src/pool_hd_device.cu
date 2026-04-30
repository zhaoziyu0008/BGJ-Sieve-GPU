/// @file pool_hd_device.cu
/// @brief Implementation of device related pool operations

#include "../include/pool_hd_device.h"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace nvcuda;

#include <algorithm>    // for sort, maybe...

double dot_avx2(double *src1, double *src2, long n);
void red(float *dst, float *src, float q, long n);
void copy(float *dst, float *src, long n);

static std::atomic<int> ck_allocator_started{0};

#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#include <fcntl.h>

struct chunk_allocator_t {
    static constexpr long max_cached_chunks = PWC_DEFAULT_MAX_CACHED_CHUNKS + 
              BWC_DEFAULT_MAX_CACHED_CHUNKS + SWC_DEFAULT_MAX_CACHED_CHUNKS + 256;

    chunk_allocator_t() {}

    ~chunk_allocator_t() {
        if (!ck_allocator_started.load()) return;
        pthread_spin_lock(&cache_lock);
        cached_num = 0;
        if (using_num) {
            fprintf(stderr, "[Warning] %ld chunks not freed\n", using_num);
            using_num = 0;
        }
        pthread_spin_unlock(&cache_lock);
        pthread_spin_destroy(&cache_lock);
        free(cached_chunks);
    }

    void destory() {
        if (!ck_allocator_started.load()) return;
        pthread_spin_lock(&cache_lock);
        if (space) {
            #if USE_HUGE_PAGE
            CHECK_CUDA_ERR(cudaHostUnregister(space));
            munmap(space, hp_size);
            close(hp_fd);
            #else
            CHECK_CUDA_ERR(cudaHostUnregister(space));
            free(space);
            #endif
            space = NULL;
        }
        cached_num = 0;
        if (using_num) {
            fprintf(stderr, "[Warning] %ld chunks not freed\n", using_num);
            using_num = 0;
        }
        pthread_spin_unlock(&cache_lock);
        cudaDeviceReset();
    }

    void _ck_allocator_start() {
        struct bitmask *nodes = numa_get_mems_allowed();
        numa_set_interleave_mask(nodes);
        numa_bitmask_free(nodes);

        cached_chunks = (chunk_t *) malloc(max_cached_chunks * sizeof(chunk_t));
        pthread_spin_init(&cache_lock, PTHREAD_PROCESS_SHARED);

        #if ONE_TIME_IO
        long chunk_nbytes = Pool_hd_t::chunk_max_nvecs * (long)(176 + 8 + 4 + 2) + 4096;
        #else
        long chunk_nbytes = Pool_hd_t::chunk_max_nvecs * (long)(176 + 8 + 4 + 2);
        #endif

        #if USE_HUGE_PAGE
        size_t hp_size = ((max_cached_chunks * chunk_nbytes + 2097151L) / 2097152L) * 2097152L;
        hp_fd = open("/dev/hugepages/hugepagefile", O_CREAT | O_RDWR, 0755);
        if (hp_fd < 0) {
            perror("open");
            exit(1);
        }
        space = (int8_t *)mmap(NULL, hp_size, PROT_READ | PROT_WRITE, MAP_SHARED, hp_fd, 0);
        CHECK_CUDA_ERR(cudaHostRegister(space, hp_size, cudaHostAllocPortable));
        if (space == MAP_FAILED) {
            perror("mmap");
            close(hp_fd);
            exit(1);
        }
        #else
        if (posix_memalign((void **)&space, 4096, max_cached_chunks * chunk_nbytes)) {
            printf("[Error] _ck_allocator_start: posix_memalign failed");
            fflush(stdout);
        }
        CHECK_CUDA_ERR(cudaHostRegister(space, max_cached_chunks * chunk_nbytes, cudaHostAllocPortable));
        #endif


        for (long i = 0; i < max_cached_chunks; i++) {
            cached_chunks[i].score = (uint16_t *) (space + i * chunk_nbytes + (ONE_TIME_IO ? 12L : 0));
            cached_chunks[i].norm  = (int32_t  *) (cached_chunks[i].score + Pool_hd_t::chunk_max_nvecs);
            cached_chunks[i].u     = (uint64_t *) (cached_chunks[i].norm + Pool_hd_t::chunk_max_nvecs);
            cached_chunks[i].vec   = (int8_t   *) (cached_chunks[i].u + Pool_hd_t::chunk_max_nvecs);
        }
        cached_num = max_cached_chunks;
    }

    void _outer_malloc_chunk(chunk_t *chunk) {
        for (;;) {
            pthread_spin_lock(&cache_lock);
            if (cached_num > 0) {
                using_num++;
                chunk_t *src = &cached_chunks[--cached_num];
                chunk->score = src->score;
                chunk->vec   = src->vec;
                chunk->norm  = src->norm;
                chunk->u     = src->u;
                src->score   = NULL;
                src->vec     = NULL;
                src->norm    = NULL;
                src->u       = NULL;
                pthread_spin_unlock(&cache_lock);
                return;
            }
            pthread_spin_unlock(&cache_lock);
            usleep(1000);
        }
    }

    void _outer_free_chunk(chunk_t *src) {
        pthread_spin_lock(&cache_lock);
        using_num--;
        chunk_t *dst = &cached_chunks[cached_num++];
        dst->score = src->score;
        dst->vec   = src->vec;
        dst->norm  = src->norm;
        dst->u     = src->u;
        pthread_spin_unlock(&cache_lock);
    }


    private:
    int                hp_fd;
    size_t             hp_size;
    pthread_spinlock_t cache_lock;
    long               cached_num = 0;
    long               using_num  = 0;
    chunk_t           *cached_chunks = NULL;
    int8_t            *space = NULL;
};

static chunk_allocator_t chunk_allocator;

void _start_ck_allocator() {
    if (!ck_allocator_started.fetch_or(1, std::memory_order_acq_rel)) {
        chunk_allocator._ck_allocator_start();
    }
}

void _destory_ck_allocator() {
    chunk_allocator.destory();
}

extern void _malloc_chunk(chunk_t *chunk) {
    chunk_allocator._outer_malloc_chunk(chunk);
}

extern void _free_chunk(chunk_t *chunk) {
    chunk_allocator._outer_free_chunk(chunk);
}

extern int _prepare_device_prop(int &num_devices, cudaDeviceProp device_props[]) {
    num_devices = hw::gpu_num;
    if (num_devices == 0) {
        return -1;
    } else {
        for (int i = 0; i < num_devices; i++) {
            CHECK_CUDA_ERR(cudaGetDeviceProperties(&device_props[i], hw::gpu_id_list[i]));
        }
    }
    return 0;
}

#define WRITE_BACK_TO_CHUNK(_chunk)                                                                 \
    do {                                                                                            \
        _chunk->size = 0;                                                                           \
        int _n = task_vecs - num_used > chunk_max_nvecs ? chunk_max_nvecs : task_vecs - num_used;   \
        if (_n > 0) {                                                                               \
            memcpy(_chunk->score, &h_buffer_score[num_used], _n * sizeof(uint16_t));                \
            memcpy(_chunk->norm, &h_buffer_norm[num_used], _n * sizeof(int32_t));                   \
            memcpy(_chunk->u, &h_buffer_u[num_used], _n * sizeof(uint64_t));                        \
        } else _n = 0;                                                                              \
        if (_n < chunk_max_nvecs) {                                                                 \
            memset(&_chunk->score[_n], 0, (chunk_max_nvecs - _n) * sizeof(uint16_t));               \
            memset(&_chunk->norm[_n], 0, (chunk_max_nvecs - _n) * sizeof(int32_t));                 \
        }                                                                                           \
        for (int j = 0; j < _n; j++) if (_chunk->score[j]) _chunk->size++;                          \
        ut_checker.task_commit(_chunk);                                                             \
        num_used += _n;                                                                             \
    } while (0)


template <>
void check_traits::_prep_device_local_data(int CSD16, int ESD8, local_data_t *&local_data, Pool_hd_t *p) {
    local_data_t *host_local_data = (local_data_t *) calloc(1, sizeof(local_data_t));
    host_local_data->CSD = p->CSD;
    host_local_data->ESD = p->ESD;
    host_local_data->dhalf = p->_dhalf;
    host_local_data->dshift = p->_dshift;

    const int CSD = p->CSD;
    const int ESD = p->ESD;

    for (int i = ESD; i < ESD8; i++) host_local_data->igh[i]    = 0x1p20f;
    for (int i = 0; i < ESD; i++) host_local_data->igh[i]       = p->_boost_data->igh[i];
    for (int i = 0; i < ESD; i++) host_local_data->inorm[i]     = p->_boost_data->inorm[i];
    for (int i = 0; i < CSD; i++) host_local_data->uid_coeff[i] = p->uid_table->coeff(i);
    
    int8_t *ip = host_local_data->b_dual;
    for (int row = CSD16 - 16; row >= 0; row -= 16) {
        for (int col = row; col < CSD16; col += 16) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 16; j++) *ip++ = (i < CSD && j < CSD) ? p->_b_dual[i * vec_nbytes + j] : 0;
            }
        }
    }

    float *fp = host_local_data->b_full;
    for (int row = CSD16 - 16; row >= 0; row -= 16) {
        for (int col = 0; col < ESD8; col += 8) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 8; j++) {
                    *fp++ = (i < CSD && j < ESD) ? p->_boost_data->evec[(ESD + i) * ESD + j] : 0.0f;
                }
            }
        }
        for (int col = 0; col < row + 16; col += 16) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 8; j++) {
                    *fp++ = (i < CSD && j < CSD) ? p->_b_local[i][j] : 0.0f;
                }
            }

            for (int i = row; i < row + 16; i++) {
                for (int j = col + 8; j < col + 16; j++) {
                    *fp++ = (i < CSD && j < CSD) ? p->_b_local[i][j] : 0.0f;
                }
            }
        }
    }

    float *ep = host_local_data->b_ext_head;
    for (int i = ESD8 - 1; i >= 0; i--) {
        for (int j = 0; j < ((i + 2) & ~1); j++) {
            *ep++ = (i < ESD && j < ESD) ? p->_boost_data->evec[i * ESD + j] : 0.0f;
        }
    }
    
    CHECK_CUDA_ERR(cudaMalloc(&local_data, sizeof(local_data_t)));
    CHECK_CUDA_ERR(cudaMemcpy(local_data, host_local_data, sizeof(local_data_t), cudaMemcpyHostToDevice));
    free(host_local_data);
}

template <>
void extend_left_traits::_prep_device_local_data(int CSD16, int ESD8, local_data_t *&local_data, Pool_hd_t *p) {
    int32_t _dhalf_old = p->_dhalf;
    int32_t _dshift_old = p->_dshift;
    int8_t *_b_dual_old = p->_b_dual;
    p->_b_dual = NULL;

    p->index_l--;
    p->CSD++;
    p->_update_b_local();
    p->uid_table->reset_hash_function(p->CSD);

    if (p->ESD > p->index_l) p->ESD--;
    if (p->ESD) p->_update_boost_data();
    
    
    local_data_t *host_local_data = (local_data_t *) calloc(1, sizeof(local_data_t));
    host_local_data->CSD = p->CSD;
    host_local_data->ESD = p->ESD;
    host_local_data->dhalf = _dhalf_old;
    host_local_data->dshift = _dshift_old;

    const int CSD = p->CSD;
    const int ESD = p->ESD;

    for (int i = ESD; i < ESD8; i++) host_local_data->igh[i]        = 0x1p20f;
    for (int i = 0; i < ESD; i++) host_local_data->igh[i]           = p->_boost_data->igh[i];
    for (int i = 0; i < ESD; i++) host_local_data->inorm[i]         = p->_boost_data->inorm[i];
    for (int i = 0; i < CSD - 1; i++) host_local_data->uid_coeff[i] = p->uid_table->coeff(i + 1);
    host_local_data->uid_coeff[CSD16 - 1]                           = p->uid_table->coeff(0);

    int8_t *ip = host_local_data->b_dual;
    for (int row = CSD16 - 16; row >= 0; row -= 16) {
        for (int col = row; col < CSD16; col += 16) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 16; j++) *ip++ = (i < CSD - 1 && j < CSD - 1) ? _b_dual_old[i * vec_nbytes + j] : 0;
            }
        }
    }

    float *fp = host_local_data->b_full;
    for (int row = CSD16 - 16; row >= 0; row -= 16) {
        for (int col = 0; col < ESD8; col += 8) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 8; j++) {
                    *fp++ = (i + 1 < CSD && j < ESD) ? p->_boost_data->evec[(ESD + i + 1) * ESD + j] : 0.0f;
                }
            }
        }
        for (int col = 0; col <= row + 16 && col < CSD16; col += 16) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 8; j++) {
                    *fp++ = (i + 1 < CSD && j < CSD) ? p->_b_local[i + 1][j] : 0.0f;
                }
            }

            for (int i = row; i < row + 16; i++) {
                for (int j = col + 8; j < col + 16; j++) {
                    *fp++ = (i + 1 < CSD && j < CSD) ? p->_b_local[i + 1][j] : 0.0f;
                }
            }
        }
    }
    host_local_data->b_full[ESD8 * CSD16 + (CSD16 / 2 + 8) * CSD16 + 16 * (CSD16 - 16) + ESD8] = p->_b_local[0][0];
    for (int i = 0; i < ESD8; i++) host_local_data->b_full[ESD8 * CSD16 + (CSD16 / 2 + 8) * CSD16  + 16 * (CSD16 - 16) + i] = 
                                                                    (i < ESD) ? p->_boost_data->evec[ESD * ESD + i] : 0.0f;

    float *ep = host_local_data->b_ext_head;
    for (int i = ESD8 - 1; i >= 0; i--) {
        for (int j = 0; j < ((i + 2) & ~1); j++) {
            *ep++ = (i < ESD && j < ESD) ? p->_boost_data->evec[i * ESD + j] : 0.0f;
        }
    }

    CHECK_CUDA_ERR(cudaMalloc(&local_data, sizeof(local_data_t)));
    CHECK_CUDA_ERR(cudaMemcpy(local_data, host_local_data, sizeof(local_data_t), cudaMemcpyHostToDevice));
    free(host_local_data);
    FREE_VEC((void *)_b_dual_old);
}

template <>
void shrink_left_traits::_prep_device_local_data(int CSD16, int ESD8, local_data_t *&local_data, Pool_hd_t *p) {
    int32_t _dhalf_old = p->_dhalf;
    int32_t _dshift_old = p->_dshift;
    int8_t *_b_dual_old = p->_b_dual;
    p->_b_dual = NULL;

    p->index_l++;
    p->CSD--;
    p->_update_b_local();
    p->uid_table->reset_hash_function(p->CSD);

    if (p->ESD) p->_update_boost_data();

    local_data_t *host_local_data = (local_data_t *) calloc(1, sizeof(local_data_t));
    host_local_data->CSD = p->CSD;
    host_local_data->ESD = p->ESD;
    host_local_data->dhalf = _dhalf_old;
    host_local_data->dshift = _dshift_old;

    const int CSD = p->CSD;
    const int ESD = p->ESD;

    for (int i = ESD; i < ESD8; i++) host_local_data->igh[i]        = 0x1p20f;
    for (int i = 0; i < ESD; i++) host_local_data->igh[i]           = p->_boost_data->igh[i];
    for (int i = 0; i < ESD; i++) host_local_data->inorm[i]         = p->_boost_data->inorm[i];
    for (int i = 1; i < CSD + 1; i++) host_local_data->uid_coeff[i] = p->uid_table->coeff(i - 1);
    host_local_data->uid_coeff[0]                                   = 0ULL;

    int8_t *ip = host_local_data->b_dual;
    for (int row = CSD16 - 16; row >= 0; row -= 16) {
        for (int col = row; col < CSD16; col += 16) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 16; j++) *ip++ = (i < CSD + 1 && j < CSD + 1) ? _b_dual_old[i * vec_nbytes + j] : 0;
            }
        }
    }

    float *fp = host_local_data->b_full;
    for (int row = CSD16 - 16; row >= 0; row -= 16) {
        for (int col = 0; col < ESD8; col += 8) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 8; j++) {
                    *fp++ = (i && i < CSD + 1 && j < ESD) ? p->_boost_data->evec[(ESD + i - 1) * ESD + j] : 0.0f;
                }
            }
        }
        for (int col = 0; col < row + 16; col += 16) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 8; j++) {
                    *fp++ = (i && i < CSD + 1 && j < CSD) ? p->_b_local[i - 1][j] : 0.0f;
                }
            }

            for (int i = row; i < row + 16; i++) {
                for (int j = col + 8; j < col + 16; j++) {
                    *fp++ = (i && i < CSD + 1 && j < CSD) ? p->_b_local[i - 1][j] : 0.0f;
                }
            }
        }
    }

    float *ep = host_local_data->b_ext_head;
    for (int i = ESD8 - 1; i >= 0; i--) {
        for (int j = 0; j < ((i + 2) & ~1); j++) {
            *ep++ = (i < ESD && j < ESD) ? p->_boost_data->evec[i * ESD + j] : 0.0f;
        }
    }

    CHECK_CUDA_ERR(cudaMalloc(&local_data, sizeof(local_data_t)));
    CHECK_CUDA_ERR(cudaMemcpy(local_data, host_local_data, sizeof(local_data_t), cudaMemcpyHostToDevice));
    free(host_local_data);
    FREE_VEC((void *)_b_dual_old);
}

template <>
void min_lift_traits::_prep_device_local_data(int CSD16, int ESD8, local_data_t *&local_data, Pool_hd_t *p) {
    int force_one = ((int *)local_data)[0];
    free(local_data);
    local_data_t *host_local_data = (local_data_t *) calloc(1, sizeof(local_data_t) + ESD8 * (1024 + 4 + 4));
    host_local_data->CSD = p->CSD;
    host_local_data->ESD = p->ESD;
    host_local_data->dhalf = p->_dhalf;
    host_local_data->dshift = p->_dshift;

    const int CSD = p->CSD;
    const int ESD = p->ESD;

    for (int i = ESD; i < ESD8; i++) host_local_data->igh[i]    = 0x1p20f;
    for (int i = 0; i < ESD; i++) host_local_data->igh[i]       = p->_boost_data->igh[i];
    for (int i = 0; i < ESD; i++) host_local_data->inorm[i]     = p->_boost_data->inorm[i];
    for (int i = 0; i < CSD; i++) host_local_data->uid_coeff[i] = p->uid_table->coeff(i);
    host_local_data->uid_coeff[0] = force_one;
    
    int8_t *ip = host_local_data->b_dual;
    for (int row = CSD16 - 16; row >= 0; row -= 16) {
        for (int col = row; col < CSD16; col += 16) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 16; j++) *ip++ = (i < CSD && j < CSD) ? p->_b_dual[i * vec_nbytes + j] : 0;
            }
        }
    }

    float *fp = host_local_data->b_full;
    for (int row = CSD16 - 16; row >= 0; row -= 16) {
        for (int col = 0; col < ESD8; col += 8) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 8; j++) {
                    *fp++ = (i < CSD && j < ESD) ? p->_boost_data->evec[(ESD + i) * ESD + j] : 0.0f;
                }
            }
        }
        for (int col = 0; col < row + 16; col += 16) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 8; j++) {
                    *fp++ = (i < CSD && j < CSD) ? p->_b_local[i][j] : 0.0f;
                }
            }

            for (int i = row; i < row + 16; i++) {
                for (int j = col + 8; j < col + 16; j++) {
                    *fp++ = (i < CSD && j < CSD) ? p->_b_local[i][j] : 0.0f;
                }
            }
        }
    }

    float *ep = host_local_data->b_ext_head;
    for (int i = ESD8 - 1; i >= 0; i--) {
        for (int j = 0; j < ((i + 2) & ~1); j++) {
            *ep++ = (i < ESD && j < ESD) ? p->_boost_data->evec[i * ESD + j] : 0.0f;
        }
    }

    float *tail = (float *)(host_local_data + 1);
    for (int i = 0; i < ESD8; i++) {
        tail[i] = i < ESD ? p->_boost_data->evec[i * ESD + i] * p->_boost_data->evec[i * ESD + i] * 0x1p20f : 0.0f;
    }
    
    CHECK_CUDA_ERR(cudaMalloc(&local_data, sizeof(local_data_t) + ESD8 * (1024 + 4 + 4)));
    CHECK_CUDA_ERR(cudaMemcpy(local_data, host_local_data, sizeof(local_data_t) + ESD8 * (1024 + 4 + 4), cudaMemcpyHostToDevice));
    free(host_local_data);
}

template <>
void insert_traits::_prep_device_local_data(int CSD16, int ESD8, local_data_t *&local_data, Pool_hd_t *p) {
    int *coeff = (int *)local_data;
    const int insert_pos = ((int *)local_data)[256];
    const int remove_pos = ((int *)local_data)[257];
    const int old_ESD = ((int *)local_data)[258];
    const int auto_lll = ((int *)local_data)[259];
    
    struct timeval lll_start, lll_stop;
    gettimeofday(&lll_start, NULL);
    VEC_QP v_QP = NEW_VEC_QP(p->basis->NumCols());
    MAT_QP b_QP = p->basis->get_b();
    MAT_QP b_trans_QP = min_lift_traits::b_trans_QP(p);

    int CSD = p->CSD;
    int ESD = p->ESD;
    int bias = p->index_l - p->ESD;

    for (int i = 0; i < ESD + CSD; i++) 
        copy(b_QP.hi[bias + i], b_QP.lo[bias + i], b_trans_QP.hi[i], b_trans_QP.lo[i], p->basis->NumCols());

    FREE_MAT_QP(b_trans_QP);

    for (int j = 0; j < ESD; j++) {
        red(v_QP.hi, v_QP.lo, b_QP.hi[bias + j], b_QP.lo[bias + j], NTL::quad_float(-coeff[j]), p->basis->NumCols());
    }
    for (int j = 0; j < CSD; j++) {
        red(v_QP.hi, v_QP.lo, b_QP.hi[p->index_l + j], b_QP.lo[p->index_l + j], NTL::quad_float(-coeff[ESD8 + j]), p->basis->NumCols());
    }
    for (long j = bias - 1; j >= 0; j--) {
        int32_t c = round(dot_avx2(v_QP.hi, p->basis->get_b_star().hi[j], p->basis->NumCols()) / p->basis->get_B().hi[j]);
        red(v_QP.hi, v_QP.lo, b_QP.hi[j], b_QP.lo[j], NTL::quad_float(c), p->basis->NumCols());
    }

    for (long i = remove_pos; i > insert_pos; i--) {
        copy(b_QP.hi[i], b_QP.lo[i], b_QP.hi[i-1], b_QP.lo[i-1], p->basis->NumCols());
    }
    copy(b_QP.hi[insert_pos], b_QP.lo[insert_pos], v_QP.hi, v_QP.lo, p->basis->NumCols());
    FREE_VEC_QP(v_QP);

    p->basis->compute_gso_QP();

    float **b_trans = (float **) NEW_MAT(CSD, vec_nbytes, sizeof(float));
    for (int j = 0; j < CSD - 1; j++) {
        float x = sqrt(p->basis->get_B().hi[j + p->index_l + 1]);
        for (int i = 0; i < CSD; i++) {
            if (i < remove_pos - p->index_l)
                b_trans[i][j] = p->basis->get_miu().hi[i + p->index_l + 1][j + p->index_l + 1] * x;
            if (i > remove_pos - p->index_l)
                b_trans[i][j] = p->basis->get_miu().hi[i + p->index_l + 0][j + p->index_l + 1] * x;
        }
        for (int i = 0; i < CSD; i++) {
            if (i == remove_pos - p->index_l) continue;
            b_trans[remove_pos - p->index_l][j] -= b_trans[i][j] * coeff[ESD8 + i];
        }
    }
    free(local_data);
    p->basis->size_reduce();
    if (auto_lll) {
        Lattice_QP *L_tmp = p->basis->b_loc_QP(p->index_l + 1, p->index_r);
        L_tmp->LLL_QP();
        int deep_ind_l = L_tmp->NumRows() >= 90 ? (L_tmp->NumRows() - 90) : 0;
        int deep_ind_r = L_tmp->NumRows();
        L_tmp->LLL_DEEP_QP(0.97, deep_ind_l, deep_ind_r);
        L_tmp->LLL_QP();
        Lattice_QP *L_tmp_dual = L_tmp->dual_QP();
        p->basis->trans_to(p->index_l + 1, p->index_r, L_tmp);
        p->basis->compute_gso_QP();
        p->basis->size_reduce();
        delete L_tmp;

        Lattice_QP *L_normal = p->basis->b_loc_QP(p->index_l + 1, p->index_r);

        double max_fperr = 0.0;
        for (int i = 0; i < CSD; i++) {
            double c[vec_nbytes] = {};
            double v[vec_nbytes] = {};
            for (int j = 0; j < CSD - 1; j++) {
                for (int l = 0; l < CSD - 1; l++) c[j] += L_tmp_dual->get_b().hi[j][l] * b_trans[i][l];
                max_fperr = fmax(max_fperr, fabs(c[j] - round(c[j])));
                c[j] = round(c[j]);
                for (int l = 0; l < CSD - 1; l++) v[l] += c[j] * L_normal->get_b().hi[j][l];
            }
            for (int l = 0; l < CSD - 1; l++) b_trans[i][l] = v[l];
        }

        delete L_normal;
        delete L_tmp_dual;
    }
    gettimeofday(&lll_stop, NULL);
    #if ENABLE_PROFILING
    p->logger->ev_init_time.tv_sec += lll_stop.tv_sec - lll_start.tv_sec;
    p->logger->ev_init_time.tv_usec += lll_stop.tv_usec - lll_start.tv_usec;
    double lll_time = lll_stop.tv_sec - lll_start.tv_sec + 1e-6 * (lll_stop.tv_usec - lll_start.tv_usec);
    p->logger->info("insert: lll time = %.3fs", lll_time);
    #endif

    int8_t *_b_dual_old = p->_b_dual;
    int32_t _dhalf_old = p->_dhalf;
    int32_t _dshift_old = p->_dshift;
    p->_b_dual = NULL;

    p->index_l++;
    p->CSD--;
    p->_update_b_local();
    p->set_boost_depth(old_ESD);
    p->uid_table->reset_hash_function(p->CSD);

    for (int i = 0; i < CSD; i++) {
        for (int j = 0; j < CSD; j++) b_trans[i][j] *= p->_ratio;
    }

    float max_fperr = 0.0f;
    uint64_t u_trans[vec_nbytes] = {};
    float **e_trans = (float **) NEW_MAT(CSD, p->ESD, sizeof(float));
    for (int i = 0; i < CSD; i++) {
        __attribute__ ((aligned (64))) float tmp[256];
        copy(tmp, b_trans[i], vec_nbytes);
        for (int j = CSD - 2; j >= 0; j--) {
            float c = roundf(tmp[j] / p->_b_local[j][j]);
            max_fperr = fmaxf(fabsf(c - tmp[j] / p->_b_local[j][j]), max_fperr);
            red(tmp, p->_b_local[j], c, vec_nbytes);
            u_trans[i] += p->uid_table->coeff(j) * (int)c;
            for (int l = 0; l < p->ESD; l++) e_trans[i][l] += p->_boost_data->evec[(p->ESD + j) * p->ESD + l] * c;
        }
    }
    if (max_fperr > 0.1) {
        fprintf(stderr, "[Error] insert_traits::_prep_device_local_data: "
                        "large floating point error(%.2f)\n", max_fperr);
    } else if (max_fperr > 0.01) {
        fprintf(stderr, "[Warning] insert_traits::_prep_device_local_data: "
                        "floating point error(%.2f) warning\n", max_fperr);
    }


    local_data_t *host_local_data = (local_data_t *) calloc(1, sizeof(local_data_t));
    host_local_data->CSD = p->CSD;
    host_local_data->ESD = p->ESD;
    host_local_data->dhalf = _dhalf_old;
    host_local_data->dshift = _dshift_old;

    CSD = p->CSD;
    ESD = p->ESD;

    for (int i = ESD; i < ESD8; i++) host_local_data->igh[i]        = 0x1p20f;
    for (int i = 0; i < ESD; i++) host_local_data->igh[i]           = p->_boost_data->igh[i];
    for (int i = 0; i < ESD; i++) host_local_data->inorm[i]         = p->_boost_data->inorm[i];
    for (int i = 0; i < CSD + 1; i++) host_local_data->uid_coeff[i] = u_trans[i];
    
    int8_t *ip = host_local_data->b_dual;
    for (int row = CSD16 - 16; row >= 0; row -= 16) {
        for (int col = row; col < CSD16; col += 16) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 16; j++) *ip++ = (i < CSD + 1 && j < CSD + 1) ? _b_dual_old[i * vec_nbytes + j] : 0;
            }
        }
    }

    float *fp = host_local_data->b_full;
    for (int row = CSD16 - 16; row >= 0; row -= 16) {
        for (int col = 0; col < ESD8; col += 8) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 8; j++) {
                    *fp++ = (i < CSD + 1 && j < ESD) ? e_trans[i][j] : 0.0f;
                }
            }
        }
        for (int col = 0; col < CSD16; col += 16) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 8; j++) {
                    *fp++ = (i < CSD + 1 && j < CSD) ? b_trans[i][j] : 0.0f;
                }
            }

            for (int i = row; i < row + 16; i++) {
                for (int j = col + 8; j < col + 16; j++) {
                    *fp++ = (i < CSD + 1 && j < CSD) ? b_trans[i][j] : 0.0f;
                }
            }
        }
    }

    FREE_MAT(b_trans);
    FREE_MAT(e_trans);
    FREE_VEC((void *)_b_dual_old);

    float *ep = host_local_data->b_ext_head;
    for (int i = ESD8 - 1; i >= 0; i--) {
        for (int j = 0; j < ((i + 2) & ~1); j++) {
            *ep++ = (i < ESD && j < ESD) ? p->_boost_data->evec[i * ESD + j] : 0.0f;
        }
    }
    
    CHECK_CUDA_ERR(cudaMalloc(&local_data, sizeof(local_data_t)));
    CHECK_CUDA_ERR(cudaMemcpy(local_data, host_local_data, sizeof(local_data_t), cudaMemcpyHostToDevice));
    free(host_local_data);
}

template <>
void dim_lose_traits::_prep_device_local_data(int CSD16, int ESD8, local_data_t *&local_data, Pool_hd_t *p) {
    local_data_t *host_local_data = (local_data_t *) calloc(1, sizeof(local_data_t));
    host_local_data->CSD = p->CSD;
    host_local_data->ESD = p->ESD;
    host_local_data->dhalf = p->_dhalf;
    host_local_data->dshift = p->_dshift;

    const int CSD = p->CSD;

    int8_t *ip = host_local_data->b_dual;
    for (int row = CSD16 - 16; row >= 0; row -= 16) {
        for (int col = row; col < CSD16; col += 16) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 16; j++) *ip++ = (i < CSD && j < CSD) ? p->_b_dual[i * vec_nbytes + j] : 0;
            }
        }
    }

    CHECK_CUDA_ERR(cudaMalloc(&local_data, sizeof(local_data_t)));
    CHECK_CUDA_ERR(cudaMemcpy(local_data, host_local_data, sizeof(local_data_t), cudaMemcpyHostToDevice));
    free(host_local_data);
}

template <>
void (*const check_traits::kernel)(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data) = 
    check_kernel<check_traits::vec_nbytes, check_traits::max_boost_dim>;

template <>
void (*const extend_left_traits::kernel)(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data) = 
    extend_left_kernel<extend_left_traits::vec_nbytes, extend_left_traits::max_boost_dim>;

template <>
void (*const shrink_left_traits::kernel)(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data) = 
    check_kernel<shrink_left_traits::vec_nbytes, shrink_left_traits::max_boost_dim>;

template <>
void (*const min_lift_traits::kernel)(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data) = 
    min_lift_kernel<min_lift_traits::vec_nbytes, min_lift_traits::max_boost_dim>;

template <>
void (*const insert_traits::kernel)(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data) = 
    insert_kernel<insert_traits::vec_nbytes, insert_traits::max_boost_dim>;

template <>
void (*const dim_lose_traits::kernel)(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data) = 
    dim_lose_kernel<dim_lose_traits::vec_nbytes>;


template <class traits> 
int Pool_hd_t::stream_task_template(int num_devices, cudaDeviceProp device_props[], local_data_t **local_data) {
    #if ENABLE_PROFILING
    logger->ev_total_chunks = pwc_manager->num_chunks();
    logger->num_thread = this->_num_threads;
    #endif

    for (int i = 0; i < 65536; i++) score_stat[i] = 0U;
    pthread_spinlock_t score_stat_lock;
    pthread_spin_init(&score_stat_lock, PTHREAD_PROCESS_SHARED);
    ut_checker_t ut_checker(ut_checker_t::type_others, this, uid_table, pwc_manager);
    ut_checker.set_score_stat(score_stat, score_stat_lock);

    int old_CSD = CSD;
    {
        CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[0]));
        local_data_t *h_local_data;
        CHECK_CUDA_ERR(cudaMalloc(&h_local_data, sizeof(local_data_t)));
        traits::prep_device_local_data(local_data[0], this);
        CHECK_CUDA_ERR(cudaMemcpy(h_local_data, local_data[0], sizeof(local_data_t), cudaMemcpyHostToDevice));
        for (int i = 1; i < num_devices; i++) {
            CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[i]));
            CHECK_CUDA_ERR(cudaMalloc(&local_data[i], sizeof(local_data_t)));
            CHECK_CUDA_ERR(cudaMemcpy(local_data[i], h_local_data, sizeof(local_data_t), cudaMemcpyDeviceToDevice));
        }
        CHECK_CUDA_ERR(cudaFree(h_local_data));
    }
    constexpr int taskChunks = traits::taskChunks;
    constexpr int taskVecs = traits::taskVecs;
    
    const int total_tasks = (pwc_manager->num_chunks() + taskChunks - 1) / taskChunks;
    
    #pragma omp parallel for num_threads(_num_threads)
    for (long thread = 0; thread < _num_threads; thread++) {
        const int device_ptr = hw::gpu_ptr(thread, _num_threads);
        const int begin_ind = (thread * total_tasks) / _num_threads;
        const int end_ind = ((thread + 1) * total_tasks) / _num_threads;

        CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[device_ptr]));

        traits::init_shared_mem_limit();

        cudaStream_t stream;
        typename traits::stream_with_l2_holder_t 
        l2_holder(stream, local_data[device_ptr]->b_full, traits::b_full_active_nbytes);

        int8_t *h_buffer, *d_buffer, *pack_buffer;
        uint16_t *h_buffer_score; int32_t *h_buffer_norm; uint64_t *h_buffer_u;
        typename traits::pool_hd_buffer_holder_t 
        buffer_holder(stream, d_buffer, pack_buffer, h_buffer, h_buffer_score, h_buffer_norm, h_buffer_u);

        #if ENABLE_PROFILING
        cudaEvent_t ev[taskChunks * 2 + 5];
        for (int i = 0; i < taskChunks * 2 + 5; i++) CHECK_CUDA_ERR(cudaEventCreate(&ev[i]));
        #endif

        for (int i = 0; i < taskChunks && end_ind > begin_ind; i++) 
            pwc_manager->prefetch(begin_ind * taskChunks + i);

        for (int ind = begin_ind; ind < end_ind; ind++) {
            uint64_t ld_stall_us = 0;
            int task_chunks, task_vecs = 0;
            if (ind != total_tasks - 1) {
                for (int i = 0; i < taskChunks && ind + 1 < end_ind; i++) pwc_manager->prefetch((ind + 1) * taskChunks + i);
                task_chunks = taskChunks;
            } else {
                task_chunks = pwc_manager->num_chunks() - ind * taskChunks;
            }

            chunk_t *working_chunk[taskChunks];
            for (int i = 0; i < task_chunks; i++) {
                struct timeval ld_start, ld_end;
                gettimeofday(&ld_start, NULL);
                working_chunk[i] = pwc_manager->fetch(ind * taskChunks + i);
                gettimeofday(&ld_end, NULL);
                ld_stall_us += (ld_end.tv_sec - ld_start.tv_sec) * 1000000ULL + 
                                (ld_end.tv_usec - ld_start.tv_usec);
                if (working_chunk[i] == NULL) continue;

                _normalize_chunk(working_chunk[i], old_CSD);
                #if ENABLE_PROFILING
                CHECK_CUDA_ERR(cudaEventRecord(ev[i*2], stream));
                #endif
                CHECK_CUDA_ERR(cudaMemcpyAsync(pack_buffer + old_CSD * task_vecs, working_chunk[i]->vec, 
                            working_chunk[i]->size * old_CSD, cudaMemcpyHostToDevice, stream));
                #if ENABLE_PROFILING
                CHECK_CUDA_ERR(cudaEventRecord(ev[i*2+1], stream));
                #endif
                task_vecs += working_chunk[i]->size;
            }
            
            #if ENABLE_PROFILING
            CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2+0], stream));
            utils_t::device_unpackf(stream, d_buffer, pack_buffer, old_CSD, task_vecs);
            CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2+1], stream));
            traits::launch(stream, d_buffer, task_vecs, local_data[device_ptr]);
            CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2+2], stream));
            utils_t::device_packf(stream, pack_buffer, d_buffer, CSD, task_vecs);
            CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2+3], stream));
            int real_i = 0;
            for (int i = 0; i < task_chunks; i++) {
                if (working_chunk[i] == NULL) continue;
                int nn = (real_i + 1) * chunk_max_nvecs > task_vecs ? 
                        task_vecs - real_i * chunk_max_nvecs : chunk_max_nvecs;
                if (nn <= 0) break;
                CHECK_CUDA_ERR(cudaMemcpyAsync(working_chunk[i]->vec, pack_buffer + real_i * chunk_max_nvecs * CSD, 
                            nn * CSD, cudaMemcpyDeviceToHost, stream));
                real_i++;
            }
            CHECK_CUDA_ERR(cudaMemcpyAsync(h_buffer_score, d_buffer + taskVecs * vec_nbytes, taskVecs * 14,
                        cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2+4], stream));
            CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
            float x, y, z, w;
            for (int i = 0; i < task_chunks; i++) {
                if (working_chunk[i] == NULL) continue;
                CHECK_CUDA_ERR(cudaEventElapsedTime(&x, ev[2*i+0], ev[2*i+1]));
                logger->ev_h2d_us += (uint64_t) round(x * 1e3);
            }
            CHECK_CUDA_ERR(cudaEventElapsedTime(&x, ev[2*task_chunks+0], ev[2*task_chunks+1]));
            CHECK_CUDA_ERR(cudaEventElapsedTime(&y, ev[2*task_chunks+1], ev[2*task_chunks+2]));
            CHECK_CUDA_ERR(cudaEventElapsedTime(&z, ev[2*task_chunks+2], ev[2*task_chunks+3]));
            CHECK_CUDA_ERR(cudaEventElapsedTime(&w, ev[2*task_chunks+3], ev[2*task_chunks+4]));
            logger->ev_unpack_us    += (uint64_t) round(x * 1e3);
            logger->ev_kernel_us    += (uint64_t) round(y * 1e3);
            logger->ev_pack_us      += (uint64_t) round(z * 1e3);
            logger->ev_d2h_us       += (uint64_t) round(w * 1e3);
            logger->ev_h2d_nbytes   += CSD * task_vecs;
            logger->ev_d2h_nbytes   += (CSD + 14) * task_vecs;
            logger->ev_vec_nbytes   += CSD * task_vecs;
            logger->ev_curr_chunks  += task_chunks;
            logger->ev_ld_stall_us  += ld_stall_us;
            #else
            utils_t::device_unpackf(stream, d_buffer, pack_buffer, old_CSD, task_vecs);
            traits::launch(stream, d_buffer, task_vecs, local_data[device_ptr]);
            utils_t::device_packf(stream, pack_buffer, d_buffer, CSD, task_vecs);
            int real_i = 0;
            for (int i = 0; i < task_chunks; i++) {
                if (working_chunk[i] == NULL) continue;
                int nn = (real_i + 1) * chunk_max_nvecs > task_chunks ? 
                        task_vecs - real_i * chunk_max_nvecs : chunk_max_nvecs;
                if (nn <= 0) break;
                CHECK_CUDA_ERR(cudaMemcpyAsync(working_chunk[i]->vec, pack_buffer + real_i * chunk_max_nvecs * CSD, 
                            nn * CSD, cudaMemcpyDeviceToHost, stream));
                real_i++;
            }
            CHECK_CUDA_ERR(cudaMemcpyAsync(h_buffer_score, d_buffer + taskVecs * vec_nbytes, taskVecs * 14,
                        cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
            #endif

            int num_used = 0;
            for (int i = 0; i < task_chunks; i++) {
                chunk_t *chunk = working_chunk[i];
                if (chunk == NULL) continue;
                WRITE_BACK_TO_CHUNK(chunk);            
            }
        }

        #if ENABLE_PROFILING
        for (int i = 0; i < taskChunks * 2 + 5; i++) CHECK_CUDA_ERR(cudaEventDestroy(ev[i]));
        #endif
    }
    ut_checker.input_done();

    ut_checker.wait_work();
    pthread_spin_destroy(&score_stat_lock);

    return 0;
}

template <class traits>
int Pool_hd_t::stream_stat_template(int num_devices, cudaDeviceProp device_props[], local_data_t **local_data) {
    #if ENABLE_PROFILING
    logger->ev_total_chunks = pwc_manager->num_chunks();
    logger->num_thread = this->_num_threads;
    #endif

    for (int i = 0; i < num_devices; i++) {
        CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[i]));
        traits::prep_device_local_data(local_data[i], this);
    }
    constexpr int taskChunks = traits::taskChunks;
    
    const int total_tasks = (pwc_manager->num_chunks() + taskChunks - 1) / taskChunks;
    
    #pragma omp parallel for num_threads(_num_threads)
    for (long thread = 0; thread < _num_threads; thread++) {
        const int device_ptr = hw::gpu_ptr(thread, _num_threads);
        const int begin_ind = (thread * total_tasks) / _num_threads;
        const int end_ind = ((thread + 1) * total_tasks) / _num_threads;

        CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[device_ptr]));

        traits::init_shared_mem_limit();

        cudaStream_t stream;
        typename traits::stream_with_l2_holder_t 
        l2_holder(stream, local_data[device_ptr]->b_full, traits::b_full_active_nbytes);

        int8_t *h_buffer, *d_buffer, *pack_buffer;
        uint16_t *h_buffer_score; int32_t *h_buffer_norm; uint64_t *h_buffer_u;
        typename traits::pool_hd_buffer_holder_t 
        buffer_holder(stream, d_buffer, pack_buffer, h_buffer, h_buffer_score, h_buffer_norm, h_buffer_u);

        #if ENABLE_PROFILING
        cudaEvent_t ev[2 * taskChunks + 3];
        for (int i = 0; i < 2 * taskChunks + 3; i++) CHECK_CUDA_ERR(cudaEventCreate(&ev[i]));
        #endif

        for (int i = 0; i < taskChunks && end_ind > begin_ind; i++) 
            pwc_manager->prefetch(begin_ind * taskChunks + i);

        for (int ind = begin_ind; ind < end_ind; ind++) {
            uint64_t ld_stall_us = 0;
            int task_chunks, task_vecs = 0;
            if (ind != total_tasks - 1) {
                for (int i = 0; i < taskChunks && ind + 1 < end_ind; i++) pwc_manager->prefetch((ind + 1) * taskChunks + i);
                task_chunks = taskChunks;
            } else {
                task_chunks = pwc_manager->num_chunks() - ind * taskChunks;
            }

            chunk_t *working_chunk[taskChunks];
            int chunk_modified[taskChunks] = {};
            for (int i = 0; i < task_chunks; i++) {
                struct timeval ld_start, ld_end;
                gettimeofday(&ld_start, NULL);
                working_chunk[i] = pwc_manager->fetch(ind * taskChunks + i);
                gettimeofday(&ld_end, NULL);
                ld_stall_us += (ld_end.tv_sec - ld_start.tv_sec) * 1000000ULL + 
                                (ld_end.tv_usec - ld_start.tv_usec);
                
                if (working_chunk[i] == NULL) continue;

                chunk_modified[i] = _normalize_chunk(working_chunk[i], CSD);

                #if ENABLE_PROFILING
                CHECK_CUDA_ERR(cudaEventRecord(ev[i*2], stream));
                CHECK_CUDA_ERR(cudaMemcpyAsync(pack_buffer + CSD * task_vecs, working_chunk[i]->vec, 
                            working_chunk[i]->size * CSD, cudaMemcpyHostToDevice, stream));
                CHECK_CUDA_ERR(cudaEventRecord(ev[i*2+1], stream));
                #else 
                CHECK_CUDA_ERR(cudaMemcpyAsync(pack_buffer + vec_nbytes * task_vecs, working_chunk[i]->vec, 
                            working_chunk[i]->size * CSD, cudaMemcpyHostToDevice, stream));
                #endif
                task_vecs += working_chunk[i]->size;
            }
            cudaStreamSynchronize(stream);
            for (int i = 0; i < task_chunks; i++) {
                if (working_chunk[i]) {
                    if (chunk_modified[i]) pwc_manager->release_sync(working_chunk[i]->id);
                    else pwc_manager->release(working_chunk[i]->id);
                }
            }

            #if ENABLE_PROFILING
            float x, y;
            if (ind != begin_ind) {
                CHECK_CUDA_ERR(cudaEventElapsedTime(&x, ev[taskChunks*2+0], ev[taskChunks*2+1]));
                CHECK_CUDA_ERR(cudaEventElapsedTime(&y, ev[taskChunks*2+1], ev[taskChunks*2+2]));
                logger->ev_unpack_us += (uint64_t) round(x * 1e3);
                logger->ev_kernel_us += (uint64_t) round(y * 1e3);
            }

            CHECK_CUDA_ERR(cudaEventRecord(ev[taskChunks*2+0], stream));
            utils_t::device_unpackf(stream, d_buffer, pack_buffer, CSD, task_vecs);
            CHECK_CUDA_ERR(cudaEventRecord(ev[taskChunks*2+1], stream));
            traits::launch(stream, d_buffer, task_vecs, local_data[device_ptr]);
            CHECK_CUDA_ERR(cudaEventRecord(ev[taskChunks*2+2], stream));
            for (int i = 0; i < task_chunks; i++) {
                if (working_chunk[i] == NULL) continue;
                CHECK_CUDA_ERR(cudaEventElapsedTime(&x, ev[i*2], ev[i*2+1]));
                logger->ev_h2d_us  += (uint64_t) round(x * 1e3);
            }
            logger->ev_ld_stall_us += ld_stall_us;
            logger->ev_vec_nbytes  += CSD * task_vecs;
            logger->ev_h2d_nbytes  += CSD * task_vecs;
            logger->ev_curr_chunks += task_chunks;
            #else
            utils_t::device_unpackf(stream, d_buffer, pack_buffer, CSD, task_vecs);
            traits::launch(stream, d_buffer, task_vecs, local_data[device_ptr]);
            #endif
        }

        CHECK_CUDA_ERR(cudaStreamSynchronize(stream));

        #if ENABLE_PROFILING
        if (end_ind > begin_ind) {
            float x, y;
            CHECK_CUDA_ERR(cudaEventElapsedTime(&x, ev[taskChunks*2+0], ev[taskChunks*2+1]));
            CHECK_CUDA_ERR(cudaEventElapsedTime(&y, ev[taskChunks*2+1], ev[taskChunks*2+2]));
            logger->ev_unpack_us  += (uint64_t) round(x * 1e3);
            logger->ev_kernel_us  += (uint64_t) round(y * 1e3);
        }
        
        for (int i = 0; i < 2 * taskChunks + 3; i++) CHECK_CUDA_ERR(cudaEventDestroy(ev[i]));
        #endif
    }

    return 0;
}

int Pool_hd_t::extend_left() {
    lg_init();
    if (this->index_l == 0) {
        lg_warn("index_l = 0, cannot extend_left, nothing done.");
        lg_exit();
        return -2;
    }

    pwc_manager->wait_work();
    if (CSD > 120) {
        long target_cached_chunks = PWC_DEFAULT_MAX_CACHED_CHUNKS + 
                                    BWC_DEFAULT_MAX_CACHED_CHUNKS +
                                    SWC_DEFAULT_MAX_CACHED_CHUNKS;
        if (pwc_manager->max_cached_chunks() != target_cached_chunks) {
            pwc_manager->set_max_cached_chunks(target_cached_chunks);
        }
    }

    // check for device
    int num_devices;
    cudaDeviceProp device_props[MAX_NUM_DEVICE];
    if (_prepare_device_prop(num_devices, device_props)) {
        lg_err("no device detected, nothing done.");
        lg_exit();
        return -1;
    }

    local_data_t *local_data[MAX_NUM_DEVICE];
    stream_task_template<extend_left_traits>(num_devices, device_props, local_data);
    for (int i = 0; i < num_devices; i++) CHECK_CUDA_ERR(cudaFree(local_data[i]));

    lg_report();
    lg_exit();

    return 0;
}

int Pool_hd_t::shrink_left() {
    lg_init();

    pwc_manager->wait_work();
    if (CSD > 120) {
        long target_cached_chunks = PWC_DEFAULT_MAX_CACHED_CHUNKS + 
                                    BWC_DEFAULT_MAX_CACHED_CHUNKS +
                                    SWC_DEFAULT_MAX_CACHED_CHUNKS;
        if (pwc_manager->max_cached_chunks() != target_cached_chunks) {
            pwc_manager->set_max_cached_chunks(target_cached_chunks);
        }
    }

    // check for device
    int num_devices;
    cudaDeviceProp device_props[MAX_NUM_DEVICE];
    if (_prepare_device_prop(num_devices, device_props)) {
        lg_err("no device detected, nothing done.");
        lg_exit();
        return -1;
    }

    local_data_t *local_data[MAX_NUM_DEVICE];
    stream_task_template<shrink_left_traits>(num_devices, device_props, local_data);
    for (int i = 0; i < num_devices; i++) CHECK_CUDA_ERR(cudaFree(local_data[i]));

    lg_report();
    lg_exit();

    return 0;
}

int Pool_hd_t::show_min_lift(long index) {
    lg_init();

    if (index > index_l) {
        lg_warn("index(%ld) > index_l(%ld), nothing done.", index, index_l);
        lg_exit();
        return -2;
    }
    if (index >= 0 && index_l - index > boost_data_t::max_boost_dim) {
        lg_warn("index(%ld) < index_l(%ld) - 48, nothing done.", index, index_l);
    }
    if (index < 0 && index < -boost_data_t::max_boost_dim) {
        lg_warn("out of range(%ld), nothing done.", index);
        lg_exit();
        return -2;
    }

    // check for device
    int num_devices;
    cudaDeviceProp device_props[MAX_NUM_DEVICE];
    if (_prepare_device_prop(num_devices, device_props)) {
        lg_err("no device detected, nothing done.");
        lg_exit();
        return -1;
    }

    pwc_manager->wait_work();

    const int old_ESD = ESD;
    const int new_ESD = index >= 0 ? index_l - index : -index;

    this->set_boost_depth(new_ESD);

    typedef min_lift_traits traits;

    local_data_t *local_data[MAX_NUM_DEVICE];
    for (int i = 0; i < num_devices; i++) local_data[i] = (local_data_t *)calloc(1, 4);
    stream_stat_template<traits>(num_devices, device_props, local_data);

    int *res[MAX_NUM_DEVICE];
    for (int i = 0; i < num_devices; i++) {
        res[i] = (int *) malloc(traits::max_boost_dim * (1024 + 4 + 4));
        CHECK_CUDA_ERR(cudaMemcpy(res[i], local_data[i] + 1, 
            traits::max_boost_dim * (1024 + 4 + 4), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaFree(local_data[i]));
    }

    for (int l = 1; l < num_devices; l++) {
        for (int i = 0; i < ESD; i++) {
            if (res[l][i] < res[0][i]) {
                res[0][i] = res[l][i];
                memcpy(res[0] + traits::max_boost_dim * 2 + i * 256, res[l] + traits::max_boost_dim * 2 + i * 256, 1024);
            }
        }
    }


    VEC_QP v_QP = NEW_VEC_QP(basis->NumCols());
    MAT_QP b_QP = basis->get_b();
    MAT_QP b_trans_QP = traits::b_trans_QP(this);

    for (int i = 0; i < ESD; i++) {
        int pos = index_l - ESD + i;
        if ((index >= 0 && index != pos) || (index < 0 && index_l - pos > -index)) continue;
        float old_scaled_norm = _boost_data->evec[i * ESD + i] * _boost_data->evec[i * ESD + i];
        float new_scaled_norm = ((float *)res[0])[i];
        if (new_scaled_norm < 0.9995f * old_scaled_norm || 1) {
            int *coeff = &res[0][traits::max_boost_dim * 2 + i * 256];
            int has_one = 0;
            for (int j = 48; j < 48 + CSD; j++) if (coeff[j] == 1 || coeff[j] == -1) has_one = 1;
            for (int j = 0; j < basis->NumCols(); j++) {
                v_QP.hi[j] = 0.0;
                v_QP.lo[j] = 0.0;
            }
            for (int j = 0; j < ESD; j++) {
                red(v_QP.hi, v_QP.lo, b_trans_QP.hi[j], b_trans_QP.lo[j], NTL::quad_float(-coeff[j]), basis->NumCols());
            }
            for (int j = 0; j < CSD; j++) {
                red(v_QP.hi, v_QP.lo, b_trans_QP.hi[ESD+j], b_trans_QP.lo[ESD+j], NTL::quad_float(-coeff[traits::max_boost_dim + j]), basis->NumCols());
            }

            for (long j = index_l - ESD - 1; j >= 0; j--) {
                int32_t c = round(dot_avx2(v_QP.hi, basis->get_b_star().hi[j], basis->NumCols()) / basis->get_B().hi[j]);
                red(v_QP.hi, v_QP.lo, b_QP.hi[j], b_QP.lo[j], NTL::quad_float(c), basis->NumCols());
            }

            printf("%s[pos %d] length = %.2f(%.3f gh, %.3f old)\033[0m, vec = [", has_one ? "" : "\033[33m", pos, sqrt(new_scaled_norm) / _ratio, 
                    sqrt(new_scaled_norm) / _ratio / basis->gh(pos, index_r), sqrt(new_scaled_norm / old_scaled_norm));
            fflush(stdout);
            for (int j = 0; j < basis->NumCols(); j++) {
                std::cout << NTL::quad_float(v_QP.hi[j], v_QP.lo[j]);
                if (j < basis->NumCols() - 1) std::cout << " ";
                else std::cout << "]\n";
            }
            std::cout.flush();
        }
    }

    this->set_boost_depth(old_ESD);

    for (int i = 0; i < num_devices; i++) free(res[i]);
    FREE_VEC_QP(v_QP);
    FREE_MAT_QP(b_trans_QP);

    lg_report();
    lg_exit();

    return 0;
}

int Pool_hd_t::insert(long index, double eta, long *pos, long auto_lll) {
    lg_init();
    if (index < 0 || index > index_l || index < index_l - boost_data_t::max_boost_dim) {
        lg_warn("index(%ld) out of range [%ld, %ld], nothing done.\n",
                        index, std::max(0L, index_l - boost_data_t::max_boost_dim), index_l);
        lg_exit();
        return -2;
    }

    // check for device
    int num_devices;
    cudaDeviceProp device_props[MAX_NUM_DEVICE];
    if (_prepare_device_prop(num_devices, device_props)) {
        lg_err("no device detected, nothing done.");
        lg_exit();
        return -1;
    }

    pwc_manager->wait_work();
    if (CSD > 120) {
        long target_cached_chunks = PWC_DEFAULT_MAX_CACHED_CHUNKS + 
                                    BWC_DEFAULT_MAX_CACHED_CHUNKS +
                                    SWC_DEFAULT_MAX_CACHED_CHUNKS;
        if (pwc_manager->max_cached_chunks() != target_cached_chunks) {
            pwc_manager->set_max_cached_chunks(target_cached_chunks);
        }
    }

    const int old_ESD = ESD;
    const int new_ESD = index_l - index;

    this->set_boost_depth(new_ESD);

    local_data_t *local_data[MAX_NUM_DEVICE];
    for (int i = 0; i < num_devices; i++) {
        local_data[i] = (local_data_t *) malloc(4);
        *((int *)local_data[i]) = 1;
    }
    stream_stat_template<min_lift_traits>(num_devices, device_props, local_data);

    lg_report();
    lg_info("min_lift search done");
    #if ENABLE_PROFILING
    logger->clear();
    #endif

    int *res[MAX_NUM_DEVICE];
    for (int i = 0; i < num_devices; i++) {
        res[i] = (int *) malloc(min_lift_traits::max_boost_dim * (1024 + 4 + 4));
        CHECK_CUDA_ERR(cudaMemcpy(res[i], local_data[i] + 1, 
            min_lift_traits::max_boost_dim * (1024 + 4 + 4), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaFree(local_data[i]));
    }

    for (int l = 1; l < num_devices; l++) {
        for (int i = 0; i < ESD; i++) {
            if (res[l][i] < res[0][i]) {
                res[0][i] = res[l][i];
                memcpy(res[0] + min_lift_traits::max_boost_dim * 2 + i * 256, res[l] + min_lift_traits::max_boost_dim * 2 + i * 256, 1024);
            }
        }
    }
    

    float best_ratio = 0x1p30f;
    int remove_pos = -1, insert_pos = -1;
    for (int i = 0; i < ESD; i++) {
        float ratio = ((float *)res[0])[i] / (_boost_data->evec[i * ESD + i] * _boost_data->evec[i * ESD + i]);
        if (ratio * pow(eta, i) < best_ratio && (ratio < 0.9995f)) {
            best_ratio = ratio;
            insert_pos = index_l - ESD + i;
        }
    }

    for (int i = CSD - 1; i >= 0 && insert_pos != -1; i--) {
        int *coeff = &res[0][2 * min_lift_traits::max_boost_dim + (insert_pos - index_l + ESD) * 256];
        if (coeff[boost_data_t::max_boost_dim + i] == 1) {
            remove_pos = index_l + i;
            break;
        }
        if (coeff[boost_data_t::max_boost_dim + i] == -1) {
            remove_pos = index_l + i;
            for (int j = 0; j < 256; j++) coeff[j] = -coeff[j];
            break;
        }
    }

    if (insert_pos == -1 || remove_pos == -1) {
        if (insert_pos != -1 && remove_pos == -1) 
            lg_err("all coeff is not 1 or -1, please check");
        for (int i = 0; i < num_devices; i++) free(res[i]);
        this->set_boost_depth(old_ESD);
        this->_update_boost_data();
        if (pos) pos[0] = -1;
        lg_exit();
        this->shrink_left();
        return -1;
    }
    lg_info("insert_pos = %d, remove_pos = %d", insert_pos, remove_pos);
    if (pos) pos[0] = insert_pos;

    typedef insert_traits traits;

    for (int i = 0; i < num_devices; i++) {
        local_data[i] = (local_data_t *) calloc(256 + 4, 4);
        ((int *)local_data[i])[256] = insert_pos;
        ((int *)local_data[i])[257] = remove_pos;
        ((int *)local_data[i])[258] = old_ESD;
        ((int *)local_data[i])[259] = auto_lll;
        for (int j = 0; j < 256; j++) ((int *)local_data[i])[j] = res[0][2 * traits::max_boost_dim + (insert_pos - index_l + ESD) * 256 + j];
    }
    for (int i = 0; i < num_devices; i++) free(res[i]);

    pwc_manager->wait_work();

    stream_task_template<traits>(num_devices, device_props, local_data);
    for (int i = 0; i < num_devices; i++) CHECK_CUDA_ERR(cudaFree(local_data[i]));

    lg_report();
    lg_exit();

    return 0;
}

int Pool_hd_t::check_dim_lose() {
    lg_init();

    // check for device
    int num_devices;
    cudaDeviceProp device_props[MAX_NUM_DEVICE];
    if (_prepare_device_prop(num_devices, device_props)) {
        lg_err("no device detected, nothing done.");
        lg_exit();
        return -1;
    }

    local_data_t *local_data[MAX_NUM_DEVICE];
    stream_stat_template<dim_lose_traits>(num_devices, device_props, local_data);

    uint64_t res[vec_nbytes] = {};
    uint64_t tmp[vec_nbytes];
    for (int i = 0; i < num_devices; i++) {
        CHECK_CUDA_ERR(cudaMemcpy(tmp, local_data[i]->uid_coeff, 8 * vec_nbytes, cudaMemcpyDeviceToHost));
        for (int j = 0; j < vec_nbytes; j++) res[j] += tmp[j];
        CHECK_CUDA_ERR(cudaFree(local_data[i]));
    }

    int ret = 0;
    long num_vec = pwc_manager->num_vec();
    for (long i = 0; i < CSD; i++) {
        if (res[i] == 0) {
            ret = -1;
            printf("%ld-th basis vec definitely lost\n", i);
        } else if (res[i] * 100 < num_vec) {
            ret = -1;
            printf("%ld-th basis vec may lost, %ld/%ld nonzero\n", i, res[i], num_vec);
        }
    }

    lg_report();
    lg_exit();
    
    return ret;
}

int pop_check_vec_err(int8_t *src1, int8_t *src2, int CSD);

int Pool_hd_t::check(int log_level) {
    #define LOG(level, fmt, ...) if (log_level >= level) { \
        const char* color = ""; \
        switch (level) { \
            case 1: color = "\033[31m"; break; /* Red for ERROR */ \
            case 2: color = "\033[33m"; break; /* Yellow for WARN */ \
            case 3: color = "\033[0m"; break; /* Green for INFO */ \
            default: color = "\033[0m"; break; /* Reset */ \
        } \
        fprintf(stdout, "%s" fmt "\033[0m", color, ##__VA_ARGS__); \
    }

    constexpr int ll_info   = 3;
    constexpr int ll_warn   = 2;
    constexpr int ll_err    = 1;

    lg_init();

    int ret = 0;
    int64_t num_overflow = 0;       
    int64_t num_uid_notin = 0;      
    int64_t num_uid_err = 0;        
    int64_t vec_err_stat[4] = {};
    int64_t norm_err_stat[128] = {};
    int64_t score_err_stat[128] = {};
    int64_t *norm_stat = (int64_t *) calloc(65536 * 2, sizeof(int64_t));
    int64_t *score_stat64 = (int64_t *) calloc(65536, sizeof(int64_t));
    pthread_spinlock_t stat_lock;
    pthread_spin_init(&stat_lock, PTHREAD_PROCESS_SHARED);

    // check for device
    int num_devices;
    cudaDeviceProp device_props[MAX_NUM_DEVICE];
    if (_prepare_device_prop(num_devices, device_props)) {
        lg_err("no device detected, nothing done.");
        lg_exit();
        return -1;
    } 

    #if ENABLE_PROFILING
    logger->ev_total_chunks = pwc_manager->num_chunks();
    logger->num_thread = this->_num_threads;
    #endif

    ut_checker_t ut_checker(ut_checker_t::type_check, this, uid_table, NULL);

    typedef check_traits traits;

    local_data_t *local_data[MAX_NUM_DEVICE];
    for (int i = 0; i < num_devices; i++) {
        CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[i]));
        traits::prep_device_local_data(local_data[i], this);
    }
    constexpr int taskChunks = traits::taskChunks;
    constexpr int taskVecs = traits::taskVecs;
    
    const int total_tasks = (pwc_manager->num_chunks() + taskChunks - 1) / taskChunks;

    uint64_t rscore[65536] = {};

    #pragma omp parallel for num_threads(_num_threads) reduction(+:rscore[:65536])
    for (long thread = 0; thread < _num_threads; thread++) {
        int64_t _num_overflow = 0;
        int64_t _num_uid_err = 0;
        int64_t _vec_err_stat[4] = {};
        int64_t _norm_err_stat[128] = {};
        int64_t _score_err_stat[128] = {};
        int64_t *_norm_stat = (int64_t *) calloc(65536 * 2, sizeof(int64_t));
        int64_t *_score_stat64 = (int64_t *) calloc(65536, sizeof(int64_t));

        const int device_ptr = hw::gpu_ptr(thread, _num_threads);
        const int begin_ind = (thread * total_tasks) / _num_threads;
        const int end_ind = ((thread + 1) * total_tasks) / _num_threads;

        CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[device_ptr]));

        traits::init_shared_mem_limit();

        cudaStream_t stream;
        traits::stream_with_l2_holder_t 
        l2_holder(stream, local_data[device_ptr]->b_full, traits::b_full_active_nbytes);

        int8_t *h_buffer, *d_buffer, *pack_buffer;
        uint16_t *h_buffer_score; int32_t *h_buffer_norm; uint64_t *h_buffer_u;
        traits::pool_hd_buffer_holder_t 
        buffer_holder(stream, d_buffer, pack_buffer, h_buffer, h_buffer_score, h_buffer_norm, h_buffer_u);

        for (int i = 0; i < taskChunks; i++) pwc_manager->prefetch(begin_ind * taskChunks + i);

        chunk_t working_chunk[taskChunks];
        for (int i = 0; i < taskChunks; i++) _malloc_chunk(&working_chunk[i]);

        #if ENABLE_PROFILING
        cudaEvent_t ev[taskChunks * 2 + 5];
        for (int i = 0; i < taskChunks * 2 + 5; i++) CHECK_CUDA_ERR(cudaEventCreate(&ev[i]));
        #endif
        
        for (int ind = begin_ind; ind < end_ind; ind++) {
            uint64_t ld_stall_us = 0;
            int task_chunks, task_vecs = 0;
            if (ind != total_tasks - 1) {
                for (int i = 0; i < taskChunks && ind + 1 < end_ind; i++) pwc_manager->prefetch((ind + 1) * taskChunks + i);
                task_chunks = taskChunks;
            } else {
                task_chunks = pwc_manager->num_chunks() - ind * taskChunks;
            }

            for (int i = 0; i < task_chunks; i++) {
                struct timeval ld_start, ld_end;
                gettimeofday(&ld_start, NULL);
                chunk_t *chunk = pwc_manager->fetch(ind * taskChunks + i);
                if (chunk == NULL) {
                    working_chunk[i].size = 0;
                } else {
                    memcpy(working_chunk[i].vec, chunk->vec, chunk_max_nvecs * CSD);
                    memcpy(working_chunk[i].score, chunk->score, chunk_max_nvecs * sizeof(uint16_t));
                    memcpy(working_chunk[i].norm, chunk->norm, chunk_max_nvecs * sizeof(int32_t));
                    memcpy(working_chunk[i].u, chunk->u, chunk_max_nvecs * sizeof(uint64_t));
                    working_chunk[i].size = chunk->size;
                    pwc_manager->release(chunk->id);
                }
                gettimeofday(&ld_end, NULL);
                ld_stall_us += (ld_end.tv_sec - ld_start.tv_sec) * 1000000ULL + 
                               (ld_end.tv_usec - ld_start.tv_usec);
                if (working_chunk[i].size) _normalize_chunk(&working_chunk[i], CSD);
                for (int j = 0; j < chunk_max_nvecs; j++) rscore[working_chunk[i].score[j]]++;
                #if ENABLE_PROFILING
                CHECK_CUDA_ERR(cudaEventRecord(ev[i*2], stream));
                #endif
                CHECK_CUDA_ERR(cudaMemcpyAsync(pack_buffer + CSD * task_vecs, working_chunk[i].vec, 
                            working_chunk[i].size * CSD, cudaMemcpyHostToDevice, stream));
                #if ENABLE_PROFILING
                CHECK_CUDA_ERR(cudaEventRecord(ev[i*2+1], stream));
                #endif
                task_vecs += working_chunk[i].size;
            }
            
            #if ENABLE_PROFILING
            CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2+0], stream));
            utils_t::device_unpackf(stream, d_buffer, pack_buffer, CSD, task_vecs);
            CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2+1], stream));
            traits::launch(stream, d_buffer, task_vecs, local_data[device_ptr]);
            CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2+2], stream));
            utils_t::device_packf(stream, pack_buffer, d_buffer, CSD, task_vecs);
            CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2+3], stream));
            CHECK_CUDA_ERR(cudaMemcpyAsync(h_buffer, pack_buffer, 
                        task_vecs * CSD, cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA_ERR(cudaMemcpyAsync(h_buffer_score, d_buffer + taskVecs * vec_nbytes, taskVecs * 14, 
                        cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2+4], stream));
            CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
            float x, y, z, w;
            for (int i = 0; i < task_chunks; i++) {
                CHECK_CUDA_ERR(cudaEventElapsedTime(&x, ev[2*i+0], ev[2*i+1]));
                logger->ev_h2d_us += (uint64_t) round(x * 1e3);
            }
            CHECK_CUDA_ERR(cudaEventElapsedTime(&x, ev[2*task_chunks+0], ev[2*task_chunks+1]));
            CHECK_CUDA_ERR(cudaEventElapsedTime(&y, ev[2*task_chunks+1], ev[2*task_chunks+2]));
            CHECK_CUDA_ERR(cudaEventElapsedTime(&z, ev[2*task_chunks+2], ev[2*task_chunks+3]));
            CHECK_CUDA_ERR(cudaEventElapsedTime(&w, ev[2*task_chunks+3], ev[2*task_chunks+4]));
            logger->ev_unpack_us    += (uint64_t) round(x * 1e3);
            logger->ev_kernel_us    += (uint64_t) round(y * 1e3);
            logger->ev_pack_us      += (uint64_t) round(z * 1e3);
            logger->ev_d2h_us       += (uint64_t) round(w * 1e3);
            logger->ev_h2d_nbytes   += CSD * task_vecs;
            logger->ev_d2h_nbytes   += (CSD + 14) * task_vecs;
            logger->ev_vec_nbytes   += CSD * task_vecs;
            logger->ev_curr_chunks  += task_chunks;
            logger->ev_ld_stall_us  += ld_stall_us;
            #else
            utils_t::device_unpackf(stream, d_buffer, pack_buffer, CSD, task_vecs);
            traits::launch(stream, d_buffer, task_vecs, local_data[device_ptr]);
            utils_t::device_packf(stream, pack_buffer, d_buffer, CSD, task_vecs);
            CHECK_CUDA_ERR(cudaMemcpyAsync(h_buffer, pack_buffer, 
                        task_vecs * CSD, cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA_ERR(cudaMemcpyAsync(h_buffer_score, d_buffer + taskVecs * vec_nbytes, taskVecs * 14, 
                        cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
            #endif

            int num_to_check_u = 0;
            uint64_t to_check_u[taskVecs];

            int ptr = 0;
            for (int i = 0; i < task_chunks; i++) {
                chunk_t *chunk = &working_chunk[i];
                for (int j = 0; j < chunk->size; j++) {
                    if (chunk->score[j] == 0) continue;
                    if (h_buffer_score[ptr] == 0) _num_overflow++;
                    else {
                        if (h_buffer_u[ptr] != chunk->u[j]) _num_uid_err++;
                        else {
                            to_check_u[num_to_check_u++] = h_buffer_u[ptr];
                            do {
                                int vec_err = pop_check_vec_err(&h_buffer[ptr * CSD], &chunk->vec[j * CSD], CSD);
                                _vec_err_stat[vec_err]++;

                                int norm_err = abs(h_buffer_norm[ptr] - chunk->norm[j]);
                                int score_err = abs(h_buffer_score[ptr] - chunk->score[j]);
                                int norm = h_buffer_norm[ptr] > 131071 ? 131071 : h_buffer_norm[ptr];
                                int score = h_buffer_score[ptr] > 65535 ? 65535 : h_buffer_score[ptr];
                                norm_err = norm_err > 127 ? 127 : norm_err;
                                score_err = score_err > 127 ? 127 : score_err;

                                _norm_err_stat[norm_err]++;
                                _score_err_stat[score_err]++;
                                _norm_stat[norm]++;
                                _score_stat64[score]++;
                            } while (0);
                        }
                    }
                    ptr++;
                }
            }

            ut_checker.task_commit(to_check_u, num_to_check_u);
        }

        #if ENABLE_PROFILING
        for (int i = 0; i < taskChunks * 2 + 5; i++) CHECK_CUDA_ERR(cudaEventDestroy(ev[i]));
        #endif

        for (int i = 0; i < taskChunks; i++) _free_chunk(&working_chunk[i]);

        pthread_spin_lock(&stat_lock);
        num_overflow += _num_overflow;
        num_uid_err += _num_uid_err;
        for (int i = 0; i < 4; i++) vec_err_stat[i] += _vec_err_stat[i];
        for (int i = 0; i < 128; i++) norm_err_stat[i] += _norm_err_stat[i];
        for (int i = 0; i < 128; i++) score_err_stat[i] += _score_err_stat[i];
        for (int i = 0; i < 65536 * 2; i++) norm_stat[i] += _norm_stat[i];
        for (int i = 0; i < 65536; i++) score_stat64[i] += _score_stat64[i];
        pthread_spin_unlock(&stat_lock);
        free(_norm_stat);
        free(_score_stat64);
    }
    ut_checker.input_done();
    pthread_spin_destroy(&stat_lock);
    
    #if MULTI_SSD
    LOG(ll_info, "basis hash = %lx, prefix = \"%s\", context [%ld, %ld], gh ~ %.1f\n", 
                this->basis_hash, this->pwc_manager->pfx(), index_l, index_r, 
                this->_gh2 * this->_ratio * this->_ratio * 0.25);
    #else
    LOG(ll_info, "basis hash = %lx, prefix = \"%s\", context [%ld, %ld], gh ~ %.1f\n", 
                this->basis_hash, this->pwc_manager->prefix(), index_l, index_r, 
                this->_gh2 * this->_ratio * this->_ratio * 0.25);
    #endif
    
    long num_chunk = pwc_manager->num_chunks();
    long num_empty = pwc_manager->num_empty();
    long num_vec = pwc_manager->num_vec();
    long num_uid = uid_table->size() - CSD - 1;
    int rscore_err = 0;
    for (int i = 1; i < 65536; i++) if (rscore[i] != score_stat[i]) rscore_err++;
    num_uid_notin = ut_checker.wait_work();

    LOG(ll_info, "#chunks = %ld (%ld inactive), #vecs = %ld, %.2f vecs / chunk, %s\n",
            num_chunk, num_empty, num_vec, (num_vec + 0.0) / num_chunk, !rscore_err ? "score_stat ok" : "wrong score_stat");

    {
        int ll = ll_info;
        int th = num_vec * 0.001 > 10 ? 10 : num_vec * 0.001;
        if (num_uid != num_vec || num_overflow || num_uid_notin || num_uid_err) ll = ll_warn;
        if (abs(num_uid - num_vec) > th || num_overflow > th || num_uid_notin > th || num_uid_err > th) ll = ll_err;
        LOG(ll, "#uid - #vec = %ld, #overflow = %ld, #uid_notin = %ld, #uid_err = %ld\n", 
                num_uid - num_vec, num_overflow, num_uid_notin, num_uid_err);
        if (ll == ll_err) ret = -1;
    }

    {
        int ll = ll_info;
        if (vec_err_stat[1] > 0.1 * num_vec || vec_err_stat[2] > 0.01 * num_vec || vec_err_stat[3] > 0.001 * num_vec) ll = ll_warn;
        if (vec_err_stat[1] > 0.5 * num_vec || vec_err_stat[2] > 0.1 * num_vec || vec_err_stat[3] > 0.01 * num_vec) ll = ll_err;
        LOG(ll, "#vec_err:\t%ld(%.2f%%) is 0, %ld(%.2f%%) is 1, %ld(%.2f%%) is 2, %ld(%.2f%%) > 2\n", 
                vec_err_stat[0], vec_err_stat[0] * 100.0 / num_vec,
                vec_err_stat[1], vec_err_stat[1] * 100.0 / num_vec,
                vec_err_stat[2], vec_err_stat[2] * 100.0 / num_vec,
                vec_err_stat[3], vec_err_stat[3] * 100.0 / num_vec);
        if (ll == ll_err) ret = -1;
    }

    {
        int ll = ll_info;
        int64_t acc[8];
        double wth[8] = {1.0, 0.8, 0.5, 0.3, 0.15, 0.08, 0.03, 0.005};
        double eth[8] = {1.0, 1.0, 0.9, 0.5, 0.3, 0.15, 0.08, 0.02};
        for (int i = 0; i < 8; i++) {
            acc[i] = 0;
            if (i) for (int j = (1 << (i-1)); j < (1 << i); j++) acc[i] += norm_err_stat[j];
        }
        acc[0] = norm_err_stat[0];
        for (int i = 1; i < 8; i++) {
            if (acc[i] > wth[i] * num_vec) ll = ll_warn;
            if (acc[i] > eth[i] * num_vec) ll = ll_err;
        }
        LOG(ll, "#norm_err:\t%ld(%.2f%%) is 0, %ld(%.2f%%) is 1, %ld(%.2f%%) is 2-3, %ld(%.2f%%) is 4-7, "
                "%ld(%.2f%%) is 8-15, %ld(%.2f%%) is 16-31, %ld(%.2f%%) is 32-63, %ld(%.2f%%) > 63\n",
                acc[0], acc[0] * 100.0 / num_vec,
                acc[1], acc[1] * 100.0 / num_vec,
                acc[2], acc[2] * 100.0 / num_vec,
                acc[3], acc[3] * 100.0 / num_vec,
                acc[4], acc[4] * 100.0 / num_vec,
                acc[5], acc[5] * 100.0 / num_vec,
                acc[6], acc[6] * 100.0 / num_vec,
                acc[7], acc[7] * 100.0 / num_vec);
        if (ll == ll_err) ret = -1;
    }

    {
        int ll = ll_info;
        int64_t acc[8];
        double wth[8] = {1.0, 0.65, 0.3, 0.15, 0.08, 0.03, 0.005, 0.001};
        double eth[8] = {1.0, 1.0, 0.4, 0.2, 0.15, 0.08, 0.02, 0.005};
        for (int i = 0; i < 8; i++) {
            acc[i] = 0;
            if (i) for (int j = (1 << (i-1)); j < (1 << i); j++) acc[i] += score_err_stat[j];
        }
        acc[0] = score_err_stat[0];
        for (int i = 1; i < 8; i++) {
            if (acc[i] > wth[i] * num_vec) ll = ll_warn;
            if (acc[i] > eth[i] * num_vec) ll = ll_err;
        }
        LOG(ll, "#score_err:\t%ld(%.2f%%) is 0, %ld(%.2f%%) is 1, %ld(%.2f%%) is 2-3, %ld(%.2f%%) is 4-7, "
                "%ld(%.2f%%) is 8-15, %ld(%.2f%%) is 16-31, %ld(%.2f%%) is 32-63, %ld(%.2f%%) > 63\n",
                acc[0], acc[0] * 100.0 / num_vec,
                acc[1], acc[1] * 100.0 / num_vec,
                acc[2], acc[2] * 100.0 / num_vec,
                acc[3], acc[3] * 100.0 / num_vec,
                acc[4], acc[4] * 100.0 / num_vec,
                acc[5], acc[5] * 100.0 / num_vec,
                acc[6], acc[6] * 100.0 / num_vec,
                acc[7], acc[7] * 100.0 / num_vec);
        if (ll == ll_err) ret = -1;
    }

    {
        int ll = ll_info;
        int gh = this->_gh2 * this->_ratio * this->_ratio * 0.5;
        double gh_ratio[19] = {1.00, 1.02, 1.05, 1.10, 1.11, 1.12, 1.13, 1.14, sqrt(4./3.), 1.17, 
                               1.19, 1.22, 1.25, 1.30, 1.35, 1.40, 1.50, 2.00, 3.00};
        int64_t gh_stat[20] = {};
        int ptr = 0;
        int th = gh_ratio[0] * gh_ratio[0] * gh;
        for (int i = 0; i < 131072; i++) {
            if (i > th) {
                ptr++;
                if (ptr < 19) th = gh_ratio[ptr] * gh_ratio[ptr] * gh;
                else th = 131072;
            }
            gh_stat[ptr] += norm_stat[i];
        }

        if (gh < 10000 || gh > 50000) ll = ll_warn;
        if (norm_stat[131071] > 0.1 * num_vec) ll = ll_warn;
        if (norm_stat[131071] > 0.5 * num_vec) ll = ll_err;
        for (int i = 0; i < 19; i++) {
            if (gh_stat[i] > num_vec * pow(gh_ratio[i], CSD) * 2) ll = ll_warn;
            if (gh_stat[i] > num_vec * pow(gh_ratio[i], CSD) * 10) ll = ll_err;
        }
        LOG(ll, "================= norm stat =================\n"
                "radius\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t\n"
                "#vec\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t\t\n"
                "radius\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t res\n"
                "#vec\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t\t\n",
                gh_ratio[0], gh_ratio[1], gh_ratio[2], gh_ratio[3], gh_ratio[4], gh_ratio[5], gh_ratio[6], gh_ratio[7], gh_ratio[8], gh_ratio[9],
                gh_stat[0], gh_stat[1], gh_stat[2], gh_stat[3], gh_stat[4], gh_stat[5], gh_stat[6], gh_stat[7], gh_stat[8], gh_stat[9], 
                gh_ratio[10], gh_ratio[11], gh_ratio[12], gh_ratio[13], gh_ratio[14], gh_ratio[15], gh_ratio[16], gh_ratio[17], gh_ratio[18],
                gh_stat[10], gh_stat[11], gh_stat[12], gh_stat[13], gh_stat[14], gh_stat[15], gh_stat[16], gh_stat[17], gh_stat[18], gh_stat[19]);
        if (ll == ll_err) ret = -1;
    }

    {
        int ll = ll_info;
        int gh = this->_gh2 * this->_ratio * this->_ratio * 0.25;
        double gh_ratio[19] = {1.00, 1.02, 1.05, 1.10, 1.11, 1.12, 1.13, 1.14, sqrt(4./3.), 1.17, 
                               1.19, 1.22, 1.25, 1.30, 1.35, 1.40, 1.50, 2.00, 3.00};
        int64_t gh_stat[20] = {};
        int ptr = 0;
        int th = gh_ratio[0] * gh_ratio[0] * gh;
        for (int i = 0; i < 65536; i++) {
            if (i > th) {
                ptr++;
                if (ptr < 19) th = gh_ratio[ptr] * gh_ratio[ptr] * gh;
                else th = 65536;
            }
            gh_stat[ptr] += score_stat64[i];
        }
        if (gh < 5000 || gh > 25000) ll = ll_warn;
        if (score_stat64[65535] > 0.1 * num_vec) ll = ll_warn;
        if (score_stat64[65535] > 0.5 * num_vec) ll = ll_err;
        for (int i = 0; i < 19; i++) {
            if (gh_stat[i] > num_vec * pow(gh_ratio[i], CSD) * 5) ll = ll_warn;
            if (gh_stat[i] > num_vec * pow(gh_ratio[i], CSD) * 50) ll = ll_err;
        }
        LOG(ll, "================= score stat =================\n"
                "radius\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t\n"
                "#vec\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t\t\n"
                "radius\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t res\n"
                "#vec\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t%12ld\t\t\n",
                gh_ratio[0], gh_ratio[1], gh_ratio[2], gh_ratio[3], gh_ratio[4], gh_ratio[5], gh_ratio[6], gh_ratio[7], gh_ratio[8], gh_ratio[9],
                gh_stat[0], gh_stat[1], gh_stat[2], gh_stat[3], gh_stat[4], gh_stat[5], gh_stat[6], gh_stat[7], gh_stat[8], gh_stat[9], 
                gh_ratio[10], gh_ratio[11], gh_ratio[12], gh_ratio[13], gh_ratio[14], gh_ratio[15], gh_ratio[16], gh_ratio[17], gh_ratio[18],
                gh_stat[10], gh_stat[11], gh_stat[12], gh_stat[13], gh_stat[14], gh_stat[15], gh_stat[16], gh_stat[17], gh_stat[18], gh_stat[19]);
        if (ll == ll_err) ret = -1;
    }
    
    for (int i = 0; i < num_devices; i++) CHECK_CUDA_ERR(cudaFree(local_data[i]));
    free(norm_stat);
    free(score_stat64);
    
    lg_report();
    lg_exit();

    return ret;
    
    #undef LOG
}

#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sys/select.h>
#include <dirent.h>
#include <cstring>

struct load_ckpfcher_t {
    static constexpr int pfch_ahead = 32;
    static constexpr int tail_stop = 128;

    load_ckpfcher_t(Pool_hd_t *pool, int *exp_left) {
        this->pwc_manager = pool->pwc_manager;
        this->exp_left = exp_left;
        pthread_spin_init(&lock, PTHREAD_PROCESS_SHARED);

        if (exp_left[0] >= tail_stop) {
            for (int i = 0; i < pfch_ahead; i++) {
                pfch_id[i] = pwc_manager->create_chunk();
                pwc_manager->prefetch(pfch_id[i]);
            }
        }
    }

    ~load_ckpfcher_t() {
        pthread_spin_destroy(&lock);
    }

    chunk_t *pop() {
        int to_pfch_id = -1;
        pthread_spin_lock(&lock);
        int ret_id = pfch_id[curr_pos];
        if (ret_id == -1) {
            pthread_spin_unlock(&lock);
            ret_id = pwc_manager->create_chunk();
            return pwc_manager->fetch(ret_id);
        }
        if (exp_left[0] >= tail_stop) {
            pfch_id[curr_pos] = pwc_manager->create_chunk();
            to_pfch_id = pfch_id[curr_pos];
        } else pfch_id[curr_pos] = -1;
        curr_pos = (curr_pos + 1) % pfch_ahead;
        pthread_spin_unlock(&lock);

        chunk_t *ret = pwc_manager->fetch(ret_id);
        if (to_pfch_id >= 0) pwc_manager->prefetch(to_pfch_id);
        
        return ret;
    }
    
    int curr_pos = 0;
    int pfch_id[pfch_ahead];
    
    int *exp_left = NULL;
    pthread_spinlock_t lock;
    pwc_manager_t *pwc_manager;
};

int Pool_hd_t::load(long log_level) {
    lg_init();
    
    if (pwc_manager->num_chunks()) {
        lg_err("pool is not empty, nothing done.");
        lg_exit();
        return -1;
    }

    int num_devices;
    cudaDeviceProp device_props[MAX_NUM_DEVICE];
    if (_prepare_device_prop(num_devices, device_props)) {
        lg_err("no device detected, nothing done.");
        lg_exit();
        return -1;
    }

    pwc_manager->wait_work();
    if (CSD > 120) {
        long target_cached_chunks = PWC_DEFAULT_MAX_CACHED_CHUNKS + 
                                    BWC_DEFAULT_MAX_CACHED_CHUNKS +
                                    SWC_DEFAULT_MAX_CACHED_CHUNKS;
        if (pwc_manager->max_cached_chunks() != target_cached_chunks) {
            pwc_manager->set_max_cached_chunks(target_cached_chunks);
        }
    }

    constexpr uint32_t _id_processing = 0x80000000;

    std::vector<std::string> wrong_name;
    int32_t *exist_ids_ptr;
    int32_t exist_ids_size;
    std::vector<int32_t> exist_ids;
    std::vector<int32_t> wrong_hash;
    std::vector<int32_t> wrong_context;
    std::vector<int32_t> wrong_format;
    int max_available_id = -1;
    pthread_spinlock_t lock;
    pthread_spinlock_t stat_lock;
    pthread_spin_init(&lock, PTHREAD_PROCESS_SHARED);
    pthread_spin_init(&stat_lock, PTHREAD_PROCESS_SHARED);

    #if MULTI_SSD
    char protect_prefix[32];
    for (int I = 0; I < hw::ssd_num; I++)
    #else
    char dir[32];
    char protect_prefix[32];
    #endif
    {   
        // get all possible file names
        #if MULTI_SSD
        char dir[256];
        snprintf(dir, sizeof(dir), "%s/%s", pwc_manager->dir(), hw::ssd_name_list[I]);
        #else
        memcpy(dir, pwc_manager->prefix(), 32);
        for (int i = strlen(dir) - 1; i >= 0; i--) {
            if (dir[i] == '/') {
                dir[i] = '\0';
                break;
            }
        }
        #endif

        DIR* dirp = opendir(dir);
        if (!dirp) {
            pthread_spin_destroy(&lock);
            pthread_spin_destroy(&stat_lock);
            lg_err("opendir failed, %s", strerror(errno));
            lg_exit();
            return -1;
        }

        #if MULTI_SSD
        const int exp_len = strlen(pwc_manager->pfx()) + 6;
        #else
        const int dir_len = strlen(dir);
        const int exp_len = strlen(pwc_manager->prefix()) + 6 - dir_len - 1;
        #endif
        struct dirent* entry;
        while ((entry = readdir(dirp)) != NULL) {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            int len = strlen(entry->d_name);
            #if MULTI_SSD
            if (len != exp_len) wrong_name.push_back(std::string(hw::ssd_name_list[I]) + "/" + std::string(entry->d_name));
            #else
            if (len != exp_len) wrong_name.push_back(std::string(entry->d_name));
            #endif
            else {
                int format_ok = 1;
                int id = 0;
                for (int i = 0; i < exp_len - 6; i++) {
                    #if MULTI_SSD
                    if (entry->d_name[i] != pwc_manager->pfx()[i]) format_ok = 0;
                    #else
                    if (entry->d_name[i] != pwc_manager->prefix()[dir_len + 1 + i]) format_ok = 0;
                    #endif
                }
                for (int i = 0; i < 6; i++) {
                    if (entry->d_name[exp_len - 6 + i] >= 'a' && entry->d_name[exp_len - 6 + i] <= 'f') id += (entry->d_name[exp_len - 6 + i] - 'a' + 10) << (4 * (5 - i));
                    else if (entry->d_name[exp_len - 6 + i] >= '0' && entry->d_name[exp_len - 6 + i] <= '9') id += (entry->d_name[exp_len - 6 + i] - '0') << (4 * (5 - i));
                    else format_ok = 0;
                }
                #if MULTI_SSD
                if (strcmp(hw::ssd_name(id), hw::ssd_name_list[I])) format_ok = 0;
                #endif
                if (format_ok) exist_ids.push_back(id);
                #if MULTI_SSD
                else wrong_name.push_back(std::string(hw::ssd_name_list[I]) + "/" + std::string(entry->d_name));
                #else
                else wrong_name.push_back(std::string(entry->d_name));
                #endif
            }
        }

        closedir(dirp);
        #if MULTI_SSD
    }
        #endif
        
        std::sort(exist_ids.begin(), exist_ids.end(), std::greater<int32_t>());
        exist_ids_ptr = exist_ids.data();
        exist_ids_size = exist_ids.size();

        time_t now = time(NULL);
        struct tm *local_time = localtime(&now);
        strftime(protect_prefix, sizeof(protect_prefix), "PROTECTED-%Y-%m-%d-%H-%M-%S", local_time);
    #if !MULTI_SSD
    }
    #endif

    #if ENABLE_PROFILING
    logger->ev_total_chunks = exist_ids_size;
    logger->num_thread = this->_num_threads;
    #endif

    ut_checker_t ut_checker(ut_checker_t::type_others, this, uid_table, pwc_manager);

    ut_checker.set_exp_batch(exist_ids_size * ut_checker_t::default_batch_ratio);
    ut_checker.set_max_available_id(&max_available_id);
    ut_checker.set_score_stat(score_stat, stat_lock);

    typedef check_traits traits;

    for (int i = 0; i < 65536; i++) score_stat[i] = 0U;

    local_data_t *local_data[MAX_NUM_DEVICE];
    for (int i = 0; i < num_devices; i++) {
        CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[i]));
        traits::prep_device_local_data(local_data[i], this);
    }
    constexpr int taskChunks = traits::taskChunks;
    constexpr int taskVecs = traits::taskVecs;

    load_ckpfcher_t ckpfcher(this, &exist_ids_size);

    #pragma omp parallel for num_threads(_num_threads)
    for (long thread = 0; thread < _num_threads; thread++) {
        const int device_ptr = hw::gpu_ptr(thread, _num_threads);
        CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[device_ptr]));

        traits::init_shared_mem_limit();

        cudaStream_t stream;
        traits::stream_with_l2_holder_t 
        l2_holder(stream, local_data[device_ptr]->b_full, traits::b_full_active_nbytes);

        int8_t *h_buffer, *d_buffer, *pack_buffer;
        uint16_t *h_buffer_score; int32_t *h_buffer_norm; uint64_t *h_buffer_u;
        traits::pool_hd_buffer_holder_t 
        buffer_holder(stream, d_buffer, pack_buffer, h_buffer, h_buffer_score, h_buffer_norm, h_buffer_u);

        chunk_t working_chunk[taskChunks];
        for (int i = 0; i < taskChunks; i++) _malloc_chunk(&working_chunk[i]);

        #if ENABLE_PROFILING
        cudaEvent_t ev[taskChunks * 2 + 5];
        for (int i = 0; i < taskChunks * 2 + 5; i++) CHECK_CUDA_ERR(cudaEventCreate(&ev[i]));
        #endif

        for (;;) {
            int task_chunks = 0, task_vecs = 0;
            uint64_t ld_stall_us = 0;
            while (task_chunks < taskChunks) {
                int32_t ptr, id = -1;
                pthread_spin_lock(&lock);
                while (exist_ids_size > 0) {
                    if (exist_ids_ptr[exist_ids_size - 1] == -1) exist_ids_size--;
                    else break;
                }
                for (int i = exist_ids_size - 1; i >= 0; i--) {
                    if ((exist_ids_ptr[i] & _id_processing) == 0) {
                        id = exist_ids[i];
                        ptr = i;
                        exist_ids[i] |= _id_processing;
                        break;
                    }
                }
                if (exist_ids_size == 0) max_available_id = 0xfffffff;
                else max_available_id = (exist_ids_ptr[exist_ids_size - 1] & 0x7fffffff) - 1 > max_available_id ? 
                                        (exist_ids_ptr[exist_ids_size - 1] & 0x7fffffff) - 1 : max_available_id;                
                pthread_spin_unlock(&lock);

                if (id != -1) {
                    char chunk_filename[256];
                    #if MULTI_SSD
                    snprintf(chunk_filename, sizeof(chunk_filename), "%s/%s/%s%06x", pwc_manager->dir(), 
                                hw::ssd_name(id), pwc_manager->pfx(), id);
                    #else
                    snprintf(chunk_filename, sizeof(chunk_filename), "%s%06x", pwc_manager->prefix(), id);
                    #endif
                    chunk_t *dst_chunk = &working_chunk[task_chunks];

                    int read_vecs, read_bytes;

                    struct timeval ld_start, ld_end;
                    gettimeofday(&ld_start, NULL);
                    #if ONE_TIME_IO
                    char *meta_data = ((char *)dst_chunk->score) - 12;
                    #else
                    char meta_data[12];
                    #endif
                    int fd = open(chunk_filename, O_RDONLY);
                    if (fd == -1) {
                        if (errno != ENOENT) lg_err("open %s failed, %s, ignored", chunk_filename, strerror(errno));
                        goto pool_load_id_process_end;
                    }

                    #define FAIL_DUETO(_reason) do {    \
                        pthread_spin_lock(&lock);       \
                        _reason.push_back(id);          \
                        pthread_spin_unlock(&lock);     \
                        goto pool_load_id_process_end;  \
                    } while (0)
                    
                    #if ONE_TIME_IO
                    read_bytes = read(fd, meta_data, 12 + chunk_max_nvecs * (2 + 4 + 8 + CSD));
                    if (read_bytes < 12 + (2 + 4 + 8) * chunk_max_nvecs) FAIL_DUETO(wrong_format);
                    #else
                    read_bytes = read(fd, meta_data, 12);
                    if (read_bytes != 12) FAIL_DUETO(wrong_format);
                    #endif
                    
                    if (*((uint8_t *)(&meta_data[10])) != index_l || 
                        *((uint8_t *)(&meta_data[11])) != index_r) FAIL_DUETO(wrong_context);

                    if (*((uint64_t *)(&meta_data[2])) != basis_hash) FAIL_DUETO(wrong_hash);

                    #if !ONE_TIME_IO
                    read_bytes = read(fd, dst_chunk->score, sizeof(uint16_t) * chunk_max_nvecs);
                    read_bytes += read(fd, dst_chunk->norm, sizeof(int32_t) * chunk_max_nvecs);
                    read_bytes += read(fd, dst_chunk->u, sizeof(uint64_t) * chunk_max_nvecs);
                    if (read_bytes < chunk_max_nvecs * (2 + 4 + 8)) FAIL_DUETO(wrong_format);
                    #endif
                    #undef FAIL_DUETO

                    #if ONE_TIME_IO
                    read_bytes -= 12 + (2 + 4 + 8) * chunk_max_nvecs;
                    #else
                    read_bytes = read(fd, dst_chunk->vec, CSD * chunk_max_nvecs);
                    #endif
                    read_vecs = read_bytes / CSD;
                    memset(dst_chunk->norm + read_vecs, 0, (chunk_max_nvecs - read_vecs) * sizeof(int32_t));
                    memset(dst_chunk->score + read_vecs, 0, (chunk_max_nvecs - read_vecs) * sizeof(uint16_t));
                    dst_chunk->size = 0;
                    for (int i = 0; i < read_vecs; i++) {
                        if (dst_chunk->score[i]) dst_chunk->size++;
                    }
                    if (dst_chunk->size == 0) {
                        pthread_spin_lock(&lock);
                        wrong_format.push_back(id);
                        pthread_spin_unlock(&lock);
                        goto pool_load_id_process_end;
                    } else if (dst_chunk->size != *((uint16_t *)(&meta_data[0]))) {
                        lg_err("%s size validation failed, real %d, disk %d, ignored", 
                                chunk_filename, dst_chunk->size, *((uint16_t *)(&meta_data[0])));
                    }
                    gettimeofday(&ld_end, NULL);
                    ld_stall_us += (ld_end.tv_sec - ld_start.tv_sec) * 1000000ULL + 
                                   (ld_end.tv_usec - ld_start.tv_usec);
                    _normalize_chunk(dst_chunk, CSD);
                    #if ENABLE_PROFILING
                    CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2], stream));
                    #endif
                    CHECK_CUDA_ERR(cudaMemcpyAsync(pack_buffer + CSD * task_vecs, dst_chunk->vec, 
                                dst_chunk->size * CSD, cudaMemcpyHostToDevice, stream));
                    #if ENABLE_PROFILING
                    CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2+1], stream));
                    #endif
                    task_vecs += dst_chunk->size;
                    task_chunks++;
                    close(fd);
                    exist_ids[ptr] = -1;
                    continue;
                    
                    pool_load_id_process_end:
                    #if ENABLE_PROFILING
                    logger->ev_total_chunks--;
                    #endif
                    if (fd != -1) close(fd);
                    char old_filename[256];
                    char new_filename[256+32];
                    #if MULTI_SSD
                    snprintf(old_filename, sizeof(old_filename), "%s/%s/%s%06x", pwc_manager->dir(), 
                                hw::ssd_name(id), pwc_manager->pfx(), id);
                    snprintf(new_filename, sizeof(new_filename), "%s/%s/%s%s%06x", pwc_manager->dir(), 
                                hw::ssd_name(id), protect_prefix, pwc_manager->pfx(), id);
                    #else
                    snprintf(old_filename, sizeof(old_filename), "%s%06x", pwc_manager->prefix(), id);
                    snprintf(new_filename, sizeof(new_filename), "%s/%s%s%06x", dir, protect_prefix, 
                                pwc_manager->prefix() + strlen(dir) + 1, id);
                    #endif
                    if (rename(old_filename, new_filename)) {
                        lg_warn("rename wrong file %s --> %s fail, %s, ignored.", old_filename, new_filename, strerror(errno));
                    }
                    exist_ids[ptr] = -1;
                    continue;
                }
                
                if (exist_ids_size == 0) break;
            }
            
            chunk_t *to_store[taskChunks];
            for (int i = 0; i * chunk_max_nvecs < task_vecs; i++) {
                to_store[i] = ckpfcher.pop();
            }
            
            #if ENABLE_PROFILING
            CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2+0], stream));
            utils_t::device_unpackf(stream, d_buffer, pack_buffer, CSD, task_vecs);
            CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2+1], stream));
            traits::launch(stream, d_buffer, task_vecs, local_data[device_ptr]);
            CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2+2], stream));
            utils_t::device_packf(stream, pack_buffer, d_buffer, CSD, task_vecs);
            CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2+3], stream));
            for (int i = 0; i * chunk_max_nvecs < task_vecs; i++) {
                int n = (i + 1) * chunk_max_nvecs <= task_vecs ? chunk_max_nvecs : task_vecs - i * chunk_max_nvecs;
                CHECK_CUDA_ERR(cudaMemcpyAsync(to_store[i]->vec, pack_buffer + i * chunk_max_nvecs * CSD, 
                            n * CSD, cudaMemcpyDeviceToHost, stream));
            }
            CHECK_CUDA_ERR(cudaMemcpyAsync(h_buffer_score, d_buffer + taskVecs * vec_nbytes, taskVecs * 14, 
                        cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA_ERR(cudaEventRecord(ev[task_chunks*2+4], stream));
            CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
            float x, y, z, w;
            for (int i = 0; i < task_chunks; i++) {
                CHECK_CUDA_ERR(cudaEventElapsedTime(&x, ev[2*i+0], ev[2*i+1]));
                logger->ev_h2d_us += (uint64_t) round(x * 1e3);
            }
            CHECK_CUDA_ERR(cudaEventElapsedTime(&x, ev[2*task_chunks+0], ev[2*task_chunks+1]));
            CHECK_CUDA_ERR(cudaEventElapsedTime(&y, ev[2*task_chunks+1], ev[2*task_chunks+2]));
            CHECK_CUDA_ERR(cudaEventElapsedTime(&z, ev[2*task_chunks+2], ev[2*task_chunks+3]));
            CHECK_CUDA_ERR(cudaEventElapsedTime(&w, ev[2*task_chunks+3], ev[2*task_chunks+4]));
            logger->ev_unpack_us   += (uint64_t) round(x * 1e3);
            logger->ev_kernel_us   += (uint64_t) round(y * 1e3);
            logger->ev_pack_us     += (uint64_t) round(z * 1e3);
            logger->ev_d2h_us      += (uint64_t) round(w * 1e3);
            logger->ev_h2d_nbytes  += CSD * task_vecs;
            logger->ev_d2h_nbytes  += (CSD + 14) * task_vecs;
            logger->ev_vec_nbytes  += CSD * task_vecs;
            logger->ev_curr_chunks += task_chunks;
            logger->ev_ld_stall_us += ld_stall_us;

            #else
            traits::launch(stream, d_buffer, task_vecs, local_data[device_ptr]);
            utils_t::device_packf(stream, pack_buffer, d_buffer, CSD, task_vecs);
            for (int i = 0; i * chunk_max_nvecs < task_vecs; i++) {
                int n = (i + 1) * chunk_max_nvecs <= task_vecs ? chunk_max_nvecs : task_vecs - i * chunk_max_nvecs;
                CHECK_CUDA_ERR(cudaMemcpyAsync(to_store[i]->vec, pack_buffer + i * chunk_max_nvecs * CSD, 
                            n * CSD, cudaMemcpyDeviceToHost, stream));
            }
            CHECK_CUDA_ERR(cudaMemcpyAsync(h_buffer_score, d_buffer + taskVecs * vec_nbytes, taskVecs * 14, 
                        cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
            #endif

            // write back
            int num_used = 0;
            for (int i = 0; i * chunk_max_nvecs < task_vecs; i++) {
                chunk_t *chunk = to_store[i];

                WRITE_BACK_TO_CHUNK(chunk);
                if (num_used == task_vecs) break;
            }
            if (exist_ids_size == 0) break;
        }
        
        #if ENABLE_PROFILING
        for (int i = 0; i < taskChunks * 2 + 5; i++) CHECK_CUDA_ERR(cudaEventDestroy(ev[i]));
        #endif

        for (int i = 0; i < taskChunks; i++) _free_chunk(&working_chunk[i]);
    }

    ut_checker.input_done();

    for (int i = 0; i < num_devices; i++) CHECK_CUDA_ERR(cudaFree(local_data[i]));
    pthread_spin_destroy(&lock);

    #if MULTI_SSD
    #define PRINT_DUETO(_reason, _rstr) do {                                                \
        char out_buf[4000];                                                                 \
        int len = snprintf(out_buf, sizeof(out_buf), "detected %lu %s (protected):\n",      \
                                                    _reason.size(), _rstr);                 \
        for (int i = 0; i < _reason.size(); i++) {                                          \
            if (len > 3900) {                                                               \
                len += snprintf(&out_buf[len], sizeof(out_buf) - len, "......");            \
                break;                                                                      \
            } else len += snprintf(&out_buf[len], sizeof(out_buf) - len, "%s%06x   ",       \
                                                        pwc_manager->pfx(), _reason[i]);    \
        }                                                                                   \
        lg_warn("%s", out_buf);                                                             \
    } while (0)

    #define DELETE_DUETO(_reason) do {                                                                  \
        for (int i = 0; i < _reason.size(); i++) {                                                      \
            char chunk_filename[256+32];                                                                \
            snprintf(chunk_filename, sizeof(chunk_filename), "%s/%s/%s%s%06x", pwc_manager->dir(),      \
                    hw::ssd_name(_reason[i]), protect_prefix, pwc_manager->pfx(), _reason[i]);          \
            if (remove(chunk_filename)) lg_warn("remove wrong file %s fail, %s, ignored.",              \
                                                chunk_filename, strerror(errno));                       \
        }                                                                                               \
    } while (0)
    #else
    #define PRINT_DUETO(_reason, _rstr) do {                                                \
        char out_buf[4000];                                                                 \
        int len = snprintf(out_buf, sizeof(out_buf), "detected %lu %s (protected):\n",      \
                                                    _reason.size(), _rstr);                 \
        for (int i = 0; i < _reason.size(); i++) {                                          \
            if (len > 3900) {                                                               \
                len += snprintf(&out_buf[len], sizeof(out_buf) - len, "......");            \
                break;                                                                      \
            } else len += snprintf(&out_buf[len], sizeof(out_buf) - len, "%s%06x   ",       \
                                  pwc_manager->prefix() + strlen(dir) + 1, _reason[i]);     \
        }                                                                                   \
        lg_warn("%s", out_buf);                                                             \
    } while (0)

    #define DELETE_DUETO(_reason) do {                                                                  \
        for (int i = 0; i < _reason.size(); i++) {                                                      \
            char chunk_filename[256+32];                                                                \
            snprintf(chunk_filename, sizeof(chunk_filename), "%s/%s%s%06x", dir, protect_prefix,        \
                                pwc_manager->prefix() + strlen(dir) + 1, _reason[i]);                   \
            if (remove(chunk_filename)) lg_warn("remove wrong file %s fail, %s, ignored.",              \
                                                chunk_filename, strerror(errno));                       \
        }                                                                                               \
    } while (0)
    #endif

    #define WAIT_CONFIRM_FOR(action) \
    do { \
        char input[8]; \
        int confirmed = 0; \
        fprintf(stderr, "Type 'confirm' to execute delete.\n"); \
        for (int i = 9; i >= 0; i--) { \
            fprintf(stderr, "\r%d", i); \
            fflush(stdout); \
            sleep(1); \
            struct timeval timeout; \
            fd_set set; \
            FD_ZERO(&set); \
            FD_SET(STDIN_FILENO, &set); \
            timeout.tv_sec = 0; \
            timeout.tv_usec = 0; \
            int rv = select(STDIN_FILENO + 1, &set, NULL, NULL, &timeout); \
            if (rv > 0) { \
                if (fgets(input, sizeof(input), stdin) != NULL) { \
                    if (strncmp(input, "confirm", 7) == 0) { \
                        confirmed = 1; \
                        break; \
                    } \
                } \
            } \
            FILE *file = fopen(".in", "r"); \
            if (file) { \
                char buffer[8]; \
                if (fgets(buffer, sizeof(buffer), file) != NULL) { \
                    if (strncmp(buffer, "confirm", 7) == 0) { \
                        confirmed = 1; \
                        fclose(file); \
                        remove(".in"); \
                        break; \
                    } \
                } \
                fclose(file); \
            } \
        } \
        if (confirmed) { \
            fprintf(stderr, "\r\033[36mCONFIRMED\033[0m\n"); \
            sleep(1); \
            action; \
        } else { \
            fprintf(stderr, "\r\033[36mNOT CONFIRMED\033[0m\n"); \
            sleep(1); \
        } \
    } while (0)

    uint64_t total_wrong = wrong_name.size() + wrong_format.size() + wrong_context.size() + wrong_hash.size();
    if (log_level == -1 && total_wrong) {
        lg_warn("detected %lu wrong name, %lu wrong format, %lu wrong context, %lu wrong hash, all deleted\n",
                                wrong_name.size(), wrong_format.size(), wrong_context.size(), wrong_hash.size());
        for (int i = 0; i < wrong_name.size(); i++) {
            char chunk_filename[256];
            #if MULTI_SSD
            snprintf(chunk_filename, sizeof(chunk_filename), "%s/%s", pwc_manager->dir(), wrong_name[i].c_str());
            #else
            snprintf(chunk_filename, sizeof(chunk_filename), "%s/%s", dir, wrong_name[i].c_str());
            #endif
            if (remove(chunk_filename)) {
                lg_warn("remove wrong file %s fail, %s, ignored.", chunk_filename, strerror(errno));
            }
        }
        DELETE_DUETO(wrong_format);
        DELETE_DUETO(wrong_context);
        DELETE_DUETO(wrong_hash);
    }

    if (log_level >= 0) {
        if (wrong_name.size()) {
            char out_buf[4000];
            int len = snprintf(out_buf, sizeof(out_buf), "detected %lu wrong name (protected):\n", wrong_name.size());
            for (int i = 0; i < wrong_name.size(); i++) {
                if (len > 3900) {
                    len += snprintf(&out_buf[len], sizeof(out_buf) - len, "......");
                    break;
                } else {
                    len += snprintf(&out_buf[len], sizeof(out_buf) - len, "%s   ", wrong_name[i].c_str());
                    if (len >= 4000) lg_warn("wrong name too long, truncated");
                }
            }
            lg_warn("%s", out_buf);
            if (log_level >= 1) {
                int need_delete = 0;
                WAIT_CONFIRM_FOR(need_delete = 1);
                if (need_delete) {
                    for (int i = 0; i < wrong_name.size(); i++) {
                        char chunk_filename[256];
                        #if MULTI_SSD
                        snprintf(chunk_filename, sizeof(chunk_filename), "%s/%s", pwc_manager->dir(), wrong_name[i].c_str());
                        #else
                        snprintf(chunk_filename, sizeof(chunk_filename), "%s/%s", dir, wrong_name[i].c_str());
                        #endif
                        if (remove(chunk_filename)) {
                            lg_warn("remove wrong file %s fail, %s, ignored.", chunk_filename, strerror(errno));
                        }
                    }
                }
            }
        }

        if (wrong_format.size()) {
            PRINT_DUETO(wrong_format, "wrong format");
            if (log_level >= 1) WAIT_CONFIRM_FOR(DELETE_DUETO(wrong_format));
        }
        if (wrong_context.size()) {
            PRINT_DUETO(wrong_context, "wrong context");
            if (log_level >= 1) WAIT_CONFIRM_FOR(DELETE_DUETO(wrong_context));
        }
        if (wrong_hash.size()) {
            PRINT_DUETO(wrong_hash, "wrong hash");
            if (log_level >= 1) WAIT_CONFIRM_FOR(DELETE_DUETO(wrong_hash));
        }
        lg_info("%ld vectors loaded to %ld chunks", pwc_manager->num_vec(), pwc_manager->num_chunks());
    }

    #undef PRINT_DUETO
    #undef DELETE_DUETO
    #undef WAIT_CONFIRM_FOR

    ut_checker.wait_work();
    pthread_spin_destroy(&stat_lock);
    
    lg_report();
    lg_exit();

    return 0;
}

/// for dh_insert
template int Pool_hd_t::stream_task_template<pdev_traits_t<5u>>(int, cudaDeviceProp*, local_data_t**);
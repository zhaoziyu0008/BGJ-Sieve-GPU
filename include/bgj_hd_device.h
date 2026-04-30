#ifndef __BGJ_HD_DEVICE_H
#define __BGJ_HD_DEVICE_H

#include "common_device.h"
#include "random_device.h"
#include "pool_hd.h"

#include <cuda_runtime.h>

template <int32_t layer, int32_t batchsize, int32_t CSD16>
__global__ void _bucket_kernel(uint32_t *__restrict__ out, int out_max_size, int8_t *__restrict__ vec_pad16,
                                int32_t *__restrict__ vnorm, const int8_t *__restrict__ center, uint32_t in_max_size,
                                const int8_t *__restrict__ vec, int *n_ptr, float alpha, int CSD, int gbuc_freq, int ind_bias);

/// launch it with 1 block, 512 threads
template <int32_t layer>
__global__ void _reduce_prepare(int32_t *vnorm, int *n_ptr, int max_n);

template <int32_t layer, int32_t CSD16, uint32_t sbuc_freq, uint32_t gbuc_freq>
__global__ void _reduce_kernel(int *out, int *num_out, int out_max_size, const int8_t *__restrict__ vec_pad16,
                                      int32_t *__restrict__ vnorm, int *n_ptr, int32_t goal_norm);

#if !USE_GRAPH
__global__ void _multi_reduce_prepare(int32_t *vids, int *n_ptr, int max_n);

template <int32_t CSD16, uint32_t sbuc_freq, uint32_t gbuc_freq>
__global__ void _multi_reduce_kernel(int *out, int *num_out, int out_max_size, const int8_t *__restrict__ vec_pad16, 
                                    int32_t *__restrict__ vids, int *n_ptr, int buc_max_size, int32_t goal_norm);
#endif

constexpr int filter_taskVecs   = 262144;
constexpr int fpv_shmem         = 68608;
constexpr int fpv_blocks        = 64;
constexpr int fpv_threads       = 256;
constexpr int fcs_shmem         = 0;
constexpr int fcs_blocks        = 64;
constexpr int fcs_threads       = 256;

template <uint32_t CSD16, uint32_t ESD8>
__global__ void filter_kernel(int8_t *__restrict__ data, int n, local_data_t *__restrict__ local_data);

template <uint32_t CSD16>
__global__ void filter_prepare_vec(int8_t *__restrict__ data, const int8_t *__restrict__ vec_pad16, 
                                   int *__restrict__ pairs, int n);

template <uint32_t CSD16>
__global__ void filter_collect_sol(int8_t *vec_out, uint16_t *score_out, int32_t *norm_out, uint64_t *u_out, 
                                   int *num_out, int out_max_size, int8_t *__restrict__ data, int n, int goal_score);

struct buc_traits_t {
    typedef void (*l0_buc_kernel_t)(uint32_t *, int, int8_t *, int32_t *, const int8_t *, uint32_t,
                                    const int8_t *, int *, float, int, int, int);

    static constexpr unsigned int taskChunks     = 8;
    static constexpr unsigned int taskVecs       = taskChunks * Pool_hd_t::chunk_max_nvecs;
    static constexpr double l0_center_precentile = BGJ_CENTER_IMPROVE_RATIO;

    static double  l0_buc_ratio_estimate(double alpha0, long CSD);
    static double  l0_buc_alpha0_estimate(double ratio, long CSD, double min, double max);
    static double  l0_out_max_size_ratio(double alpha0, long ESD, long CSD);
    static int32_t l0_gbuc_freq(double alpha0, long CSD);
    static int32_t l0_max_batch0_under(long lim);
    static int32_t l0_replace_threshold(Pool_hd_t *p, long sol_nvecs, double _improve_ratio);
    static int32_t num_threads(int CSD, int ESD, double alpha0, int max_batch0);
    static inline void l0_sign_copy_epi8(int8_t *dst, int8_t *src, int CSD, int sign) {
        memcpy(dst, src, CSD);
        if (sign) for (int i = 0; i < CSD; i++) dst[i] = -dst[i];
    }
    static l0_buc_kernel_t kernel_chooser(int batch0, int CSD16);


    static constexpr unsigned int vec_nbytes    = Pool_hd_t::vec_nbytes;
    static constexpr unsigned int kernelBlocks  = 32;
    static constexpr unsigned int ldWarps       = 8;
    static constexpr unsigned int blockWarps    = 8;
    static constexpr unsigned int blockThreads  = blockWarps * 32;
    
    static constexpr unsigned int buc_unit      = 512;
    static constexpr unsigned int buc_batchVecs = 64;
    
    static constexpr unsigned int l0_sbucSize   = 30;
    static constexpr unsigned int l1_sbucSize   = 30;
    static constexpr unsigned int l2_sbucSize   = 30;
    static constexpr unsigned int l1_use_if     = 0;
    static constexpr unsigned int l2_use_if     = 1;
    static constexpr unsigned int l0_shmem      = buc_batchVecs * (vec_nbytes * 3 + 4 * 2) + buc_unit * (4 * l0_sbucSize + 5);
    static constexpr unsigned int l1_shmem      = buc_batchVecs * (vec_nbytes * 3 + 4 * (l1_use_if ? 2 : 3)) + buc_unit * (4 * l1_sbucSize + 5);
    static constexpr unsigned int l2_shmem      = buc_batchVecs * (vec_nbytes * 2 + 4 * 6) + buc_unit * (4 * l2_sbucSize + 5);
};

struct red_traits_t {
    typedef void (*pk_kernel_t)(int8_t *, int8_t *, int, int);
    typedef void (*upk_kernel_t)(int8_t *, int8_t *, int, int);
    typedef void (*rpp_kernel_t)(int *, int *, int);
    #if USE_GRAPH
    typedef void (*red_kernel_t)(int *, int *, int, const int8_t *, int32_t *, int *, int32_t);
    #else
    typedef void (*rd1_kernel_t)(int *, int *, int, const int8_t *, int32_t *, int *, int32_t);
    typedef void (*red_kernel_t)(int *, int *, int, const int8_t *, int32_t *, int *, int, int32_t);
    #endif
    typedef void (*fpv_kernel_t)(int8_t *, const int8_t *, int *, int);
    typedef void (*flt_kernel_t)(int8_t *, int n, local_data_t *);
    typedef void (*fcs_kernel_t)(int8_t *, uint16_t *, int32_t *, uint64_t *, int *, int, int8_t *, int, int);
    typedef void (*bk_kernel_t)(uint32_t *, int, int8_t *, int32_t *, const int8_t *, uint32_t, const int8_t *, int *, float, int, int, int);
    
    static constexpr unsigned int taskChunks     = 8;
    static constexpr unsigned int taskVecs       = 65536;

    static int num_threads(int CSD, int ESD, int strategy);
    static int sieving_stuck(uint64_t total_check, uint64_t total_notin, long CSD);
    
    static int red_ESD8(int CSD, int ESD);
    static int red_sbuc_freq(int CSD, int strategy);
    static int red_gbuc_freq(int CSD, int strategy);
    static int bk1_gbuc_freq(int CSD, int strategy);
    static int bk2_gbuc_freq(int CSD, int strategy);
    static int bk3_gbuc_freq(int CSD, int strategy);
    static long buc_max_size(int CSD, int ESD, int strategy);
    static long red_out_max_size(int CSD, int ESD, int strategy);
    static long flt_out_max_size(int CSD, int ESD, int strategy);
    static long bk1_max_size(int CSD, int ESD, int strategy);
    static long bk2_max_size(int CSD, int ESD, int strategy);
    static long bk3_max_size(int CSD, int ESD, int strategy);
    static long l1_out_max_size(int CSD, int ESD, int strategy);
    static pk_kernel_t pk_kernel_chooser(int CSD16);
    static upk_kernel_t upk_kernel_chooser(int CSD16);
    static rpp_kernel_t rpp_kernel_chooser(int CSD16, int strategy);
    #if !USE_GRAPH
    static rd1_kernel_t rd1_kernel_chooser(int CSD16, int strategy, int sbuc_size, int gbuc_freq);
    #endif
    static red_kernel_t red_kernel_chooser(int CSD16, int strategy, int sbuc_size, int gbuc_freq);
    static fpv_kernel_t fpv_kernel_chooser(int CSD16);
    static flt_kernel_t flt_kernel_chooser(int CSD16, int ESD8);
    static fcs_kernel_t fcs_kernel_chooser(int CSD16);
    static  bk_kernel_t bk1_kernel_chooser(int CSD16, int batch1, int strategy);
    static  bk_kernel_t bk2_kernel_chooser(int CSD16, int batch2, int strategy);
    static  bk_kernel_t bk3_kernel_chooser(int CSD16, int batch3, int strategy);
    static  bk_kernel_t rep_kernel_chooser(int CSD16, int batch2, int strategy);

    static constexpr unsigned int vec_nbytes    = 176;
    static constexpr unsigned int kernelBlocks  = 64;
    static constexpr unsigned int ldWarps       = 8;
    static constexpr unsigned int blockWarps    = 8;
    static constexpr unsigned int blockThreads  = blockWarps * 32;

    static constexpr unsigned int rbuc_size     = 3;
    static constexpr unsigned int sbuc_size     = 512;
    static constexpr unsigned int red_unit      = 512;
    static constexpr unsigned int red_batchVecs = 64;

    static constexpr unsigned int l0_shmem      = red_unit * 8 + red_batchVecs * 8 + blockWarps * (4 + 4 * sbuc_size) + 
                                                  vec_nbytes * 2 * (blockWarps * 16 > red_batchVecs ? blockWarps * 16 : red_batchVecs);
    static constexpr unsigned int l1_shmem      = red_unit * 8 + red_batchVecs * 24 + blockWarps * (4 + 4 * sbuc_size) + 
                                                  vec_nbytes * 2 * (blockWarps * 16 > red_batchVecs ? blockWarps * 16 : red_batchVecs);
};

struct Bucketer_t;
struct Reducer_t;

#if ENABLE_PROFILING
struct buc_logger_t : public generic_logger_t {
    struct timeval              ev_init_time;
    struct rusage               ev_init_cpu;
    std::atomic<uint64_t>       ev_kernel_us;
    std::atomic<uint64_t>       ev_kernel_vmmas;
    std::atomic<uint64_t>       ev_h2d_us;
    std::atomic<uint64_t>       ev_h2d_nbytes;
    std::atomic<uint64_t>       ev_d2h_us;
    std::atomic<uint64_t>       ev_d2h_nbytes;
    std::atomic<uint64_t>       ev_ld_stall_us;
    std::atomic<uint64_t>       ev_ld_chunks;
    std::atomic<uint64_t>       ev_st_chunks;
    std::atomic<uint64_t>       ev_wasted_num;
    std::atomic<uint64_t>       ev_inserted_num;
    std::atomic<uint64_t>       ev_replaced_num;
    std::atomic<uint64_t>       ev_old_score_sum;
    std::atomic<uint64_t>       ev_new_score_sum;
    std::atomic<uint64_t>       ev_bk0_num;
    std::atomic<uint64_t>       ev_batch_num;
    std::atomic<uint64_t>       ev_batch_us;

    int h2d_count[BUC_DEFAULT_NUM_THREADS] = {};
    cudaEvent_t h2d_start[BUC_DEFAULT_NUM_THREADS][buc_traits_t::taskChunks];
    cudaEvent_t h2d_stop[BUC_DEFAULT_NUM_THREADS][buc_traits_t::taskChunks];
    cudaEvent_t d2h_start[BUC_DEFAULT_NUM_THREADS];
    cudaEvent_t d2h_stop[BUC_DEFAULT_NUM_THREADS];
    cudaEvent_t h2d_norm_start[BUC_DEFAULT_NUM_THREADS];
    cudaEvent_t kernel_start[BUC_DEFAULT_NUM_THREADS];

    int num_threads, num_devices, chunk_nbytes, CSD16;
    Bucketer_t *bucketer = NULL;

    inline int log_level() override { return _ll; }
    inline uint64_t log_prefix() override { return _log_prefix; }

    inline void init(const char *keyfunc) override;
    inline void exit(const char *keyfunc) override;
    inline void report(const char *keyfunc) override;

    inline void clear();

    //private:
    static constexpr int _ll = BUCKETER_LOG_LEVEL;
    static constexpr uint64_t _log_prefix = 'b' + ('u' << 8) + ('c' << 16);
};

struct red_logger_t : public generic_logger_t {
    struct timeval              ev_init_time;
    struct rusage               ev_init_cpu;

    std::atomic<uint64_t>       ev_bk1_us;
    std::atomic<uint64_t>       ev_bk2_us;
    std::atomic<uint64_t>       ev_bk3_us;
    std::atomic<uint64_t>       ev_red_us;
    std::atomic<uint64_t>       ev_bk1_vmmas;
    std::atomic<uint64_t>       ev_bk2_vmmas;
    std::atomic<uint64_t>       ev_bk3_vmmas;
    std::atomic<uint64_t>       ev_red_vmmas;
    std::atomic<uint64_t>       ev_bk0_num;
    std::atomic<uint64_t>       ev_bk1_num;
    std::atomic<uint64_t>       ev_bk2_num;
    std::atomic<uint64_t>       ev_bk3_num;
    std::atomic<uint64_t>       ev_bk0_max;
    std::atomic<uint64_t>       ev_bk1_max;
    std::atomic<uint64_t>       ev_bk2_max;
    std::atomic<uint64_t>       ev_bk3_max;
    std::atomic<uint64_t>       ev_bk0_ssum;
    std::atomic<uint64_t>       ev_bk1_ssum;
    std::atomic<uint64_t>       ev_bk2_ssum;
    std::atomic<uint64_t>       ev_bk3_ssum;
    std::atomic<uint64_t>       ev_red_max;
    std::atomic<uint64_t>       ev_red_ssum;
    std::atomic<uint64_t>       ev_red_msum;
    std::atomic<uint64_t>       ev_red_usum;
    std::atomic<uint64_t>       ev_flt_num;
    std::atomic<uint64_t>       ev_flt_max;
    std::atomic<uint64_t>       ev_flt_ssum;
    
    std::atomic<uint64_t>       ev_fff_us;
    std::atomic<uint64_t>       ev_upk_us;
    std::atomic<uint64_t>       ev_h2d_us;
    std::atomic<uint64_t>       ev_h2d_nbytes;
    std::atomic<uint64_t>       ev_d2h_us;
    std::atomic<uint64_t>       ev_d2h_nbytes;
    std::atomic<uint64_t>       ev_collect_us;
    std::atomic<uint64_t>       ev_ld_stall_us;
    std::atomic<uint64_t>       ev_ld_chunks;
    std::atomic<uint64_t>       ev_st_chunks;

    uint64_t                   *ev_total_check_ptr;
    uint64_t                   *ev_total_notin_ptr;
    int32_t                    *ev_goal_score_ptr;

    static constexpr int max_bgj3_tpb = BGJ3_DEFAULT_THREADS_PER_BUC > BGJ3L_DEFAULT_THREADS_PER_BUC ? BGJ3_DEFAULT_THREADS_PER_BUC : BGJ3L_DEFAULT_THREADS_PER_BUC;
    static constexpr int max_tpb = max_bgj3_tpb > BGJ4_DEFAULT_THREADS_PER_BUC ? max_bgj3_tpb : BGJ4_DEFAULT_THREADS_PER_BUC;
    static constexpr int id_range = RED_MAX_NUM_THREADS * max_tpb;
    static constexpr int tid_range = RED_MAX_NUM_THREADS;

    int h2d_count[id_range] = {};
    cudaEvent_t h2d_sstart[tid_range];
    cudaEvent_t h2d_sstop[tid_range];
    cudaEvent_t upk_stop[tid_range];
    cudaEvent_t h2d_start[id_range][buc_traits_t::taskChunks];
    cudaEvent_t h2d_stop[id_range][buc_traits_t::taskChunks];
    cudaEvent_t d2h_start[id_range];
    cudaEvent_t d2h_stop[id_range];
    cudaEvent_t h2d_norm_start[id_range];
    cudaEvent_t bk1_start[id_range];
    cudaEvent_t fff_start[id_range];
    cudaEvent_t fff_stop[id_range];
    cudaEvent_t bk2_start[id_range];
    cudaEvent_t bk2_stop[id_range];
    cudaEvent_t red_start[id_range];
    cudaEvent_t red_stop[id_range];
    
    int num_threads, num_devices, strategy, CSD16, chunk_nbytes;
    Reducer_t *reducer;

    inline int log_level() override { return _ll; }
    inline uint64_t log_prefix() override { return _log_prefix; }

    inline void init(const char *keyfunc) override;
    inline void exit(const char *keyfunc) override;
    inline void report(const char *keyfunc) override;

    inline void clear() {
        gettimeofday(&ev_init_time, NULL);
        getrusage(RUSAGE_SELF, &ev_init_cpu);
        ev_bk1_us = ev_bk2_us = ev_bk3_us = ev_red_us = 0;
        ev_bk1_vmmas = ev_bk2_vmmas = ev_bk3_vmmas = ev_red_vmmas = 0;
        ev_bk0_num = ev_bk1_num = ev_bk2_num = ev_bk3_num = 0;
        ev_bk0_max = ev_bk1_max = ev_bk2_max = ev_bk3_max = 0;
        ev_bk0_ssum = ev_bk1_ssum = ev_bk2_ssum = ev_bk3_ssum = 0;
        ev_red_max = ev_red_ssum = ev_red_msum = 0;
        ev_flt_max = ev_flt_ssum = ev_flt_num = 0;
        ev_fff_us = ev_upk_us = ev_h2d_us = ev_d2h_us = ev_ld_stall_us = ev_collect_us = 0;
        ev_h2d_nbytes = ev_d2h_nbytes = ev_ld_chunks = ev_st_chunks = 0;
    }

    //private:
    static constexpr int _ll = REDUCER_LOG_LEVEL;
    static constexpr uint64_t _log_prefix = 'r' + ('e' << 8) + ('d' << 16);
};
#endif


struct buc_buffer_holder_t {
    typedef buc_traits_t traits;

    buc_buffer_holder_t(Bucketer_t *bucketer);

    ~buc_buffer_holder_t();

    int device_init(int tid);
    int device_done(int tid);
    int center_prep(int batch0, int &first_batch);
    int h2d(int tid, chunk_t *chunk);
    int run(int tid);
    int out(int tid, int bid, int *num, int **entry);

    #if ENABLE_PROFILING
    buc_logger_t *logger;
    #endif

    Bucketer_t *bucketer;

    /// fixed during sieving
    double alpha0;
    long CSD, CSD16, max_batch0, gbuc_freq, out_max_size;

    /// thread & device info
    cudaStream_t *streams;
    curandState *state;
    pthread_spinlock_t gram_lock;
    long num_threads, num_devices, *used_gram;
    
    std::atomic<int64_t> pageable_ram{0};
    std::atomic<int64_t> pinned_ram{0};
    
    /// runtime data
    int *task_vecs, curr_batch0;
    int8_t *h_center16, **d_center16, **d_vec;
    int32_t **h_norm, **d_norm, **d_n;
    uint32_t **h_out, **d_out;

    void (*kernel)(uint32_t *, int, int8_t *, int32_t *, const int8_t *, uint32_t, const int8_t *, int *, float, int, int, int);
};

struct red_buffer_holder_t {
    typedef red_traits_t traits;

    red_buffer_holder_t(Reducer_t *reducer);

    ~red_buffer_holder_t();

    int device_init(int tid, int sid = -1);
    int device_done(int tid, int sid = -1);
    #if USE_GRAPH
    int graph_init(int tid, int sid = -1);
    int graph_done(int tid, int sid = -1);
    #endif
    
    int bgjl_out(int tid, int sid, int *size, int8_t **h_vec, int32_t **h_norm, uint16_t **h_score, uint64_t **h_u);
    int bgjm_out(int tid, int sid, int *size, int8_t **h_vec, int32_t **h_norm, uint16_t **h_score, uint64_t **h_u);
    int bgjs_out(int tid, int *size, int8_t **h_vec, int32_t **h_norm, uint16_t **h_score, uint64_t **h_u);
    int bgjs_h2d(int tid, chunk_t *chunk, int &used);
    int bgjs_upk(int tid);
    int bgjm_upk(int tid);

    int bgj1_run(int tid);
    int bgj2_run(int tid);
    int bgj3_run(int tid, int sid, int b);
    int bgj2_ctr(int tid, int8_t *ctr0);
    int bgj3_ctr(int tid, int8_t *ctr0);

    int bgjl_buc_ctr(int tid, int8_t *ctr0);
    int bgjl_buc_h2d(int tid, int sid, chunk_t *chunk);
    int bgjl_buc_run(int tid, int sid);
    int bgjl_buc_out(int tid, int sid, int **buc_out);
    int bgjl_ctr(int tid, int sid, int bk1_size, int *bk1_ptr, chunk_t **working_chunks);
    int bgjl_h2d(int tid, int sid, int num);
    int bgjl_upk(int tid, int sid, int ind_bias);

    int bgj3l_run(int tid, int sid);
    int bgj4_run(int tid, int sid);


    #if ENABLE_PROFILING
    red_logger_t *logger;
    #endif

    Reducer_t *reducer;

    /// fixed during sieving
    long CSD, ESD, CSD16, ESD8, strategy, sbuc_freq, gbuc_freq, bk1_gbuc_freq, bk2_gbuc_freq, bk3_gbuc_freq;
    long buc_max_size, out_max_size, flt_out_max_size, bk1_max_size, bk2_max_size, bk3_max_size, l1_out_max_size, batch1, batch2, batch3;
    double alpha0, alpha1, alpha2, alpha3;
    void (*pk_kernel)(int8_t *, int8_t *, int, int);
    void (*upk_kernel)(int8_t *, int8_t *, int, int);
    void (*rpp_kernel)(int *, int *, int);
    #if USE_GRAPH
    void (*red_kernel)(int *, int *, int, const int8_t *, int32_t *, int *, int32_t);
    #else
    void (*rd1_kernel)(int *, int *, int, const int8_t *, int32_t *, int *, int32_t);
    void (*red_kernel)(int *, int *, int, const int8_t *, int32_t *, int *, int, int32_t);
    #endif
    void (*fpv_kernel)(int8_t *, const int8_t *, int *, int);
    void (*flt_kernel)(int8_t *, int n, local_data_t *);
    void (*fcs_kernel)(int8_t *, uint16_t *, int32_t *, uint64_t *, int *, int, int8_t *, int, int);
    void (*bk1_kernel)(uint32_t *, int, int8_t *, int32_t *, const int8_t *, uint32_t, const int8_t *, int *, float, int, int, int);
    void (*bk2_kernel)(uint32_t *, int, int8_t *, int32_t *, const int8_t *, uint32_t, const int8_t *, int *, float, int, int, int);
    void (*bk3_kernel)(uint32_t *, int, int8_t *, int32_t *, const int8_t *, uint32_t, const int8_t *, int *, float, int, int, int);
    void (*rep_kernel)(uint32_t *, int, int8_t *, int32_t *, const int8_t *, uint32_t, const int8_t *, int *, float, int, int, int);

    /// thread & device info
    cudaStream_t *streams;
    cudaStream_t *sstreams = NULL;
    curandState **state, **statte;
    pthread_spinlock_t gram_lock;
    long num_threads, threads_per_buc, num_devices, *used_gram;
    #if USE_GRAPH
    /// graph
    cudaGraph_t *graphs = NULL;
    cudaGraphExec_t *graphExecs = NULL;
    cudaGraphNode_t **redKernelNodes = NULL;
    cudaKernelNodeParams **redKernelParams = NULL;
    typedef void *arg_t;
    arg_t **redKernelArgsList = NULL;
    int out_max_size_i, bk1_max_size_i, bk2_max_size_i, bk3_max_size_i;
    #endif

    std::atomic<int64_t> pageable_ram{0};
    std::atomic<int64_t> pinned_ram{0};

    /// runtime data, input and reduce
    long bgj3l_repeat, bgj4_repeat;
    int **repeat_buf = NULL, **repeat_buf_size = NULL, **h_repeat_buf = NULL;
    int *task_vecs, *buc_vecs;
    int8_t **d_upk, **d_vec16;
    int32_t **h_norm, **d_norm;
    int **d_num_red_out, **d_red_out, **d_n;

    int8_t **d_ct1 = NULL, **d_ct2 = NULL, **d_ct3 = NULL, **h_ct2 = NULL, **d_ctt1 = NULL, **h_ct1 = NULL;
    int **d_bk1 = NULL, **d_bk2 = NULL, **d_bk3 = NULL, **h_bk1 = NULL, **h_bko = NULL;

    /// runtime data, filter and output
    local_data_t **local_data;
    int **d_num_flt_out, **h_num_flt_out;
    int8_t   **d_vec_out, **h_vec_out, **data;
    uint16_t **d_score_out, **h_score_out;
    int32_t  **d_norm_out, **h_norm_out;
    uint64_t **d_u_out, **h_u_out;
};

struct buc_iterator_t {
    static constexpr int pfch_ahead = 8;

    buc_iterator_t(Bucketer_t *bucketer);

    ~buc_iterator_t();

    int reset();
    chunk_t *pop(int exist_sol, int *dst_size_limit);
    chunk_t *pop_sol(int *no_more_sol);
    int rel_sol(chunk_t *sol);
    void inserted(int chunk_id);

    #if ENABLE_PROFILING
    buc_logger_t *logger;
    #endif

    int32_t num_chunk_limit;
    int32_t last_chunk_limit;
    int32_t first_empty_chunk_id;
    int32_t last_insert_chunk_id;

    int32_t total_poped_fulls;
    int32_t first_empty_poped;
    int32_t curr_empty_id;
    int32_t curr_full_id;
    pthread_spinlock_t pop_id_lock;

    int32_t num_working_sol;
    int32_t num_reading_sol;
    chunk_t **reading_sol;
    pthread_spinlock_t reading_sol_lock;

    int CSD;
    pwc_manager_t *pwc;
    swc_manager_t *swc;
};


struct Bucketer_t {
    typedef buc_traits_t traits;
    static constexpr int32_t flag_final = 0x1;
    static constexpr int32_t flag_stuck = 0x2;

    static constexpr long   bucketer_default_num_threads    = BUC_DEFAULT_NUM_THREADS;
    static constexpr double bgj_default_saturation_radius   = BGJ_DEFAULT_SATURATION_RADIUS;
    static constexpr double bgj_default_saturation_ratio    = BGJ_DEFAULT_SATURATION_RATIO;
    static constexpr double bgj_default_improve_ratio       = BGJ_DEFAULT_IMPROVE_RATIO;

    Bucketer_t(Pool_hd_t *pool, bwc_manager_t *bwc, swc_manager_t *swc, ut_checker_t *ut_checker);

    ~Bucketer_t();

    int set_num_threads(int num_threads);
    inline void set_pool(Pool_hd_t *pool) { _pool = pool; _pwc = pool->pwc_manager; }
    inline void set_bwc_manager(bwc_manager_t *bwc_manager) { _bwc = bwc_manager; }
    inline void set_swc_manager(swc_manager_t *swc_manager) { _swc = swc_manager; }
    inline void set_ut_checker(ut_checker_t *ut_checker) { _ut_checker = ut_checker; }
    inline void set_reducer(Reducer_t *reducer) { _reducer = reducer; }

    int set_alpha0(float alpha0);
    int set_batch0(long batch0);
    int auto_bgj_params_set(int bgj);

    int run();


    #if ENABLE_PROFILING
    typedef buc_logger_t logger_t;
    logger_t *logger;
    #endif

    /// parameters
    long _bgj = 0;
    volatile double _alpha0 = 0.0;
    volatile long   _min_batch0 = 0;
    volatile long   _max_batch0 = 0;
    volatile long   _num_buc_slimit = 0;
    volatile double _saturation_radius;
    volatile double _saturation_ratio;
    volatile double _improve_ratio;

    /// runtime data
    volatile int32_t flag;
    int32_t *buc_id;
    buc_iterator_t *buc_iter;
    pthread_spinlock_t score_stat_lock;
    int8_t *ctr_record = NULL;

    /// runtime functions
    int _batch(int tid, int replace_th, int batch0);
    int _update_goal();
    int _sieve_is_over();
    int _signal_buc_done();
    int _signal_new_buc_ready();
    int _num_ready_buckets();
       

    long _num_threads = 0;
    thread_pool::thread_pool **_buc_pool = NULL;

    Pool_hd_t *_pool;
    pwc_manager_t *_pwc;
    bwc_manager_t *_bwc;
    swc_manager_t *_swc;
    ut_checker_t *_ut_checker;
    Reducer_t *_reducer;

    std::mutex _buc_mtx;
    std::condition_variable _buc_cv;

    /// limits
    volatile long   _ssd_slimit  = 0;
    volatile long   _dram_slimit = 0;
    volatile long   _gram_slimit = 0;
    volatile double _size_ratio  = 0.0;
    
    buc_buffer_holder_t *_buc_buf = NULL;
};

struct Reducer_t {
    typedef red_traits_t traits;
    static constexpr int32_t flag_stop     = 0x1;
    static constexpr int32_t flag_stop_now = 0x2;

    static constexpr int32_t strategy_bgj1  = 1;
    static constexpr int32_t strategy_bgj2  = 2;
    static constexpr int32_t strategy_bgj3  = 3;
    static constexpr int32_t strategy_bgj3l = 4;
    static constexpr int32_t strategy_bgj4  = 5;

    Reducer_t(Pool_hd_t *pool, bwc_manager_t *bwc, swc_manager_t *swc, ut_checker_t *ut_checker);

    ~Reducer_t();

    int set_num_threads(long num_threads);
    inline void set_pool(Pool_hd_t *pool) { _pool = pool; }
    inline void set_bwc_manager(bwc_manager_t *bwc_manager) { _bwc = bwc_manager; }
    inline void set_swc_manager(swc_manager_t *swc_manager) { _swc = swc_manager; }
    inline void set_ut_checker(ut_checker_t *ut_checker) { _ut_checker = ut_checker; }
    inline void set_bucketer(Bucketer_t *bucketer) { _bucketer = bucketer; }

    
    int auto_bgj_params_set(int bgj);

    int run();


    #if ENABLE_PROFILING
    typedef red_logger_t logger_t;
    logger_t *logger;
    #endif

    /// parameters
    volatile double _alpha1;
    volatile double _alpha2;
    volatile double _alpha3;
    volatile long   _batch1;
    volatile long   _batch2;
    volatile long   _batch3;
    volatile long   _threads_per_buc;
    volatile long   _num_sol_chunks_slimit;
    volatile int    _strategy;

    /// runtime data
    volatile int32_t flag = 0;
    volatile int32_t center_norm;
    volatile int32_t goal_norm;
    volatile int32_t goal_score;
    volatile int32_t bgj3_repeat;
    volatile int32_t bgj3l_repeat;
    volatile int32_t bgj4_repeat;
    pthread_spinlock_t traffic_ctrl_lock;
    volatile int32_t ld_bk0_tids = 0;

    /// runtime functions
    int _reduce(int tid);
    int _red_out_2_swc(int tid, int sid = -1);
    int _ld_sbuc(int tid, int bucket_id);
    int _ld_lbuc(int tid, int bucket_id, int &num_chunks, chunk_t **&working_chunks);
    int _red_lbuc(int tid, chunk_t **&working_chunks);
    int _signal_bucket_done();
    int _signal_red_done();
    int _signal_red_stuck();

    long _num_threads = 0;
    thread_pool::thread_pool **_red_pool = NULL;
    thread_pool::thread_pool **_sub_threads = NULL;

    Pool_hd_t *_pool;
    bwc_manager_t *_bwc;
    swc_manager_t *_swc;
    ut_checker_t *_ut_checker;
    Bucketer_t *_bucketer;

    std::mutex _red_mtx;
    std::condition_variable _red_cv;

    /// limits
    volatile long   _ssd_slimit = 0;
    volatile long   _dram_slimit = 0;
    volatile long   _gram_slimit = 0;

    red_buffer_holder_t *_red_buf = NULL;

    uint64_t total_check = 0;
    uint64_t total_notin = 0;
    pthread_spinlock_t stuck_stat_lock;
};

inline void ull_2_str(char *dst, uint64_t num) {
    if  (num < 10000ULL) sprintf(dst, "%d", (int) num);
    else if (num < 1000000ULL) sprintf(dst, "%.2fK", num * 1e-3);
    else if (num < 1000000000ULL) sprintf(dst, "%.2fM", num * 1e-6);
    else if (num < 1000000000000ULL) sprintf(dst, "%.2fG", num * 1e-9);
    else if (num < 1000000000000000ULL) sprintf(dst, "%.2fT", num * 1e-12);
    else sprintf(dst, "%.2fP", num * 1e-15);
}

#if ENABLE_PROFILING
#include "../include/bgj_hd.h"
inline void buc_logger_t::clear() {
    gettimeofday(&ev_init_time, NULL);
    getrusage(RUSAGE_SELF, &ev_init_cpu);
    ev_kernel_us = ev_h2d_us = ev_d2h_us = ev_ld_stall_us = 0;
    ev_h2d_nbytes = ev_d2h_nbytes = ev_ld_chunks = ev_st_chunks = 0;
    ev_wasted_num = ev_inserted_num = ev_replaced_num = 0;
    ev_bk0_num = ev_batch_num = ev_batch_us = 0;
    ev_kernel_vmmas = 0;
    if (bucketer) {
        bucketer->_pwc->ev_pwc_fetch.store(0);
        bucketer->_pwc->ev_pwc_cache_hit.store(0);
        bucketer->_pwc->ev_ssd_ld.store(0);
        bucketer->_pwc->ev_ssd_st.store(0);
    }
}
inline void buc_logger_t::init(const char *keyfunc) {}
inline void buc_logger_t::exit(const char *keyfunc) {}
inline void buc_logger_t::report(const char *keyfunc) {
    struct timeval ev_curr_time;
    struct rusage  ev_curr_cpu;
    gettimeofday(&ev_curr_time, NULL);
    getrusage(RUSAGE_SELF, &ev_curr_cpu);
    double elapsed = (ev_curr_time.tv_sec - ev_init_time.tv_sec) + 
                        (ev_curr_time.tv_usec - ev_init_time.tv_usec) * 1e-6;
    double cpu = (ev_curr_cpu.ru_utime.tv_sec - ev_init_cpu.ru_utime.tv_sec) + 
                    (ev_curr_cpu.ru_utime.tv_usec - ev_init_cpu.ru_utime.tv_usec) * 1e-6 + 
                    (ev_curr_cpu.ru_stime.tv_sec - ev_init_cpu.ru_stime.tv_sec) + 
                    (ev_curr_cpu.ru_stime.tv_usec - ev_init_cpu.ru_stime.tv_usec) * 1e-6;
    double avg_load = cpu / elapsed;

    int curr_bk0    = bucketer->_num_ready_buckets();
    int bk0         = ev_bk0_num.load();
    int batch       = ev_batch_num.load();
    float batch_avg = ev_batch_us.load() * 1e-6 / (float) batch;
    float bw_i      = ev_ld_chunks.load() * (float) chunk_nbytes * 1e-9 / elapsed;
    float bw_o      = ev_st_chunks.load() * (float) chunk_nbytes * 1e-9 / elapsed;
    char w_str[16], i_str[16], r_str[16];
    ull_2_str(w_str, ev_wasted_num.load());
    ull_2_str(i_str, ev_inserted_num.load());
    ull_2_str(r_str, ev_replaced_num.load());

    float ld_stall = ev_ld_stall_us.load() * 1e-6;
    float h2d      = ev_h2d_us.load() * 1e-6;
    float d2h      = ev_d2h_us.load() * 1e-6;
    float kernel   = ev_kernel_us.load() * 1e-6;
    float h2d_bw   = ev_h2d_nbytes.load() / elapsed * 1e-9;
    float d2h_bw   = ev_d2h_nbytes.load() / elapsed * 1e-9;
    float k_bw     = ev_kernel_vmmas * (float)(512.0 * CSD16) * 1e-12 / elapsed;
    float h2d_tbw  = ev_h2d_nbytes.load() / h2d * 1e-9;
    float d2h_tbw  = ev_d2h_nbytes.load() / d2h * 1e-9;
    float k_tbw    = ev_kernel_vmmas * (float)(512.0 * CSD16) * 1e-12 / kernel;

    float improve_ratio = ev_new_score_sum.load() / (double) ev_old_score_sum.load();

    this->info("#bk0 %d(curr: %d), #batch %d(avg: %.2fs), elapsed %.3fs, cpu: %.3fs(avg: %.2f), #thread %d #device %d, bw: %.2f/%.2f GB/s, w %s, i %s, r %s(%.4f)",
                bk0, curr_bk0, batch, batch_avg, elapsed, cpu, avg_load, num_threads, num_devices, bw_i, bw_o, w_str, i_str, r_str, improve_ratio);
    this->info("ld_stall %.3fs, H2D: %.2fs(%.2f(%.2f) GB/s), D2H: %.2fs(%.2f(%.2f) GB/s), kernel: %.2fs(%.2f(%.2f) TFLOPS)", 
                ld_stall, h2d, h2d_bw, h2d_tbw, d2h, d2h_bw, d2h_tbw, kernel, k_bw, k_tbw);
    if (this->bucketer) {
        char pwc_f[16], bwc_fw[16], bwc_fr[16], swc_fw[16], swc_fr[16];
        ull_2_str(pwc_f, this->bucketer->_pwc->ev_pwc_fetch.load());
        ull_2_str(bwc_fw, this->bucketer->_bwc->ev_f4w.load());
        ull_2_str(bwc_fr, this->bucketer->_bwc->ev_f4r.load());
        ull_2_str(swc_fw, this->bucketer->_swc->ev_f4w.load());
        ull_2_str(swc_fr, this->bucketer->_swc->ev_f4r.load());
        float pwc_h = this->bucketer->_pwc->ev_pwc_cache_hit.load() / (float) this->bucketer->_pwc->ev_pwc_fetch.load();
        float bwc_hw = this->bucketer->_bwc->ev_f4w_hit.load() / (float) this->bucketer->_bwc->ev_f4w.load();
        float bwc_hr = this->bucketer->_bwc->ev_f4r_hit.load() / (float) this->bucketer->_bwc->ev_f4r.load();
        float swc_hw = this->bucketer->_swc->ev_f4w_hit.load() / (float) this->bucketer->_swc->ev_f4w.load();
        float swc_hr = this->bucketer->_swc->ev_f4r_hit.load() / (float) this->bucketer->_swc->ev_f4r.load();
        float pwc_i = this->bucketer->_pwc->ev_ssd_ld.load() * (float) chunk_nbytes * 1e-9 / elapsed;
        float pwc_o = this->bucketer->_pwc->ev_ssd_st.load() * (float) chunk_nbytes * 1e-9 / elapsed;
        float bwc_i = this->bucketer->_bwc->ev_ssd_ld.load() * (float) chunk_nbytes * 1e-9 / elapsed;
        float bwc_o = this->bucketer->_bwc->ev_ssd_st.load() * (float) chunk_nbytes * 1e-9 / elapsed;
        float swc_i = this->bucketer->_swc->ev_ssd_ld.load() * (float) chunk_nbytes * 1e-9 / elapsed;
        float swc_o = this->bucketer->_swc->ev_ssd_st.load() * (float) chunk_nbytes * 1e-9 / elapsed;
        this->info("pwc (f %s(%.3f)|I %.2f|O %.2f), bwc (fw %s(%.4f)|fr %s(%.3f)|I %.2f|O %.2f), swc(fw %s(%.3f)|fr %s(%.3f)|I %.2f|O %.2f)", 
                    pwc_f, pwc_h, pwc_i, pwc_o, bwc_fw, bwc_hw, bwc_fr, bwc_hr, bwc_i, bwc_o, swc_fw, swc_hw, swc_fr, swc_hr, swc_i, swc_o);
    }
}
inline void red_logger_t::init(const char *keyfunc) {}
inline void red_logger_t::exit(const char *keyfunc) {}
inline void red_logger_t::report(const char *key_func) {
    struct timeval ev_curr_time;
    struct rusage  ev_curr_cpu;
    gettimeofday(&ev_curr_time, NULL);
    getrusage(RUSAGE_SELF, &ev_curr_cpu);
    double elapsed = (ev_curr_time.tv_sec - ev_init_time.tv_sec) + 
                        (ev_curr_time.tv_usec - ev_init_time.tv_usec) * 1e-6;
    double cpu = (ev_curr_cpu.ru_utime.tv_sec - ev_init_cpu.ru_utime.tv_sec) + 
                    (ev_curr_cpu.ru_utime.tv_usec - ev_init_cpu.ru_utime.tv_usec) * 1e-6 + 
                    (ev_curr_cpu.ru_stime.tv_sec - ev_init_cpu.ru_stime.tv_sec) + 
                    (ev_curr_cpu.ru_stime.tv_usec - ev_init_cpu.ru_stime.tv_usec) * 1e-6;
    double avg_load = cpu / elapsed;

    float pg;
    {   
        int64_t goal_num = pow(reducer->_bucketer->_saturation_radius, reducer->_pool->CSD * .5) * 
                                .5 * reducer->_bucketer->_saturation_ratio;
        int32_t goal_score = round(reducer->_pool->gh2_scaled() * .25 * reducer->_bucketer->_saturation_radius);
        int64_t real_num = 0;
        for (int i = goal_score; i > 0; i--) {
            real_num += reducer->_pool->score_stat[i];
        }
        pg = real_num / (float) goal_num * 100.f;
    }
    
    char r_str[16], rr_str[16], f_str[16], u_str[16];
    ull_2_str(r_str, ev_red_ssum.load());
    ull_2_str(rr_str, (uint64_t)(ev_red_vmmas * 256.0 / ev_red_ssum.load()));
    ull_2_str(f_str, ev_flt_ssum.load());
    ull_2_str(u_str, ev_total_notin_ptr[0]);
    float f_r = ev_red_ssum.load() / (float) ev_flt_ssum.load();
    float u_r = ev_total_check_ptr[0] / (float) ev_total_notin_ptr[0];

    float ld_stall  = ev_ld_stall_us.load() * 1e-6;
    float bw_i      = ev_ld_chunks.load() * (float) chunk_nbytes * 1e-9 / elapsed;
    float bw_o      = ev_st_chunks.load() * (float) chunk_nbytes * 1e-9 / elapsed;
    float h2d       = ev_h2d_us.load() * 1e-6;
    float d2h       = ev_d2h_us.load() * 1e-6;
    float upk       = strategy >= 1 ? ev_bk1_us.load() * 1e-6 : ev_upk_us.load() * 1e-6;
    float h2d_bw    = ev_h2d_nbytes.load() / elapsed * 1e-9;
    float d2h_bw    = ev_d2h_nbytes.load() / elapsed * 1e-9;
    float upkcount  = strategy >= 4 ? .0f : CSD16 * (float) ev_bk0_ssum.load();
    float upk_bw    = upkcount / elapsed * 1e-9;
    float h2d_tbw   = ev_h2d_nbytes.load() / h2d * 1e-9;
    float d2h_tbw   = ev_d2h_nbytes.load() / d2h * 1e-9;
    float upk_tbw   = upkcount / upk * 1e-9;
    float dup_ratio = ev_red_usum.load() / (double) (ev_red_ssum.load() + 1e-9);
    float collect   = ev_collect_us.load() * 1e-6;
    
    float fff       = ev_fff_us.load() * 1e-6;
    float fff_count = (float) ev_red_ssum.load();
    double fff_bw   = fff_count / elapsed * 1e-6;
    double fff_tbw  = fff_count / fff * 1e-6;

    char bk0_n[16], bk0_v[16], bk0_m[16], bk1_n[16], bk1_v[16], bk1_m[16], bk2_n[16], bk2_v[16], bk2_m[16], bk3_n[16], bk3_v[16], bk3_m[16], red_mr[16], red_mf[16], red_ar[16];
    ull_2_str(bk0_n, ev_bk0_num.load());
    ull_2_str(bk0_v, ev_bk0_ssum.load() / (double) ev_bk0_num.load());
    ull_2_str(bk0_m, ev_bk0_max.load());
    ull_2_str(bk1_n, ev_bk1_num.load());
    ull_2_str(bk1_v, ev_bk1_ssum.load() / (double) ev_bk1_num.load());
    ull_2_str(bk1_m, ev_bk1_max.load());
    ull_2_str(bk2_n, ev_bk2_num.load());
    ull_2_str(bk2_v, ev_bk2_ssum.load() / (double) ev_bk2_num.load());
    ull_2_str(bk2_m, ev_bk2_max.load());
    ull_2_str(bk3_n, ev_bk3_num.load());
    ull_2_str(bk3_v, ev_bk3_ssum.load() / (double) ev_bk3_num.load());
    ull_2_str(bk3_m, ev_bk3_max.load());
    ull_2_str(red_mr, ev_red_max.load());
    ull_2_str(red_mf, ev_flt_max.load());
    ull_2_str(red_ar, ev_red_msum.load() / (double) ev_flt_num.load());

    float bk1 = ev_bk1_us.load() * 1e-6;
    float bk2 = ev_bk2_us.load() * 1e-6;
    float bk3 = ev_bk3_us.load() * 1e-6;
    float red = ev_red_us.load() * 1e-6;
    float bk1_bw = ev_bk1_vmmas.load() * (float)(512.0 * CSD16) * 1e-12  / elapsed;
    float bk2_bw = ev_bk2_vmmas.load() * (float)(512.0 * CSD16) * 1e-12  / elapsed;
    float bk3_bw = ev_bk3_vmmas.load() * (float)(512.0 * CSD16) * 1e-12  / elapsed;
    float red_bw = ev_red_vmmas.load() * (float)(512.0 * CSD16) * 1e-12 / elapsed;
    float bk1_tbw = ev_bk1_vmmas.load() * (float)(512.0 * CSD16) * 1e-12 / bk1;
    float bk2_tbw = ev_bk2_vmmas.load() * (float)(512.0 * CSD16) * 1e-12 / bk2;
    float bk3_tbw = ev_bk3_vmmas.load() * (float)(512.0 * CSD16) * 1e-12 / bk3;
    float red_tbw = ev_red_vmmas.load() * (float)(512.0 * CSD16) * 1e-12 / red;

    this->info("|%.2f%|g %d|r %s(%s)|f %s(%.2f)|u %s(%.2f)|, elapsed %.3fs, cpu: %.3fs(avg: %.2f), #thread %d #device %d",
                pg, *ev_goal_score_ptr, r_str, rr_str, f_str, f_r, u_str, u_r, elapsed, cpu, avg_load, num_threads, num_devices);
    this->info("load stall: %.3fs(%.2f/%.2f GB/s), H2D: %.2fs(%.2f(%.2f) GB/s), D2H: %.2fs(%.2f(%.2f) GB/s), upk: %.2fs(%.2f(%.2f) GB/s)", 
                ld_stall, bw_i, bw_o, h2d, h2d_bw, h2d_tbw, d2h, d2h_bw, d2h_tbw, upk, upk_bw, upk_tbw);
    if (strategy == Reducer_t::strategy_bgj1) {
        this->info("bk0 (n %s|v %s|m %s), red (ar %s|mr %s|mf %s|%.2fs|%.2f|%.2f), fff: %.2fs(bw: %.2f(%.2f) M/s)", 
                    bk0_n, bk0_v, bk0_m, red_ar, red_mr, red_mf, red, red_bw, red_tbw, fff, fff_bw, fff_tbw);
    }
    if (strategy == Reducer_t::strategy_bgj2) {
        this->info("bk0 (n %s|v %s|m %s), bk1 (n %s|v %s|m %s|%.2fs|%.2f|%.2f), red (ar %s|mr %s|mf %s|%.2fs|%.2f|%.2f), fff: %.2fs(bw: %.2f(%.2f) M/s)", 
                    bk0_n, bk0_v, bk0_m, bk1_n, bk1_v, bk1_m, bk1, bk1_bw, bk1_tbw, red_ar, red_mr, red_mf, red, red_bw, red_tbw, fff, fff_bw, fff_tbw);
    }
    if (strategy == Reducer_t::strategy_bgj3 || strategy == Reducer_t::strategy_bgj3l) {
        this->info("bk0 (n %s|v %s|m %s), bk1 (n %s|v %s|m %s|%.2fs|%.2fs|%.2f|%.2f), bk2 (n %s|v %s|m %s|%.2fs|%.2f|%.2f), "
                   "red (ar %s|mr %s|mf %s|%.2fs|%.2f|%.2f), fff: %.2fs(bw: %.2f(%.2f) M/s)", bk0_n, bk0_v, bk0_m, bk1_n, bk1_v, bk1_m, bk1, collect, bk1_bw, bk1_tbw, 
                    bk2_n, bk2_v, bk2_m, bk2, bk2_bw, bk2_tbw, red_ar, red_mr, red_mf, red, red_bw, red_tbw, fff, fff_bw, fff_tbw, fff, fff_bw, fff_tbw);
    }
    if (strategy == Reducer_t::strategy_bgj4) {
        this->info("bk0 (n %s|v %s|m %s), bk1 (n %s|v %s|m %s|%.2fs|%.2fs|%.2f|%.2f), bk2 (n %s|v %s|m %s|%.2fs|%.2f|%.2f), "
                   "bk3 (n %s|v %s|m %s|%.2fs|%.2f|%.2f), red (ar %s|mr %s|mf %s|%.2fs|%.2f|%.2f), fff: %.2fs(bw: %.2f(%.2f) M/s)", bk0_n, bk0_v, bk0_m, 
                    bk1_n, bk1_v, bk1_m, bk1, collect, bk1_bw, bk1_tbw, bk2_n, bk2_v, bk2_m, bk2, bk2_bw, bk2_tbw, 
                    bk3_n, bk3_v, bk3_m, bk3, bk3_bw, bk3_tbw, red_ar, red_mr, red_mf, red, red_bw, red_tbw, fff, fff_bw, fff_tbw);
    }
}
#endif

struct random_interval_iter_t {
    random_interval_iter_t(int32_t end) {
        num = end;
        entries = new int[end];
        for (int i = 0; i < end; i++) entries[i] = i;
        std::random_device rd;
        std::mt19937 g(rd());
        for (int i = 0; i < end; i++) {
            int src = i;
            int dst = std::uniform_int_distribution<int>(0, end - 1)(g);
            int tmp = entries[src];
            entries[src] = entries[dst];
            entries[dst] = tmp;
        }
    }

    ~random_interval_iter_t() {
        delete[] entries;
    }

    inline int pop() {
        return entries[--num];
    }
    int num;
    int *entries = NULL;
};

#endif
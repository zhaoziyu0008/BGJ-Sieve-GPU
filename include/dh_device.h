#ifndef __DH_DEVICE_H
#define __DH_DEVICE_H

#include "pool_hd.h"
#include "bgj_hd.h"
#include "pool_hd_device.h"

struct dh_reducer_t;
struct dh_bucketer_t;

constexpr int hdh_nbits = 256;
constexpr int tdh_nbits = 256;
constexpr int dh_nbits = 512;
constexpr int dhb_blocks = 64;
constexpr int dhb_threads = 256;
constexpr int dhb_shmem = 101376;
constexpr int v2h_blocks = 64;
constexpr int v2h_threads = 256;
constexpr int v2h_shmem = 101376;
constexpr int dhr_blocks = 64;
constexpr int dhr_threads = 256;
constexpr int dhr_shmem = 16 * dh_nbits + 513 * 32;

struct dh_data_t {
    int8_t  b_dual[16896];
    float   b_head[48 * 176];
    float   v_dual[49 * hdh_nbits];
    int     dhalf;
    int     dshift;
};

template <uint32_t CSD16, uint32_t ESD8>
__global__ void dh_buc_kernel(uint32_t *__restrict__ out, int out_max_size, float *__restrict__ center, float radius, 
                              int batch, int8_t *__restrict__ in, int n, int CSD, dh_data_t *__restrict__ data);

template <uint32_t CSD16, uint32_t ESD8, uint32_t ttype>
__global__ void dh_v2h_kernel(uint32_t *__restrict__ out, int8_t *__restrict__ in, int8_t *__restrict__ vec_pad16, 
                              int n, int CSD, dh_data_t *__restrict__ data);

__global__ void dh_red_kernel(int *__restrict__ out, int *num_out, int out_max_size, int *in, int n, int th);

template <uint32_t CSD16>
__global__ void filter_prepare_vec(int8_t *__restrict__ data, const int8_t *__restrict__ vec_pad16, 
                                   int *__restrict__ pairs, int n);

struct dh_traits_t {
    typedef void (*dhb_kernel_t)(uint32_t *, int, float *, float, int, int8_t *, int, int, dh_data_t *);
    typedef void (*v2h_kernel_t)(uint32_t *, int8_t *, int8_t *, int, int, dh_data_t *);
    typedef void (*dhr_kernel_t)(int *, int *, int, int *, int, int);
    typedef void (*fpv_kernel_t)(int8_t *, const int8_t *, int *, int);
    typedef void (*mlf_kernel_t)(int8_t *, int, local_data_t *);
    

    static constexpr int taskChunks = 8;
    static constexpr int taskVecs   = taskChunks * Pool_hd_t::chunk_max_nvecs;
    static constexpr int pfch_ahead = 64;
    
    static inline int max_batch_under(long lim) {
        return (lim / 256) * 256;
    }
    static inline int buc_num_threads(int CSD) {
        return DHB_DEFAULT_NUM_THREADS;
    }
    static inline int red_num_threads(int CSD, int ESD) {
        double beta = pow((DH_BSIZE_RATIO / sqrt(3.2 * pow(4./3., CSD * .5))), 1.0 / ESD);
        if (beta > 0.95) beta = 0.95;

        long out_max_size = dhr_out_max_size(CSD);
        long buc_max_size = DH_BSIZE_RATIO * sqrt(3.2 * pow(4./3., CSD * .5)) * 
                    (6.9367057575991202 - 0.0254642974101921 * ESD - 6.1327667887667809 * beta) * 1.1;
        if (buc_max_size > 3.2 * pow(4./3., CSD * .5)) buc_max_size = 3.2 * pow(4./3., CSD * .5);
        long gram_per_threads = out_max_size * 8L + buc_max_size * (long)(dh_nbits / 8 + 176) + taskVecs * 176;
        long threads_per_device = floor(DHR_GRAM_SLIMIT / gram_per_threads);
        int ret = threads_per_device * hw::gpu_num;
        if (ret > DHR_DEFAULT_NUM_THREADS) ret = DHR_DEFAULT_NUM_THREADS;
        return ret;
    }
    static inline int dh_threshold(int CSD, int ESD) {
        return 199;
    }
    static inline int dhr_out_max_size(int CSD) {
        /// @todo better estimation
        return 67108864;
    }
    static inline dhb_kernel_t dhb_kernel_chooser(int CSD16, int ESD8) {
        if (CSD16 != 176) return nullptr;
        if (ESD8 == 32) return dh_buc_kernel<176, 32>;
        if (ESD8 == 40) return dh_buc_kernel<176, 40>;
        if (ESD8 == 48) return dh_buc_kernel<176, 48>;
        return nullptr;
    }
    static inline v2h_kernel_t v2h_kernel_chooser(int CSD16, int ESD8, int ttype) {
        if (CSD16 != 176) return nullptr;
        if (ttype == 0) {
            if (ESD8 == 32) return dh_v2h_kernel<176, 32, 0>;
            if (ESD8 == 40) return dh_v2h_kernel<176, 40, 0>;
            if (ESD8 == 48) return dh_v2h_kernel<176, 48, 0>;
        }
        if (ttype == 1) {
            if (ESD8 == 32) return dh_v2h_kernel<176, 32, 1>;
            if (ESD8 == 40) return dh_v2h_kernel<176, 40, 1>;
            if (ESD8 == 48) return dh_v2h_kernel<176, 48, 1>;
        }
        if (ttype == 2) {
            if (ESD8 == 32) return dh_v2h_kernel<176, 32, 2>;
            if (ESD8 == 40) return dh_v2h_kernel<176, 40, 2>;
            if (ESD8 == 48) return dh_v2h_kernel<176, 48, 2>;
        }
        if (ttype == 3) {
            if (ESD8 == 32) return dh_v2h_kernel<176, 32, 3>;
            if (ESD8 == 40) return dh_v2h_kernel<176, 40, 3>;
            if (ESD8 == 48) return dh_v2h_kernel<176, 48, 3>;
        }
        if (ttype == 4) {
            if (ESD8 == 32) return dh_v2h_kernel<176, 32, 4>;
            if (ESD8 == 40) return dh_v2h_kernel<176, 40, 4>;
            if (ESD8 == 48) return dh_v2h_kernel<176, 48, 4>;
        }
        if (ttype == 5) {
            if (ESD8 == 32) return dh_v2h_kernel<176, 32, 5>;
            if (ESD8 == 40) return dh_v2h_kernel<176, 40, 5>;
            if (ESD8 == 48) return dh_v2h_kernel<176, 48, 5>;
        }
        if (ttype == 6) {
            if (ESD8 == 32) return dh_v2h_kernel<176, 32, 6>;
            if (ESD8 == 40) return dh_v2h_kernel<176, 40, 6>;
            if (ESD8 == 48) return dh_v2h_kernel<176, 48, 6>;
        }
        if (ttype == 7) {
            if (ESD8 == 32) return dh_v2h_kernel<176, 32, 7>;
            if (ESD8 == 40) return dh_v2h_kernel<176, 40, 7>;
            if (ESD8 == 48) return dh_v2h_kernel<176, 48, 7>;
        }
        if (ttype == 8) {
            if (ESD8 == 32) return dh_v2h_kernel<176, 32, 8>;
            if (ESD8 == 40) return dh_v2h_kernel<176, 40, 8>;
            if (ESD8 == 48) return dh_v2h_kernel<176, 48, 8>;
        }
        if (ttype == 9) {
            if (ESD8 == 32) return dh_v2h_kernel<176, 32, 9>;
            if (ESD8 == 40) return dh_v2h_kernel<176, 40, 9>;
            if (ESD8 == 48) return dh_v2h_kernel<176, 48, 9>;
        }
        if (ttype == 10) {
            if (ESD8 == 32) return dh_v2h_kernel<176, 32, 10>;
            if (ESD8 == 40) return dh_v2h_kernel<176, 40, 10>;
            if (ESD8 == 48) return dh_v2h_kernel<176, 48, 10>;
        }
        if (ttype == 11) {
            if (ESD8 == 32) return dh_v2h_kernel<176, 32, 11>;
            if (ESD8 == 40) return dh_v2h_kernel<176, 40, 11>;
            if (ESD8 == 48) return dh_v2h_kernel<176, 48, 11>;
        }
        if (ttype == 12) {
            if (ESD8 == 32) return dh_v2h_kernel<176, 32, 12>;
            if (ESD8 == 40) return dh_v2h_kernel<176, 40, 12>;
            if (ESD8 == 48) return dh_v2h_kernel<176, 48, 12>;
        }
        if (ttype == 13) {
            if (ESD8 == 32) return dh_v2h_kernel<176, 32, 13>;
            if (ESD8 == 40) return dh_v2h_kernel<176, 40, 13>;
            if (ESD8 == 48) return dh_v2h_kernel<176, 48, 13>;
        }
        if (ttype == 14) {
            if (ESD8 == 32) return dh_v2h_kernel<176, 32, 14>;
            if (ESD8 == 40) return dh_v2h_kernel<176, 40, 14>;
            if (ESD8 == 48) return dh_v2h_kernel<176, 48, 14>;
        }
        if (ttype == 15) {
            if (ESD8 == 32) return dh_v2h_kernel<176, 32, 15>;
            if (ESD8 == 40) return dh_v2h_kernel<176, 40, 15>;
            if (ESD8 == 48) return dh_v2h_kernel<176, 48, 15>;
        }
        
        return nullptr;
    }
    static inline dhr_kernel_t dhr_kernel_chooser() {
        return dh_red_kernel;
    }
    static inline fpv_kernel_t fpv_kernel_chooser(int CSD16) {
        if (CSD16 != 176) return nullptr;
        return filter_prepare_vec<176>;
    }
    static inline mlf_kernel_t mlf_kernel_chooser(int CSD16, int ESD8) {
        if (CSD16 != 176) return nullptr;
        if (ESD8 == 32) return min_lift_kernel<176, 32>;
        if (ESD8 == 40) return min_lift_kernel<176, 40>;
        if (ESD8 == 48) return min_lift_kernel<176, 48>;
        return nullptr;
    }
    static inline void sign_copy_epi8(int8_t *dst, int8_t *src, int CSD, int sign) {
        memcpy(dst, src, CSD);
        if (sign) for (int i = 0; i < CSD; i++) dst[i] = -dst[i];
    }
};


#if ENABLE_PROFILING
struct dh_buc_logger_t : public generic_logger_t {
    struct timeval              ev_init_time;
    struct rusage               ev_init_cpu;
    std::atomic<uint64_t>       ev_kernel_us;
    std::atomic<uint64_t>       ev_h2d_us;
    std::atomic<uint64_t>       ev_d2h_us;
    std::atomic<uint64_t>       ev_ld_stall_us;
    std::atomic<uint64_t>       ev_h2d_nbytes;
    std::atomic<uint64_t>       ev_d2h_nbytes;
    std::atomic<uint32_t>       ev_total_bucket;
    std::atomic<uint64_t>       ev_total_256ops;
    int                         ev_total_batch;
    uint64_t                    ev_total_batch_us;

    int h2d_count[DHB_DEFAULT_NUM_THREADS] = {};
    cudaEvent_t h2d_start[DHB_DEFAULT_NUM_THREADS][dh_traits_t::taskChunks];
    cudaEvent_t h2d_stop[DHB_DEFAULT_NUM_THREADS][dh_traits_t::taskChunks];
    cudaEvent_t d2h_start[DHB_DEFAULT_NUM_THREADS];
    cudaEvent_t d2h_stop[DHB_DEFAULT_NUM_THREADS];
    cudaEvent_t kernel_start[DHB_DEFAULT_NUM_THREADS];

    int num_threads, num_devices, chunk_nbytes;
    dh_bucketer_t *bucketer = NULL;

    inline int log_level() override { return _ll; }
    inline uint64_t log_prefix() override { return _log_prefix; }

    inline void init(const char *keyfunc) override;
    inline void exit(const char *keyfunc) override;
    inline void report(const char *keyfunc) override;

    inline void clear();

    //private:
    static constexpr int _ll = BUCKETER_LOG_LEVEL;
    static constexpr uint64_t _log_prefix = 'd' + ('h' << 8) + ('b' << 16);
};

struct dh_red_logger_t : public generic_logger_t {
    struct timeval              ev_init_time;
    struct rusage               ev_init_cpu;

    std::atomic<uint64_t>       ev_v2h_us;
    std::atomic<uint64_t>       ev_dhr_us;
    std::atomic<uint64_t>       ev_mlf_us;
    std::atomic<uint64_t>       ev_v2h_256ops;
    std::atomic<uint64_t>       ev_dhr_256ops;
    std::atomic<uint64_t>       ev_mlf_256ops;
    std::atomic<uint64_t>       ev_dhb_num;
    std::atomic<uint64_t>       ev_dhb_ssum;

    std::atomic<uint64_t>       ev_ld_stall_us;
    std::atomic<uint64_t>       ev_h2d_us;
    std::atomic<uint64_t>       ev_h2d_nbytes;

    cudaEvent_t h2d_start[DHR_DEFAULT_NUM_THREADS];
    cudaEvent_t h2d_stop[DHR_DEFAULT_NUM_THREADS];
    cudaEvent_t dhr_start[DHR_DEFAULT_NUM_THREADS];
    cudaEvent_t dhr_stop[DHR_DEFAULT_NUM_THREADS];
    cudaEvent_t v2h_start[DHR_DEFAULT_NUM_THREADS];
    cudaEvent_t v2h_stop[DHR_DEFAULT_NUM_THREADS];
    cudaEvent_t mlf_start[DHR_DEFAULT_NUM_THREADS];
    cudaEvent_t mlf_stop[DHR_DEFAULT_NUM_THREADS];

    int num_threads, num_devices;

    inline int log_level() override { return _ll; }
    inline uint64_t log_prefix() override { return _log_prefix; }

    inline void init(const char *keyfunc) override;
    inline void exit(const char *keyfunc) override;
    inline void report(const char *keyfunc) override;

    inline void clear();

    //private:
    static constexpr int _ll = REDUCER_LOG_LEVEL;
    static constexpr uint64_t _log_prefix = 'd' + ('h' << 8) + ('r' << 16);
};
#endif

void set_dh_head(float *dh_head_val, cudaStream_t stream);
void set_dh_inorm(float *dh_inorm_val, cudaStream_t stream);


struct dhb_buffer_t {
    typedef dh_traits_t traits;

    dhb_buffer_t(dh_bucketer_t *bucketer);

    ~dhb_buffer_t();

    int device_init(int tid);
    int device_done(int tid);
    int center_prep(int batch);
    int h2d(int tid, chunk_t *chunk);
    int run(int tid);
    int out(int tid, int bid, int *num, int **entry);

    #if ENABLE_PROFILING
    dh_buc_logger_t *logger;
    #endif

    dh_bucketer_t *bucketer;

    /// fixed during dh
    float radius;
    long CSD, CSD16, ESD, ESD8, max_batch, out_max_size;

    /// thread & device info
    cudaStream_t *streams;
    pthread_spinlock_t gram_lock;
    long num_threads, num_devices, *used_gram;

    std::atomic<int64_t> pageable_ram{0};
    std::atomic<int64_t> pinned_ram{0};

    /// runtime data
    int *task_vecs, curr_batch;
    int8_t **d_vec, **d_upk;
    uint32_t **h_out, **d_out;
    float *h_center, **d_center;
    dh_data_t *h_data, **d_data;
    local_data_t **bml_data;

    static int center_sampling(float *dst, Pool_hd_t *p, int num, int ESD8);
    static int dh_data_prepare(dh_data_t *dst, Pool_hd_t *p, int CSD16, int ESD8);

    void (*dhb_kernel)(uint32_t *, int, float *, float, int, int8_t *, int, int, dh_data_t *);
    void (*bml_kernel)(int8_t *, int, local_data_t *);
};

struct dhr_buffer_t {
    typedef dh_traits_t traits;

    dhr_buffer_t(dh_reducer_t *reducer, double target_length);

    ~dhr_buffer_t();
    
    int device_init(int tid);
    int device_done(int tid);
    int h2d(int tid, chunk_t *chunk, int &used);
    int upk(int tid);
    int run(int tid);
    int out(int tid);

    #if ENABLE_PROFILING
    dh_red_logger_t *logger;
    #endif

    dh_reducer_t *reducer;

    /// fixed during dh
    long CSD, CSD16, ESD, ESD8, th, buc_max_size, out_max_size; 
    double target_length;
    long report_range;

    /// thread & device info
    cudaStream_t *streams;
    pthread_spinlock_t gram_lock;
    long num_threads, num_devices, *used_gram;

    std::atomic<int64_t> pageable_ram{0};
    std::atomic<int64_t> pinned_ram{0};

    /// runtime data
    pthread_spinlock_t min_lock;
    int *task_vecs, *buc_vecs;
    int8_t **d_upk, **d_vec16;
    int **d_dh, **d_num_out, **d_out;
    local_data_t **local_data;
    dh_data_t *h_data, **d_data;

    static int v_dual_sampling(float *dst, Pool_hd_t *p, int num, int ESD8);

    void (*v2h_kernel[DHR_DEFAULT_NUM_THREADS])(uint32_t *, int8_t *, int8_t *, int, int, dh_data_t *);
    void (*dhr_kernel)(int *, int *, int, int *, int, int);
    void (*fpv_kernel)(int8_t *, const int8_t *, int *, int);
    void (*mlf_kernel)(int8_t *, int, local_data_t *);

    int **h_res, *res;
    MAT_QP b_trans_QP;
    VEC_QP *v_QP = NULL;
    struct timeval last_report;
};



struct dh_bucketer_t {
    typedef dh_traits_t traits;

    dh_bucketer_t(Pool_hd_t *pool, bwc_manager_t *bwc);
    ~dh_bucketer_t();

    int set_num_threads(int num_threads);
    inline void set_reducer(dh_reducer_t *reducer) { _reducer = reducer; }

    int set_beta(double beta);
    int set_min_batch(long min_batch);
    int set_max_batch(long max_batch);
    int set_num_buc_slimit(long num_buc_slimit);
    int auto_bgj_params_set();

    int run(double max_time);


    #if ENABLE_PROFILING
    typedef dh_buc_logger_t logger_t;
    logger_t *logger;
    #endif

    /// parameters
    volatile double _beta = 0.0;
    volatile long   _min_batch = 0;
    volatile long   _max_batch = 0;
    volatile long   _num_buc_slimit = 0;

    /// runtime data
    volatile long init_round = 1;

    /// runtime functions

    long _num_threads = 0;
    thread_pool::thread_pool **_buc_pool = NULL;
    Pool_hd_t *_pool;
    pwc_manager_t *_pwc;
    bwc_manager_t *_bwc;
    dh_reducer_t *_reducer;

    std::mutex _dhb_mtx;
    std::condition_variable _dhb_cv;
    
    dhb_buffer_t *_buc_buf = NULL;
};

struct dh_reducer_t {
    typedef dh_traits_t traits;
    static constexpr int32_t flag_stop     = 0x1;
    static constexpr int32_t flag_stop_now = 0x2;

    dh_reducer_t(Pool_hd_t *pool, bwc_manager_t *bwc, double eta, int force_one);
    ~dh_reducer_t();

    int set_num_threads(int num_threads);
    inline void set_bucketer(dh_bucketer_t *bucketer) { _bucketer = bucketer; }

    int auto_bgj_params_set();

    int run(double target_length);
    int *get_result();


    #if ENABLE_PROFILING
    typedef dh_red_logger_t logger_t;
    logger_t *logger;
    #endif

    /// parameters
    volatile double _eta;
    volatile int    _force_one;

    /// runtime data
    volatile int32_t flag = 0;
    volatile int32_t device_inited = 0;

    /// runtime functions

    long _num_threads = 0;
    thread_pool::thread_pool **_red_pool = NULL;
    Pool_hd_t *_pool;
    bwc_manager_t *_bwc;
    dh_bucketer_t *_bucketer;

    std::mutex _dhr_mtx;
    std::condition_variable _dhr_cv;

    dhr_buffer_t *_red_buf = NULL;
};

#if ENABLE_PROFILING
#include "../include/bgj_hd_device.h"
inline void dh_buc_logger_t::clear() {
    gettimeofday(&ev_init_time, NULL);
    getrusage(RUSAGE_SELF, &ev_init_cpu);
    ev_kernel_us = ev_h2d_us = ev_d2h_us = ev_ld_stall_us = 0;
    ev_h2d_nbytes = ev_d2h_nbytes = ev_total_batch = 0;
    ev_total_256ops = 0.0;
    if (bucketer) {
        bucketer->_pwc->ev_pwc_fetch.store(0);
        bucketer->_pwc->ev_pwc_cache_hit.store(0);
        bucketer->_pwc->ev_ssd_ld.store(0);
        bucketer->_pwc->ev_ssd_st.store(0);
    }
}
inline void dh_buc_logger_t::init(const char *keyfunc) {}
inline void dh_buc_logger_t::exit(const char *keyfunc) {}
inline void dh_buc_logger_t::report(const char *keyfunc) {
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
    double ld_stall = ev_ld_stall_us * 1e-6;
    double bw = ev_h2d_nbytes / elapsed * 0x1p-30;
    double h2d = ev_h2d_us * 1e-6;
    double d2h = ev_d2h_us * 1e-6;
    double kernel = ev_kernel_us * 1e-6;
    double h2d_bw = ev_h2d_nbytes / (ev_h2d_us * 1e-6) * 0x1p-30;
    double d2h_bw = ev_d2h_nbytes / (ev_d2h_us * 1e-6) * 0x1p-30;
    double kernel_bw = ev_h2d_nbytes / (ev_kernel_us * 1e-6) * 0x1p-30;
    double avg_batch_time = ev_total_batch_us * 1e-6 / ev_total_batch;

    double ev_total_mops = ev_total_256ops.load() * 0.000000256;

    this->info("%u bucket (%d batch, avg: %.2f s) done, elapsed: %.3fs, cpu: %.3fs(avg: %.2f), #thread %d, #device %d, bw: %.2f GB/s",
                ev_total_bucket.load(), ev_total_batch, avg_batch_time, elapsed, cpu, avg_load, num_threads, num_devices, bw);
    this->info("dhb: %.3fT, %.2f(%.2f) G/s, load stall: %.3fs, H2D: %.3fs(bw: %.2f GB/s), D2H: %.3fs(bw: %.2f GB/s), kernel: %.3fs(bw: %.2f GB/s)", 
                ev_total_mops / 1e3, ev_total_mops / kernel, ev_total_mops / elapsed, ld_stall, h2d, h2d_bw, d2h, d2h_bw, kernel, kernel_bw);
    if (this->bucketer) {
        char pwc_f[16], bwc_fw[16], bwc_fr[16];
        ull_2_str(pwc_f, this->bucketer->_pwc->ev_pwc_fetch.load());
        ull_2_str(bwc_fw, this->bucketer->_bwc->ev_f4w.load());
        ull_2_str(bwc_fr, this->bucketer->_bwc->ev_f4r.load());
        float pwc_h = this->bucketer->_pwc->ev_pwc_cache_hit.load() / (float) this->bucketer->_pwc->ev_pwc_fetch.load();
        float bwc_hw = this->bucketer->_bwc->ev_f4w_hit.load() / (float) this->bucketer->_bwc->ev_f4w.load();
        float bwc_hr = this->bucketer->_bwc->ev_f4r_hit.load() / (float) this->bucketer->_bwc->ev_f4r.load();
        float pwc_i = this->bucketer->_pwc->ev_ssd_ld.load() * (float) chunk_nbytes * 1e-9 / elapsed;
        float pwc_o = this->bucketer->_pwc->ev_ssd_st.load() * (float) chunk_nbytes * 1e-9 / elapsed;
        float bwc_i = this->bucketer->_bwc->ev_ssd_ld.load() * (float) chunk_nbytes * 1e-9 / elapsed;
        float bwc_o = this->bucketer->_bwc->ev_ssd_st.load() * (float) chunk_nbytes * 1e-9 / elapsed;
        this->info("pwc (f %s(%.3f)|I %.2f|O %.2f), bwc (fw %s(%.4f)|fr %s(%.3f)|I %.2f|O %.2f)", 
                    pwc_f, pwc_h, pwc_i, pwc_o, bwc_fw, bwc_hw, bwc_fr, bwc_hr, bwc_i, bwc_o);
    }
}
inline void dh_red_logger_t::clear() {
    gettimeofday(&ev_init_time, NULL);
    getrusage(RUSAGE_SELF, &ev_init_cpu);
    ev_v2h_us = ev_dhr_us = ev_mlf_us = 0;
    ev_v2h_256ops = ev_dhr_256ops = ev_mlf_256ops = 0.0;
    ev_ld_stall_us = ev_h2d_us = ev_h2d_nbytes = 0;
    ev_dhb_num = ev_dhb_ssum = 0;
}
inline void dh_red_logger_t::init(const char *keyfunc) {}
inline void dh_red_logger_t::exit(const char *keyfunc) {}
inline void dh_red_logger_t::report(const char *keyfunc) {
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

    double dhr_gops = ev_dhr_256ops.load() * 0.000000256;
    double mlf_mops = ev_mlf_256ops.load() * 0.000256;
    double v2h_mops = ev_v2h_256ops.load() * 0.000256;

    double dhr = dhr_gops;
    double mlf = mlf_mops * 1e-3;
    char dh_ratio_str[16];
    ull_2_str(dh_ratio_str, (uint64_t) (dhr / mlf));
    
    double ld_stall = ev_ld_stall_us * 1e-6;
    double h2d = ev_h2d_us * 1e-6;
    double h2d_bw = ev_h2d_nbytes / h2d * 0x1p-30;
    double avg_dhb_size = ev_dhb_ssum / (double) ev_dhb_num;
    char avg_dhb_size_str[16];
    ull_2_str(avg_dhb_size_str, (uint64_t)avg_dhb_size);
    double dhr_bw = dhr_gops / (ev_dhr_us * 1e-6);
    double dhr_rbw = dhr_gops / elapsed;
    double mlf_bw = mlf_mops / (ev_mlf_us * 1e-6);
    double mlf_rbw = mlf_mops / elapsed;
    double v2h_bw = v2h_mops / (ev_v2h_us * 1e-6);
    double v2h_rbw = v2h_mops / elapsed;

    this->info("dhr: %.3fT, mlf: %.3fG(%s), elapsed: %.3fs, cpu: %.3fs(avg: %.2f), #threads %d, H2D: %.3fs(%.2f GB/s)",
                dhr * 1e-3, mlf, dh_ratio_str, elapsed, cpu, avg_load, num_threads, h2d, h2d_bw);
    this->info("ld stall: %.3fs, dhr (%s|%.2f|%.2f), mlf (%.2f|%.2f), v2h (%.2f|%.2f)",
                ld_stall, avg_dhb_size_str, dhr_bw, dhr_rbw, mlf_bw, mlf_rbw, v2h_bw, v2h_rbw);
}
#endif

#endif
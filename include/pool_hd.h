#ifndef __POOL_HD_H
#define __POOL_HD_H

#include "lattice.h"
#include "sampler.h"
#include "UidTable.h"
#include "../dep/g6k/thread_pool.hpp"


#include <atomic>
#include <thread>

void _start_ck_allocator();
void _destory_ck_allocator();

struct cudaDeviceProp;
struct local_data_t;
struct boost_data_t;
struct dhb_buffer_t;
struct dhr_buffer_t;
template <class logger_t> struct pwc_manager_tmpl;
template <class logger_t> struct bwc_manager_tmpl;
template <class logger_t> struct swc_manager_tmpl;
#if ENABLE_PROFILING
struct pwc_logger_t;
struct bwc_logger_t;
struct swc_logger_t;
typedef pwc_manager_tmpl<pwc_logger_t> pwc_manager_t;
typedef bwc_manager_tmpl<bwc_logger_t> bwc_manager_t;
typedef swc_manager_tmpl<swc_logger_t> swc_manager_t;
#else
typedef pwc_manager_tmpl<int> pwc_manager_t;
typedef bwc_manager_tmpl<int> bwc_manager_t;
typedef swc_manager_tmpl<int> swc_manager_t;
#endif

#include <sys/time.h>

#if ENABLE_PROFILING
#include <unistd.h>
#include <sys/resource.h>

#define cond_lg_init(_cond)                         \
    struct timeval              __ev_init_time;     \
    if ((logger->_log_prefix & 0xff00) == 0x7700) { \
        if (logger->_ll >= 3 && (_cond)) {          \
            gettimeofday(&__ev_init_time, NULL);    \
            if (logger->_ll >= 4) {                 \
                if (__FUNCTION__[0] != '_') logger->dbg("\033[32menter keyfunc %s", __FUNCTION__);  \
                else logger->dbg("\033[35menter keyfunc %s", __FUNCTION__); \
            }                                       \
        }                                           \
    } else if (_cond) logger->init(__FUNCTION__);   \

#define cond_lg_exit(_cond)                                     \
    struct timeval              __ev_exit_time;                 \
    if ((logger->_log_prefix & 0xff00) == 0x7700 && (_cond)) {  \
        double elapsed;                                         \
        if (logger->_ll >= 3) {                                 \
            gettimeofday(&__ev_exit_time, NULL);                \
            elapsed = (__ev_exit_time.tv_sec - __ev_init_time.tv_sec) +                          \
                      (__ev_exit_time.tv_usec - __ev_init_time.tv_usec) * 1e-6;                  \
            if (logger->_ll >= 4) {                                                              \
                if (__FUNCTION__[0] != '_')                                                      \
                    logger->dbg("\033[32mexit keyfunc %s, %.3fms", __FUNCTION__, elapsed * 1e3); \
                else logger->dbg("\033[35mexit keyfunc %s, %.3fms", __FUNCTION__, elapsed * 1e3);\
            }                                                                                    \
        }                                                                                        \
    } else if (_cond) logger->exit(__FUNCTION__)

#define lg_exit() cond_lg_exit(true)
#define lg_init() cond_lg_init(true)

#define lg_report()        logger->report(__FUNCTION__)
#define lg_err(_fmt, ...)  if (logger->_ll >= 0) logger->err("%s: " _fmt, __FUNCTION__, ##__VA_ARGS__)
#define lg_warn(_fmt, ...) if (logger->_ll >= 1) logger->warn("%s: " _fmt, __FUNCTION__, ##__VA_ARGS__)
#define lg_info(_fmt, ...) if (logger->_ll >= 2) logger->info("%s: " _fmt, __FUNCTION__, ##__VA_ARGS__)
#define lg_dbg(_fmt, ...)  if (logger->_ll >= 3) logger->dbg("%s: " _fmt, __FUNCTION__, ##__VA_ARGS__)

constexpr bool str_equal(const char *a, const char *b) {
    return *a == *b && (*a == '\0' || str_equal(a + 1, b + 1));
}

#include <cstdarg>

struct generic_logger_t {
    static constexpr int ll_err  = 1;
    static constexpr int ll_warn = 2;
    static constexpr int ll_info = 3;
    static constexpr int ll_dbg  = 4;

    generic_logger_t();
    ~generic_logger_t();

    inline virtual int log_level() { return _ll; }
    inline virtual uint64_t log_prefix() { return _log_prefix; }

    inline void stop() {
        this->_stop_log = 1;
        if (this->log_thread) {
            this->log_thread->join();
            this->_stop_log = 0;
            this->info("logger_t::stop: logger stopped");
            this->_stop_log = 1;
            delete this->log_thread;
            this->log_thread = NULL;
        }
    }
    inline void start() {
        this->_stop_log = 0;
        if (this->log_thread) this->err("logger_t::start: logger already started");
        else this->log_thread = new std::thread([this](){
            double elapsed = 0.0;
            while (!this->_stop_log) {
                usleep(100000);
                elapsed += 0.1;
                if (elapsed > AUTO_REPORT_DURATION) {
                    elapsed = 0.0;
                    this->report(this->_curr_keyfunc);
                }
            }
        });
        this->info("logger_t::start: logger started");
    }

    inline FILE *log_out() { return _log_out; }
    inline FILE *log_err() { return _log_err; }
    inline int set_log_out(FILE *file) {
        if (file == NULL) {
            this->err("logger_t::set_log_out: ignored NULL input");
            return -1;
        }
        if (_log_out != stdout && _log_out != stderr) fclose(_log_out);
        this->_log_out = file;
        return 0;
    }
    inline int set_log_err(FILE *file) {
        if (file == NULL) {
            this->err("logger_t::set_log_err: ignored NULL input");
            return -1;
        }
        if (_log_err != stdout && _log_err != stderr) fclose(_log_err);
        this->_log_err = file;
        return 0;
    }
    inline int set_log_out(const char *filename){
        FILE *file = fopen(filename, "a");
        if (file == NULL) {
            this->err("logger_t::set_log_out: fail to open %s, %s", filename, strerror(errno));
            return -1;
        }
        /// if (_log_out != stdout && _log_out != stderr) fclose(_log_out);
        this->_log_out = file;
        this->info("logger_t::set_log_out: switch log_out to file \'%s\'", filename);
        return 0;
    }
    inline int set_log_err(const char *filename) {
        FILE *file = fopen(filename, "a");
        if (file == NULL) {
            this->err("logger_t::set_log_err: fail to open %s, %s", filename, strerror(errno));
            return -1;
        }
        /// if (_log_err != stdout && _log_err != stderr) fclose(_log_err);
        this->_log_err = file;
        this->info("logger_t::set_log_err: switch log_err to file \'%s\'", filename);
        return 0;
    }
    
    inline void err(const char *__format, ...) {
        if (log_level() < ll_err || _stop_log) return;

        char buf[4096];
        va_list args;
        va_start(args, __format);
        int nbytes = vsnprintf(buf, sizeof(buf), __format, args);
        va_end(args);

        if (nbytes < 0 || nbytes >= sizeof(buf)) this->warn("logger_t::err: message truncated");

        struct timeval tv;
        gettimeofday(&tv, NULL);
        struct tm tm_info;
        localtime_r(reinterpret_cast<time_t *>(&tv.tv_sec), &tm_info);
        char time_buf[64];

        strftime(time_buf, sizeof(time_buf), "%y-%m-%d %H:%M:%S", &tm_info);
        snprintf(time_buf + strlen(time_buf), sizeof(time_buf) - strlen(time_buf), ".%03ld", tv.tv_usec / 1000);

        uint64_t prefix = this->log_prefix();
        fprintf(_log_err, "\033[31m[%s|%s] %s\033[0m\n", (char *)&prefix, time_buf, buf);
        fflush(_log_err);
    }
    inline void warn(const char *__format, ...) {
        if (log_level() < ll_warn || _stop_log) return;

        char buf[4096];
        va_list args;
        va_start(args, __format);
        int nbytes = vsnprintf(buf, sizeof(buf), __format, args);
        va_end(args);

        if (nbytes < 0 || nbytes >= sizeof(buf)) this->warn("logger_t::err: message truncated");

        struct timeval tv;
        gettimeofday(&tv, NULL);
        struct tm tm_info;
        localtime_r(reinterpret_cast<time_t *>(&tv.tv_sec), &tm_info);
        char time_buf[64];

        strftime(time_buf, sizeof(time_buf), "%y-%m-%d %H:%M:%S", &tm_info);
        snprintf(time_buf + strlen(time_buf), sizeof(time_buf) - strlen(time_buf), ".%03ld", tv.tv_usec / 1000);

        uint64_t prefix = this->log_prefix();
        fprintf(_log_err, "\033[33m[%s|%s] %s\033[0m\n", (char *)&prefix, time_buf, buf);
        fflush(_log_err);
    }
    inline void info(const char *__format, ...) {
        if (log_level() < ll_info || _stop_log) return;

        char buf[4096];
        va_list args;
        va_start(args, __format);
        int nbytes = vsnprintf(buf, sizeof(buf), __format, args);
        va_end(args);

        if (nbytes < 0 || nbytes >= sizeof(buf)) this->warn("logger_t::err: message truncated");

        struct timeval tv;
        gettimeofday(&tv, NULL);
        struct tm tm_info;
        localtime_r(reinterpret_cast<time_t *>(&tv.tv_sec), &tm_info);
        char time_buf[64];

        strftime(time_buf, sizeof(time_buf), "%y-%m-%d %H:%M:%S", &tm_info);
        snprintf(time_buf + strlen(time_buf), sizeof(time_buf) - strlen(time_buf), ".%03ld", tv.tv_usec / 1000);

        uint64_t prefix = this->log_prefix();
        fprintf(_log_out, "\033[36m[%s|%s] %s\033[0m\n", (char *)&prefix, time_buf, buf);
        fflush(_log_out);
    }
    inline void dbg(const char *__format, ...) {
        if (log_level() < ll_dbg || _stop_log) return;

        char buf[4096];
        va_list args;
        va_start(args, __format);
        int nbytes = vsnprintf(buf, sizeof(buf), __format, args);
        va_end(args);

        if (nbytes < 0 || nbytes >= sizeof(buf)) this->warn("logger_t::err: message truncated");

        struct timeval tv;
        gettimeofday(&tv, NULL);
        struct tm tm_info;
        localtime_r(reinterpret_cast<time_t *>(&tv.tv_sec), &tm_info);
        char time_buf[64];

        strftime(time_buf, sizeof(time_buf), "%y-%m-%d %H:%M:%S", &tm_info);
        snprintf(time_buf + strlen(time_buf), sizeof(time_buf) - strlen(time_buf), ".%03ld", tv.tv_usec / 1000);

        uint64_t prefix = this->log_prefix();
        fprintf(_log_out, "\033[0m[%s|%s] %s\033[0m\n", (char *)&prefix, time_buf, buf);
        fflush(_log_out);
    }

    inline virtual void init(const char *keyfunc) = 0;
    inline virtual void exit(const char *keyfunc) = 0;
    inline virtual void report(const char *keyfunc) = 0;

    protected:
    static constexpr int _ll = 0;
    static constexpr uint64_t _log_prefix = 0;
    FILE *_log_out = stdout;
    FILE *_log_err = stdout;

    char _curr_keyfunc[32];
    int _stop_log;
    std::thread *log_thread;
};

struct pool_logger_t : public generic_logger_t {
    struct timeval              ev_init_time;
    struct rusage               ev_init_cpu;
    std::atomic<uint64_t>       ev_kernel_us;
    std::atomic<uint64_t>       ev_unpack_us;
    std::atomic<uint64_t>       ev_pack_us;
    std::atomic<uint64_t>       ev_h2d_us;
    std::atomic<uint64_t>       ev_d2h_us;
    std::atomic<uint64_t>       ev_ld_stall_us;
    std::atomic<uint64_t>       ev_h2d_nbytes;
    std::atomic<uint64_t>       ev_d2h_nbytes;
    std::atomic<uint64_t>       ev_vec_nbytes;
    std::atomic<uint32_t>       ev_curr_chunks;
    std::atomic<uint32_t>       ev_total_chunks;

    int num_thread;

    // for ut_checker in sieving
    long exp_batch = 0, max_batch = 0;
    std::atomic<uint64_t>       ev_batch_num;
    std::atomic<uint64_t>       ev_batch_us;
    std::atomic<uint64_t>       ev_batch_rm_ssum;
    std::atomic<uint64_t>       ev_batch_check_ssum;
    std::atomic<uint64_t>       ev_max_table_size;
    std::atomic<uint64_t>       ev_ld_frags;
    std::atomic<uint64_t>       ev_st_frags;
    struct timeval              ev_batch_start;

    inline int log_level() override { return _ll; }
    inline uint64_t log_prefix() override { return _log_prefix; }

    inline void init(const char *keyfunc) override;
    inline void exit(const char *keyfunc) override;
    inline void report(const char *keyfunc) override;

    inline void clear() {
        gettimeofday(&ev_init_time, NULL);
        getrusage(RUSAGE_SELF, &ev_init_cpu);
        ev_kernel_us = ev_unpack_us = ev_pack_us = ev_h2d_us = ev_d2h_us = ev_ld_stall_us = 0;
        ev_h2d_nbytes = ev_d2h_nbytes = ev_vec_nbytes = ev_curr_chunks = ev_total_chunks = 0;
        ev_batch_num = ev_batch_us = ev_batch_rm_ssum = ev_batch_check_ssum = ev_ld_frags = ev_st_frags = ev_max_table_size = 0;
    }

    //private:
    static constexpr int _ll = POOL_HD_LOG_LEVEL;
    static constexpr uint64_t _log_prefix = 'p' + ('o' << 8) + ('o' << 16) + ('l' << 24);
};

struct pwc_logger_t : public generic_logger_t {
    struct timeval              ev_init_time;
    struct rusage               ev_init_cpu;

    inline int log_level() override { return _ll; }
    inline uint64_t log_prefix() override { return _log_prefix; }

    inline void clear() {
        gettimeofday(&ev_init_time, NULL);
        getrusage(RUSAGE_THREAD, &ev_init_cpu);
    }

    inline void init(const char *keyfunc) override {}
    inline void exit(const char *keyfunc) override {}
    inline void report(const char *keyfunc) override {}

    inline virtual int construcion_done() { return _construcion_done; }
    int _construcion_done = 0;

    //private:
    static constexpr int _ll = PWC_LOG_LEVEL;
    static constexpr uint64_t _log_prefix = 'p' + ('w' << 8) + ('c' << 16);
};

#else
#define cond_lg_init(_cond)
#define cond_lg_exit(_cond)
#define lg_init()   
#define lg_exit()    
#define lg_report()
#define lg_err(...)  
#define lg_warn(...) 
#define lg_info(...) 
#define lg_dbg(...)  
#endif

struct Pool_hd_t {
    public:
    // static configurations
    static constexpr long vec_nbytes = 176;
    static constexpr long chunk_max_nvecs = 8192;

    // construction and distructions
    Pool_hd_t();
    Pool_hd_t(Lattice_QP *L);
    ~Pool_hd_t();

    // setup
    int set_basis(Lattice_QP *L);
    int set_num_threads(long num_threads);
    int set_sieving_context(long ind_l, long ind_r);
    int set_boost_depth(long esd);

    // basis
    Lattice_QP *basis;
    uint64_t basis_hash;

    // the hash table
    UidTable *uid_table;

    // access pool with cache via this manager
    pwc_manager_t *pwc_manager;

    // statistics for pool quality
    uint32_t score_stat[65536] = {};

    // sieving status
    long CSD;           // current sieving dimension
    long ESD;           // usually set to 0
    long index_l;       // current sieving context = [index_l, index_r]
    long index_r;       // so CSD is equal to index_r - index_l
    inline double gh2() { return _gh2; }         // gh^2 of L_{[index_l, index_r]}
    inline double gh2_scaled() { return _gh2 * _ratio * _ratio; } 

    // pool operations, in the same time only one operation is allowed

    /// @brief do gaussian sampling and size reduction to collect N vectors in the pool
    /// @note host-only
    int sampling(long N);
    /// @brief shrink the pool size to N
    /// @note host-only
    int shrink(long N);
    /// @brief extend_left
    int extend_left();
    /// @brief shrink left
    int shrink_left();
    /// @brief do insertion
    int insert(long index, double eta, long *pos = NULL, long auto_lll = 1);
    /// @brief show the minimal lift to index
    int show_min_lift(long index);
    /// @brief sync current pool to disk
    int store();
    /// @brief try to recover the pool from disk
    int load(long log_level = 0);
    /// @brief check for dimension lose
    /// @return 0 if passed, -1 otherwise
    int check_dim_lose();
    /// @brief check all, report only
    /// @return 0 if passed, -1 otherwise
    int check(int log_level);

    // sieving
    int bgj1_Sieve_hd();
    int bgj2_Sieve_hd();
    int bgj3_Sieve_hd();
    int bgj3l_Sieve_hd();
    int bgj4_Sieve_hd();

    // hash based insertion
    int dh_insert(long target_index, double eta, double max_time = 0.0, long *pos = NULL, double target_length = 0.0);
    int dh_final(long target_index, double eta, double max_time = 0.0, double target_length = 0.0);

    #if ENABLE_PROFILING
    typedef pool_logger_t logger_t;
    logger_t *logger = NULL;
    #endif

    int down_sieve_flag = 0;
    
    private:
    long _num_threads;

    // local basis information
    double _gh2;
    int32_t _dhalf;
    int32_t _dshift;
    int8_t *_b_dual = NULL;
    float _ratio;
    float **_b_local = NULL;

    boost_data_t *_boost_data = NULL;

    template <uint32_t Ver> friend struct pdev_traits_t;
    template <class traits> 
    int stream_task_template(int num_devices, cudaDeviceProp device_props[], local_data_t **local_data);
    template <class traits>
    int stream_stat_template(int num_devices, cudaDeviceProp device_props[], local_data_t **local_data);
    
    int _bgj_Sieve_hd(int bgj);

    // update b_local according to current sieving context
    int _update_b_local(float ratio = 0.0f);
    // compute b_local for sieving context with the same ratio
    float **_compute_b_local(long ind_l, long ind_r);
    // update boost data according to current sieving context
    int _update_boost_data();
    
    friend struct sampling_iterator_t;
    int _sampling(int8_t *dst_vec, uint16_t *dst_score, int32_t *dst_norm, uint64_t *dst_u, DGS1d *R);

    friend struct dhb_buffer_t;
    friend struct dhr_buffer_t;
    friend struct dh_bucketer_t;
};


struct chunk_t {
    int32_t id;
    int16_t size;
    uint16_t *score;
    int32_t *norm;
    uint64_t *u;
    int8_t *vec;
};

struct boost_data_t {
    static constexpr int max_boost_dim = 48;

    float evec[(Pool_hd_t::vec_nbytes + max_boost_dim) * max_boost_dim];
    float igh[max_boost_dim];
    float inorm[max_boost_dim];
};

struct ut_checker_t {
    static constexpr long   default_num_threads = UT_DEFAULT_NUM_THREADS;
    static constexpr long   default_max_chunks  = UT_DEFAULT_MAX_CHUNKS;
    static constexpr long   default_max_uids    = UT_DEFAULT_MAX_UIDS;
    static constexpr double default_batch_ratio = UT_DEFAULT_BATCH_RATIO;
    static constexpr long   table_dram_slimit   = UT_TABLE_DRAM_SLIMIT;

    static constexpr long type_sieve  = 0;
    static constexpr long type_check  = 1;
    static constexpr long type_shrink = 2;
    static constexpr long type_others = 3;

    ut_checker_t(long type, Pool_hd_t *p, UidTable *uid_table, pwc_manager_t *pwc_manager);
    ~ut_checker_t();

    int set_num_threads(long num_threads);
    int set_max_holding(int max_holding);
    int set_exp_batch(long exp_batch);
    inline void set_max_available_id(int *max_available_id) { _max_available_id = max_available_id; }
    inline void set_score_stat(uint32_t *score_stat, pthread_spinlock_t slock) { 
        this->score_stat = score_stat; _score_stat_lock = slock; 
    }
    inline void set_stuck_stat(uint64_t *total_check, uint64_t *total_notin, pthread_spinlock_t stuck_lock) {
        this->total_check = total_check; 
        this->total_notin = total_notin;
        _stuck_lock = stuck_lock;
    }

    int task_commit(chunk_t *chunk);
    int task_commit(uint64_t uid);      /// aborted
    int task_commit(uint64_t *uids, long num_uids);
    int trigger_batch();
    int input_done();
    int batch(long tid, long table_size, long current_hold);
    template <int type, int cross>
    long real_work(long tid, long start, long end);
    long wait_work();

    #if ENABLE_PROFILING
    pool_logger_t *logger;
    #endif

    long _type;
    long _num_threads;
    pthread_spinlock_t _ut_lock;
    thread_pool::thread_pool _ut_pool;
    uint32_t *score_stat;
    pthread_spinlock_t _score_stat_lock;

    /// for sieving stuck detection
    uint64_t *total_check;
    uint64_t *total_notin;
    pthread_spinlock_t _stuck_lock;

    /// for signal handling
    std::mutex _ut_mutex;
    std::condition_variable _ut_cv;
    std::atomic<int> _running_threads{0};
    std::atomic<int> _need_batch{0};
    std::atomic<int> _input_done{0};
    pthread_barrier_t _ut_barrier;

    UidTable *_uid_table;
    pwc_manager_t *_pwc_manager;
    swc_manager_t *_swc_manager;
    
    chunk_t *_to_check = NULL;
    chunk_t *_in_check = NULL;
    long _num_to_check = 0;
    long _num_in_check = 0;

    chunk_t *_red_to_rm = NULL;
    chunk_t *_red_in_rm = NULL;
    long _num_red_to_rm = 0;
    long _num_red_in_rm = 0;

    long _exp_batch;
    long _max_holding;
    int CSD;

    std::atomic<long> _check_fail{0};

    int *_max_available_id = NULL;
};

template <class logger_t>
struct pwc_manager_tmpl {
    public:
    typedef uint32_t chunk_status_t;
    static constexpr chunk_status_t _ck_loading = 0x80000000;         // loading from disk
    static constexpr chunk_status_t _ck_syncing = 0x40000000;         // syncing to disk
    static constexpr chunk_status_t _ck_caching = 0x20000000;         // cached in memory
    static constexpr chunk_status_t _ck_reading = 0x10000000;         // cache is being used (read-only)
    static constexpr chunk_status_t _ck_writing = 0x08000000;         // cache is being used (read & write)
    static constexpr chunk_status_t _ck_to_sync = 0x04000000;         // need to sync to disk, but not started
    static constexpr chunk_status_t _ck_size_mask = 2 * Pool_hd_t::chunk_max_nvecs - 1;
    static constexpr chunk_status_t _ck_cache_id_mask = 0x00ffffff;            


    static constexpr int32_t pwc_locks = 256;
    static constexpr long pwc_default_loading_threads = PWC_DEFAULT_LOADING_THREADS;
    static constexpr long pwc_default_syncing_threads = PWC_DEFAULT_SYNCING_THREADS;
    static constexpr long pwc_default_max_cached_chunks = PWC_DEFAULT_MAX_CACHED_CHUNKS;
    static constexpr long pwc_max_parallel_sync_chunks = PWC_MAX_PARALLEL_SYNC_CHUNKS;

    pwc_manager_tmpl();
    pwc_manager_tmpl(long loading_threads, long syncing_threads, long max_cached_chunks);
    ~pwc_manager_tmpl();

    long num_vec() const;
    inline long num_chunks() const;     // return _num_chunks
    inline long num_empty() const;      // return __num_deleted_ids
    inline long max_cached_chunks() const;      // return _max_cached_chunks
    inline long chunk_size(long chunk_id) const;
    inline chunk_status_t chunk_status(long chunk_id) const;
    #if MULTI_SSD
    inline const char *pfx() const;
    inline const char *dir() const;
    #else
    inline const char* prefix() const;
    #endif

    int set_pool(Pool_hd_t *pool);
    int set_dirname(const char *dirname);
    int set_num_threads(long loading_threads, long syncing_threads);
    int set_max_cached_chunks(long max_cached_chunks);

    /// @brief create a empty chunk, return the chunk_id
    long create_chunk();
    /// @brief try to prefetch a chunk
    int prefetch(long chunk_id);
    /// @brief will stall until the chunk is ready
    chunk_t *fetch(long chunk_id);
    /// @brief unlock the chunk only
    int release(long chunk_id);
    /// @brief first add the chunk to the list of chunks to sync, then unlock the chunk
    int release_sync(long chunk_id);
    /// @brief sync an in-writing chunk to disk and release it
    int sync_release(long chunk_id);

    /// @brief wait until all tasks are done
    int wait_work();

    #if ENABLE_PROFILING
    logger_t *logger;
    std::atomic<uint64_t> ev_pwc_fetch{0};
    std::atomic<uint64_t> ev_pwc_cache_hit{0};
    std::atomic<uint64_t> ev_f4w{0};
    std::atomic<uint64_t> ev_f4r{0};
    std::atomic<uint64_t> ev_f4w_hit{0};
    std::atomic<uint64_t> ev_f4r_hit{0};
    std::atomic<uint64_t> ev_ssd_ld{0};
    std::atomic<uint64_t> ev_ssd_st{0};
    #endif

    protected:
    #if MULTI_SSD
    char _dir[32] = {};
    char _pfx[32] = {};
    #else
    char _prefix[32] = {};
    #endif
    Pool_hd_t *_pool;
    int _set_prefix(const char *dirname, const char *basis_prefix);

    long _loading_threads;
    long _syncing_threads;
    thread_pool::thread_pool _loading_pool;
    thread_pool::thread_pool _syncing_pool;
    
    // chunk status
    long _num_chunks;
    chunk_status_t *_chunk_status;
    pthread_spinlock_t _locks[pwc_locks];

    // for chunk deletion
    int *_deleted_ids;
    int _num_deleted_ids;
    pthread_spinlock_t _deleted_ids_lock;

    /// @brief release the chunk and delete it from the pool
    int release_del(long chunk_id);

    // cache
    long _max_cached_chunks;
    chunk_t *_cached_chunks;
    int32_t _last_cache;
    pthread_spinlock_t _cached_chunks_lock;

    std::atomic<int32_t> _num_loading_chunks;
    std::atomic<int32_t> _num_syncing_chunks;
    std::queue<int32_t> _to_sync_chunks;
    pthread_spinlock_t _to_sync_chunks_lock;

    void __load_chunk(long chunk_id);
    void __sync_chunk(long chunk_id);
    void __signal_sync_done();
    int32_t __fetch_cache_for(long chunk_id);
    void __free_all();

    friend int ut_checker_t::batch(long tid, long table_size, long current_hold);

    private:
    friend int Pool_hd_t::shrink(long N);
    friend struct shrink_iterator_t;
};

template <class logger_t> 
inline long pwc_manager_tmpl<logger_t>::num_chunks() const {
    return _num_chunks;
}

template <class logger_t> 
inline long pwc_manager_tmpl<logger_t>::num_empty() const {
    return _num_deleted_ids;
}

template <class logger_t>
inline long pwc_manager_tmpl<logger_t>::max_cached_chunks() const {
    return _max_cached_chunks;
}

template <class logger_t> 
inline long pwc_manager_tmpl<logger_t>::chunk_size(long chunk_id) const {
    chunk_status_t s = _chunk_status[chunk_id];
    if (s & _ck_caching) {
        return _cached_chunks[s & _ck_cache_id_mask].size;
    }

    return s & _ck_size_mask;
}

template <class logger_t> 
inline typename pwc_manager_tmpl<logger_t>::chunk_status_t pwc_manager_tmpl<logger_t>::chunk_status(long chunk_id) const {
    return _chunk_status[chunk_id];
}

#if MULTI_SSD
template <class logger_t>
inline const char *pwc_manager_tmpl<logger_t>::pfx() const {
    return _pfx;
}

template <class logger_t>
inline const char *pwc_manager_tmpl<logger_t>::dir() const {
    return _dir;
}
#else
template <class logger_t> 
inline const char *pwc_manager_tmpl<logger_t>::prefix() const {
    return _prefix;
}
#endif

#if ENABLE_PROFILING

inline void pool_logger_t::init(const char *keyfunc) {
    this->info("\033[0menter keyfunc %s", keyfunc);
    this->clear();
    int keyfunclen = snprintf(_curr_keyfunc, sizeof(_curr_keyfunc), "%s", keyfunc);
    if (keyfunclen < 0 || keyfunclen >= sizeof(_curr_keyfunc)) this->warn("logger_t::init: keyfunc name truncated");

    if (str_equal(keyfunc, "sampling")) {

    } else if (str_equal(keyfunc, "shrink")) {

    } else if (str_equal(keyfunc, "extend_left")) {

    } else if (str_equal(keyfunc, "shrink_left")) {

    } else if (str_equal(keyfunc, "insert")) {

    } else if (str_equal(keyfunc, "show_min_lift")) {

    } else if (str_equal(keyfunc, "store")) {

    } else if (str_equal(keyfunc, "load")) {

    } else if (str_equal(keyfunc, "check_dim_lose")) {

    } else if (str_equal(keyfunc, "check")) {

    } else if (str_equal(keyfunc, "bgj1_Sieve_hd")) {

    } else if (str_equal(keyfunc, "bgj2_Sieve_hd")) {

    } else if (str_equal(keyfunc, "bgj3_Sieve_hd")) {

    } else if (str_equal(keyfunc, "bgj3l_Sieve_hd")) {
        
    } else if (str_equal(keyfunc, "bgj4_Sieve_hd")) {

    } else if (str_equal(keyfunc, "dh_final")) {

    } else if (str_equal(keyfunc, "dh_insert")) {

    } else this->warn("logger_t::init: unknown input \'%s\'", keyfunc);
}

inline void pool_logger_t::exit(const char *keyfunc) {
    _curr_keyfunc[0] = '\0';

    if (str_equal(keyfunc, "sampling")) {

    } else if (str_equal(keyfunc, "shrink")) {

    } else if (str_equal(keyfunc, "extend_left")) {

    } else if (str_equal(keyfunc, "shrink_left")) {

    } else if (str_equal(keyfunc, "insert")) {

    } else if (str_equal(keyfunc, "show_min_lift")) {

    } else if (str_equal(keyfunc, "store")) {

    } else if (str_equal(keyfunc, "load")) {

    } else if (str_equal(keyfunc, "check_dim_lose")) {

    } else if (str_equal(keyfunc, "check")) {

    } else if (str_equal(keyfunc, "bgj1_Sieve_hd")) {

    } else if (str_equal(keyfunc, "bgj2_Sieve_hd")) {

    } else if (str_equal(keyfunc, "bgj3_Sieve_hd")) {

    } else if (str_equal(keyfunc, "bgj3l_Sieve_hd")) {

    } else if (str_equal(keyfunc, "bgj4_Sieve_hd")) {

    } else if (str_equal(keyfunc, "dh_final")) {

    } else if (str_equal(keyfunc, "dh_insert")) {

    } else this->warn("logger_t::exit: unknown input \'%s\'", keyfunc);
    
    this->info("\033[0mexit keyfunc %s", keyfunc);
}

inline void pool_logger_t::report(const char *keyfunc) {
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
    double bw = ev_vec_nbytes.load() / elapsed * 0x1p-30;
    double ld_bw = (ev_h2d_nbytes.load() + ev_curr_chunks.load() * 
                            Pool_hd_t::chunk_max_nvecs * 14 + 12) / elapsed * 0x1p-30;
    double h2d = ev_h2d_us * 1e-6;
    double h2d_bw = ev_h2d_nbytes / h2d * 0x1p-30;
    double d2h = ev_d2h_us * 1e-6;
    double d2h_bw = ev_d2h_nbytes / d2h * 0x1p-30;
    double kernel = ev_kernel_us * 1e-6;
    double kernel_bw = ev_vec_nbytes / kernel * 0x1p-30;
    double unpack = ev_unpack_us * 1e-6;
    double unpack_bw = ev_vec_nbytes / unpack * 0x1p-30;
    double pack = ev_pack_us * 1e-6;
    double pack_bw = ev_vec_nbytes / pack * 0x1p-30;
    double dev_bw = ev_vec_nbytes / (kernel + unpack + pack) * 0x1p-30;

    if (str_equal(keyfunc, "sampling")) {
        
    } else if (str_equal(keyfunc, "shrink")) {
        this->info("%d / %d chunks left, elapsed: %.3fs, cpu: %.3fs(avg: %.2f), %d thread(s) working, bw: %.2f GB/s",
                    ev_curr_chunks.load(), ev_total_chunks.load(), elapsed, cpu, avg_load, num_thread, bw);
    } else if (str_equal(keyfunc, "store")) {

    } else if (str_equal(keyfunc, "extend_left") || str_equal(keyfunc, "shrink_left") || 
               str_equal(keyfunc, "load") || str_equal(keyfunc, "check") || str_equal(keyfunc, "insert") ||
               str_equal(keyfunc, "check_dim_lose") || str_equal(keyfunc, "show_min_lift")) {
        this->info("%s: %u / %u chunks done, elapsed: %.3fs, cpu: %.3fs(avg: %.2f), %d thread(s) working, bw: %.2f GB/s",
                    keyfunc, ev_curr_chunks.load(), ev_total_chunks.load(), elapsed, cpu, avg_load, num_thread, ld_bw);
        this->info("load stall: %.3fs, H2D: %.3fs(bw: %.2f GB/s), D2H: %.3fs(bw: %.2f GB/s)", 
                    ld_stall, h2d, h2d_bw, d2h, d2h_bw);
        this->info("kernel: %.3fs(bw: %.2f GB/s), unpack %.3fs(bw: %.2f GB/s), pack %.3fs(bw: %.2f GB/s), device bw: %.2f GB/s", 
                    kernel, kernel_bw, unpack, unpack_bw, pack, pack_bw, dev_bw);
    } else if (str_equal(keyfunc, "bgj1_Sieve_hd") || str_equal(keyfunc, "bgj2_Sieve_hd") || 
               str_equal(keyfunc, "bgj3_Sieve_hd") || str_equal(keyfunc, "bgj3l_Sieve_hd") || str_equal(keyfunc, "bgj4_Sieve_hd")) {
        this->info("max|UT| = 2^%.2f, #batch %d, (avg: %.2fs, #rm %.2f, #check %.2f, E %ld/%ld), ld %.3fK(%.3f frag/s), st %.3fK(%.3f frag/s)",
                    log2(ev_max_table_size.load()), ev_batch_num.load(), ev_batch_us.load() * 1e-6 / (float) ev_batch_num.load(), 
                    ev_batch_rm_ssum.load() / (float) ev_batch_num.load(),
                    ev_batch_check_ssum.load() / (float) ev_batch_num.load(),
                    exp_batch, max_batch,
                    ev_ld_frags.load() * 1e-3, ev_ld_frags.load() * 1e-3 / elapsed,
                    ev_st_frags.load() * 1e-3, ev_st_frags.load() * 1e-3 / elapsed);
    } else if (str_equal(keyfunc, "dh_insert") || str_equal(keyfunc, "dh_final")) {
        
    } else this->warn("logger_t::report: unknown keyfunc \'%s\'", keyfunc);
}


#endif

#include <sys/stat.h>
#include <sys/types.h>

#include <unistd.h>
#include <fcntl.h>

void _free_chunk(chunk_t *chunk);
void _malloc_chunk(chunk_t *chunk);
int _normalize_chunk(chunk_t *chunk, int CSD);

template <class logger_t> int pwc_manager_tmpl<logger_t>::set_num_threads(long loading_threads, long syncing_threads) {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    this->_loading_threads = loading_threads;
    this->_syncing_threads = syncing_threads;
    this->_loading_pool.resize(loading_threads);
    this->_syncing_pool.resize(syncing_threads);

    gettimeofday(&end, NULL);
    #if ENABLE_PROFILING
    if (logger->construcion_done()) lg_info("num_threads set to %ld(L) %ld(S), %.3fs", loading_threads, syncing_threads,
            (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6);
    #endif

    return 0;
}

template <class logger_t> int pwc_manager_tmpl<logger_t>::set_max_cached_chunks(long max_cached_chunks) {
    #if ENABLE_PROFILING
    cond_lg_init(logger->construcion_done());
    #endif
    if (max_cached_chunks == this->_max_cached_chunks) {
        cond_lg_exit(logger->construcion_done());
        return 0;
    }

    constexpr chunk_status_t _ck_busy = _ck_loading | _ck_syncing | 
                                          _ck_reading | _ck_writing | _ck_to_sync;
    volatile chunk_status_t *_chunk_status_vol = reinterpret_cast<volatile chunk_status_t *>(_chunk_status);
    volatile chunk_t *_cached_chunks_vol = reinterpret_cast<volatile chunk_t *>(_cached_chunks);

    // need ptr
    #define TRY_LOCK_THEN_FREE_CHUNK(_ck_id, _ck_id_vol)            \
        int32_t _id = _ck_id;                                       \
        if (_id < 0) {                                              \
            pthread_spin_lock(&_cached_chunks_lock);                \
            if (_ck_id_vol == -1) {                                 \
                _ck_id = -2;                                        \
                pthread_spin_unlock(&_cached_chunks_lock);          \
                if (_cached_chunks[ptr].score) {                    \
                    _free_chunk(&_cached_chunks[ptr]);              \
                }                                                   \
                continue;                                           \
            }                                                       \
            pthread_spin_unlock(&_cached_chunks_lock);              \
        }                                                           \
        if ((_chunk_status[_id] & _ck_busy) == 0) {                 \
            pthread_spin_lock(&_locks[_id % pwc_locks]);            \
            if ((_chunk_status_vol[_id] & _ck_busy) == 0) {         \
                _chunk_status[_id] |= _ck_busy;                     \
                pthread_spin_unlock(&_locks[_id % pwc_locks]);      \
                pthread_spin_lock(&_cached_chunks_lock);            \
                if (_id == _ck_id_vol) {                            \
                    _ck_id = -2;                                    \
                    pthread_spin_unlock(&_cached_chunks_lock);      \
                    chunk_status_t _tmp = _chunk_status[_id];       \
                    _tmp &= ~_ck_cache_id_mask;                     \
                    _tmp |= _cached_chunks[ptr].size;               \
                    _tmp &= (~_ck_busy) & (~_ck_caching);           \
                    _chunk_status[_id] = _tmp;                      \
                    if (_cached_chunks[ptr].score) {                \
                        _free_chunk(&_cached_chunks[ptr]);          \
                    }                                               \
                    continue;                                       \
                } else {                                            \
                    _chunk_status[_id] &= ~_ck_busy;                \
                    pthread_spin_unlock(&_cached_chunks_lock);      \
                }                                                   \
            } else pthread_spin_unlock(&_locks[_id % pwc_locks]);   \
        }

    if (max_cached_chunks < this->_max_cached_chunks) {
        long old_max_cached_chunks = this->_max_cached_chunks;
        this->_max_cached_chunks = max_cached_chunks;
        this->_last_cache = _last_cache % max_cached_chunks;
        
        int32_t *cache_to_evict = (int32_t *) malloc((old_max_cached_chunks - max_cached_chunks) * sizeof(int32_t));
        long num_cache_to_evict = 0;

        for (int32_t ptr = max_cached_chunks; ptr < old_max_cached_chunks; ptr++) {
            TRY_LOCK_THEN_FREE_CHUNK(_cached_chunks[ptr].id, _cached_chunks_vol[ptr].id);
            cache_to_evict[num_cache_to_evict] = ptr;
            num_cache_to_evict++;
        }

        struct timeval start, end;
        gettimeofday(&start, NULL);
        while (num_cache_to_evict) {
            int still_using = 0;
            for (long j = 0; j < num_cache_to_evict; j++) {
                int32_t ptr = cache_to_evict[j];
                TRY_LOCK_THEN_FREE_CHUNK(_cached_chunks[ptr].id, _cached_chunks_vol[ptr].id);
                cache_to_evict[still_using] = ptr;
                still_using++;
            }
            num_cache_to_evict = still_using;

            gettimeofday(&end, NULL);
            if ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6 > 10.0) {
                // recover 
                long k = 0;
                for (int32_t ptr = max_cached_chunks; ptr < old_max_cached_chunks; ptr++) {
                    if (ptr == cache_to_evict[k]) {
                        k++;
                        continue;
                    } else {
                        _cached_chunks[ptr].size = 0;
                        _cached_chunks[ptr].score = NULL;
                        _cached_chunks[ptr].norm = NULL;
                        _cached_chunks[ptr].u = NULL;
                        _cached_chunks[ptr].vec = NULL;
                        _cached_chunks[ptr].id = -1;
                    }
                }
                this->_max_cached_chunks = old_max_cached_chunks;
                lg_err("some chunks are still in use after 10 seconds, aborted");
                if (k != num_cache_to_evict) lg_err("assertion k == num_cache_to_evict failed");
                free(cache_to_evict);

                return -1;
            }
        }
        free(cache_to_evict);
    } else {
        for (int32_t ptr = this->_max_cached_chunks; ptr < max_cached_chunks; ptr++) {
            _cached_chunks[ptr].id = -1;
            _cached_chunks[ptr].size = 0;
            _cached_chunks[ptr].score = NULL;
            _cached_chunks[ptr].norm = NULL;
            _cached_chunks[ptr].u = NULL;
            _cached_chunks[ptr].vec = NULL;
        }
        this->_max_cached_chunks = max_cached_chunks;
    }
    #if ENABLE_PROFILING
    if (logger->construcion_done()) lg_info("max_cached_chunks set to %ld", max_cached_chunks);
    cond_lg_exit(logger->construcion_done());
    #endif

    return 0;
}

template <class logger_t> int32_t pwc_manager_tmpl<logger_t>::__fetch_cache_for(long chunk_id) {
    lg_init();
    constexpr chunk_status_t _ck_busy = _ck_loading | _ck_syncing | 
                                        _ck_reading | _ck_writing | _ck_to_sync;
    volatile chunk_status_t *_chunk_status_vol = reinterpret_cast<volatile chunk_status_t*>(_chunk_status);
    volatile chunk_t *_cached_chunks_vol = reinterpret_cast<volatile chunk_t*>(_cached_chunks);

    for (int32_t cache_id = _last_cache + 1;; cache_id++) {
        if (cache_id >= _max_cached_chunks) cache_id %= _max_cached_chunks;

        int32_t old_chunk_id = _cached_chunks[cache_id].id;
        if (old_chunk_id >= 0) {
            if (_chunk_status[old_chunk_id] & _ck_busy) continue;
        }
        
        if (old_chunk_id == -1) {
            pthread_spin_lock(&_cached_chunks_lock);
            if (_cached_chunks_vol[cache_id].id == -1) {
                _cached_chunks[cache_id].id = chunk_id;
                _last_cache = cache_id;
                pthread_spin_unlock(&_cached_chunks_lock);
                
                _cached_chunks[cache_id].size = _chunk_status[chunk_id] & _ck_size_mask;
                if (_cached_chunks[cache_id].vec) { lg_exit(); return cache_id; }
                _malloc_chunk(&_cached_chunks[cache_id]);
                lg_exit();
                return cache_id;
            }
            pthread_spin_unlock(&_cached_chunks_lock);
        }

        if (old_chunk_id >= 0) {
            pthread_spin_lock(&_locks[old_chunk_id % pwc_locks]);
            if (_chunk_status_vol[old_chunk_id] & _ck_busy) {
                pthread_spin_unlock(&_locks[old_chunk_id % pwc_locks]);
                continue;
            }
            _chunk_status[old_chunk_id] |= _ck_writing | _ck_reading;
            pthread_spin_unlock(&_locks[old_chunk_id % pwc_locks]);
            
            pthread_spin_lock(&_cached_chunks_lock);
            if (old_chunk_id != _cached_chunks_vol[cache_id].id) {
                pthread_spin_unlock(&_cached_chunks_lock);
                pthread_spin_lock(&_locks[old_chunk_id % pwc_locks]);
                _chunk_status[old_chunk_id] &= ~(_ck_writing | _ck_reading);
                pthread_spin_unlock(&_locks[old_chunk_id % pwc_locks]);
                continue;
            }

            _cached_chunks[cache_id].id = chunk_id;
            _last_cache = cache_id;
            pthread_spin_unlock(&_cached_chunks_lock);

            chunk_status_t new_status = _chunk_status[old_chunk_id];
            new_status &= ~_ck_cache_id_mask;
            new_status |= _cached_chunks[cache_id].size;
            new_status &= (~_ck_writing) & (~_ck_caching) & (~_ck_reading);
            pthread_spin_lock(&_locks[old_chunk_id % pwc_locks]);
            _chunk_status[old_chunk_id] = new_status;
            pthread_spin_unlock(&_locks[old_chunk_id % pwc_locks]);

            _cached_chunks[cache_id].size = _chunk_status[chunk_id] & _ck_size_mask;
            lg_exit();
            return cache_id;
        }
    }
}

template <class logger_t> void pwc_manager_tmpl<logger_t>::__load_chunk(long chunk_id) {
    lg_init();
    int32_t cache_id = __fetch_cache_for(chunk_id);
    chunk_t *dst_chunk = &_cached_chunks[cache_id];
    if ((_chunk_status[chunk_id] & _ck_size_mask) == _ck_size_mask) std::abort();
    
    char chunk_filename[256];
    #if MULTI_SSD
    snprintf(chunk_filename, sizeof(chunk_filename), "%s/%s/%s%06lx", _dir, hw::ssd_name(chunk_id), _pfx, chunk_id);
    #else
    snprintf(chunk_filename, sizeof(chunk_filename), "%s%06lx", _prefix, chunk_id);
    #endif

    #if ONE_TIME_IO
    char *meta_data = ((char *)dst_chunk->score) - 12;
    #else
    char meta_data[12];
    #endif

    if ((_chunk_status[chunk_id] & _ck_size_mask) == 0) {
        dst_chunk->size = 0;
        memset(dst_chunk->score, 0, Pool_hd_t::chunk_max_nvecs * sizeof(uint16_t));
        memset(dst_chunk->norm, 0, Pool_hd_t::chunk_max_nvecs * sizeof(int32_t));
    } else {
        int fd = open(chunk_filename, O_RDONLY | (ONE_TIME_IO ? O_DIRECT : 0));

        if (fd == -1) {
            if (errno != ENOENT) lg_err("open %s failed, %s, treat as empty chunk", chunk_filename, strerror(errno));
            dst_chunk->size = 0;
            memset(dst_chunk->norm, 0, Pool_hd_t::chunk_max_nvecs * sizeof(int32_t));
            memset(dst_chunk->score, 0, Pool_hd_t::chunk_max_nvecs * sizeof(uint16_t));
        } else {
            int read_bytes = 0;
            #if ONE_TIME_IO
            read_bytes += read(fd, meta_data, 4096 + Pool_hd_t::chunk_max_nvecs * (2 + 4 + 8 + _pool->CSD));
            #else
            read_bytes += read(fd, meta_data, 12);
            read_bytes += read(fd, dst_chunk->score, sizeof(uint16_t) * Pool_hd_t::chunk_max_nvecs);
            read_bytes += read(fd, dst_chunk->norm, sizeof(int32_t) * Pool_hd_t::chunk_max_nvecs);
            read_bytes += read(fd, dst_chunk->u, sizeof(uint64_t) * Pool_hd_t::chunk_max_nvecs);
            #endif
            if (read_bytes < sizeof(uint16_t) * Pool_hd_t::chunk_max_nvecs + 
                            sizeof(int32_t) * Pool_hd_t::chunk_max_nvecs + 
                            sizeof(uint64_t) * Pool_hd_t::chunk_max_nvecs + 12) {
                lg_err("chunk %lx corrupted, treat as empty chunk", chunk_id);
                dst_chunk->size = 0;
                memset(dst_chunk->norm, 0, Pool_hd_t::chunk_max_nvecs * sizeof(int32_t));
                memset(dst_chunk->score, 0, Pool_hd_t::chunk_max_nvecs * sizeof(uint16_t));
            } else {
                uint16_t read_size = *((uint16_t *)(&meta_data[0]));
                uint64_t read_hash = *((uint64_t *)(&meta_data[2]));
                uint8_t read_l = *((uint8_t *)(&meta_data[10]));
                uint8_t read_r = *((uint8_t *)(&meta_data[11]));

                if (read_size != dst_chunk->size) {
                    lg_err("chunk %lx size validation failed, cache %d, disk %d, ignored", 
                            chunk_id, dst_chunk->size, read_size);
                }
                if (read_hash != _pool->basis_hash) {
                    lg_err("chunk %lx basis hash validation failed, treat as empty chunk", chunk_id);
                    memset(dst_chunk->norm, 0, Pool_hd_t::chunk_max_nvecs * sizeof(int32_t));
                    memset(dst_chunk->score, 0, Pool_hd_t::chunk_max_nvecs * sizeof(uint16_t));
                }
                if (abs(read_l - _pool->index_l) > 1 || abs(read_r - _pool->index_r) > 1) {
                    lg_err("chunk %lx sieving context validation failed, cache [%ld, %ld], "
                        "disk [%d, %d], treat as empty chunk", chunk_id, _pool->index_l, 
                        _pool->index_r, read_l, read_r);
                    memset(dst_chunk->norm, 0, Pool_hd_t::chunk_max_nvecs * sizeof(int32_t));
                    memset(dst_chunk->score, 0, Pool_hd_t::chunk_max_nvecs * sizeof(uint16_t));
                }
                #if ONE_TIME_IO
                int read_vecs = (read_bytes - 12 - Pool_hd_t::chunk_max_nvecs * (2 + 4 + 8)) / _pool->CSD;
                if (read_vecs > Pool_hd_t::chunk_max_nvecs) read_vecs = Pool_hd_t::chunk_max_nvecs;
                #else
                read_bytes = read(fd, dst_chunk->vec, _pool->CSD * Pool_hd_t::chunk_max_nvecs);
                int read_vecs = read_bytes / _pool->CSD;
                #endif
                memset(dst_chunk->norm + read_vecs, 0, (Pool_hd_t::chunk_max_nvecs - read_vecs) * sizeof(int32_t));
                memset(dst_chunk->score + read_vecs, 0, (Pool_hd_t::chunk_max_nvecs - read_vecs) * sizeof(uint16_t));
            }

            close(fd);
        }
    }

    chunk_status_t new_status = _chunk_status[chunk_id];
    new_status |= _ck_caching;
    new_status &= ~_ck_loading;
    new_status &= ~_ck_cache_id_mask;
    new_status |= cache_id;
    pthread_spin_lock(&_locks[chunk_id % pwc_locks]);
    _chunk_status[chunk_id] = new_status;
    pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);

    #if ENABLE_PROFILING
    ev_ssd_ld.fetch_add(1);
    #endif

    _num_loading_chunks--;
    lg_exit();
}

template <class logger_t> void pwc_manager_tmpl<logger_t>::__sync_chunk(long chunk_id) {
    lg_init();
    chunk_t *src_chunk = &_cached_chunks[_chunk_status[chunk_id] & _ck_cache_id_mask];

    char chunk_filename[256];
    #if MULTI_SSD
    snprintf(chunk_filename, sizeof(chunk_filename), "%s/%s/%s%06lx", _dir, hw::ssd_name(chunk_id), _pfx, chunk_id);
    #else
    snprintf(chunk_filename, sizeof(chunk_filename), "%s%06lx", _prefix, chunk_id);
    #endif

    #if ONE_TIME_IO
    char *meta_data = ((char *)src_chunk->score) - 12;
    #else
    char meta_data[12];
    #endif
    *((uint16_t *)(&meta_data[0])) = (uint16_t) src_chunk->size;
    *((uint64_t *)(&meta_data[2])) = _pool->basis_hash;
    *((uint8_t *)(&meta_data[10])) = (uint8_t) (_pool->index_l);
    *((uint8_t *)(&meta_data[11])) = (uint8_t) (_pool->index_r);

    int fd = open(chunk_filename, O_WRONLY | O_CREAT | (ONE_TIME_IO ? O_DIRECT : 0), 0644);
    if (fd == -1) {
        lg_err("open %s failed, %s, nothing done", chunk_filename, strerror(errno));
    } else {
        int write_bytes = 0;
        #if ONE_TIME_IO
        write_bytes += write(fd, meta_data, 4096 + Pool_hd_t::chunk_max_nvecs * (2 + 4 + 8 + _pool->CSD));
        #else
        write_bytes += write(fd, meta_data, 12);
        write_bytes += write(fd, src_chunk->score, sizeof(uint16_t) * Pool_hd_t::chunk_max_nvecs);
        write_bytes += write(fd, src_chunk->norm, sizeof(int32_t) * Pool_hd_t::chunk_max_nvecs);
        write_bytes += write(fd, src_chunk->u, sizeof(uint64_t) * Pool_hd_t::chunk_max_nvecs);
        write_bytes += write(fd, src_chunk->vec, _pool->CSD * Pool_hd_t::chunk_max_nvecs);
        #endif
        if (write_bytes < sizeof(uint16_t) * Pool_hd_t::chunk_max_nvecs +
                          sizeof(int32_t) * Pool_hd_t::chunk_max_nvecs + 
                          sizeof(uint64_t) * Pool_hd_t::chunk_max_nvecs + 
                          _pool->CSD * Pool_hd_t::chunk_max_nvecs + (ONE_TIME_IO ? 4096 : 12)) {
            lg_err("bytes write to chunk %lx less than expect, ignored", chunk_id);
        }
        if (ftruncate(fd, write_bytes) == -1) {
            #if MULTI_SSD
            lg_err("ftruncate %s/%s/%s%06lx failed, %s, ignored", _dir, hw::ssd_name(chunk_id), _pfx, chunk_id, strerror(errno));
            #else
            lg_err("ftruncate %s%06lx failed, %s, ignored", _prefix, chunk_id, strerror(errno));
            #endif
        }
        close(fd);
    }

    pthread_spin_lock(&_locks[chunk_id % pwc_locks]);
    _chunk_status[chunk_id] &= ~_ck_syncing;
    pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);
    _num_syncing_chunks--;
    __signal_sync_done();

    #if ENABLE_PROFILING
    ev_ssd_st.fetch_add(1);
    #endif

    lg_exit();
}

template <class logger_t> void pwc_manager_tmpl<logger_t>::__signal_sync_done() {
    constexpr chunk_status_t _ck_busy = _ck_loading | _ck_syncing | 
                                        _ck_reading | _ck_writing;
    lg_init();
    pthread_spin_lock(&_to_sync_chunks_lock);
    long queue_size = _to_sync_chunks.size();
    pthread_spin_unlock(&_to_sync_chunks_lock);
    long num_try = 0;

    volatile chunk_status_t *_chunk_status_vol = reinterpret_cast<volatile chunk_status_t*>(_chunk_status);

    if (queue_size > _max_cached_chunks * 5) {
        for (int i = 0; i < queue_size; i++) {
            pthread_spin_lock(&_to_sync_chunks_lock);
            if (_to_sync_chunks.empty()) {
                pthread_spin_unlock(&_to_sync_chunks_lock);
                break;
            }
            int32_t chunk_id = _to_sync_chunks.front();
            _to_sync_chunks.pop();
            if (_chunk_status[chunk_id] & (_ck_busy | _ck_to_sync)) {
                _to_sync_chunks.push(chunk_id);
                pthread_spin_unlock(&_to_sync_chunks_lock);
            } else {
                pthread_spin_unlock(&_to_sync_chunks_lock);
                pthread_spin_lock(&_locks[chunk_id % pwc_locks]);
                if (_chunk_status_vol[chunk_id] & (_ck_busy | _ck_to_sync)) {
                    pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);
                    pthread_spin_lock(&_to_sync_chunks_lock);
                    _to_sync_chunks.push(chunk_id);
                    pthread_spin_unlock(&_to_sync_chunks_lock);
                } else pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);
            }
        }
    }

    while (num_try < queue_size && 
           _num_syncing_chunks.load() < pwc_max_parallel_sync_chunks) {
        pthread_spin_lock(&_to_sync_chunks_lock);
        if (_to_sync_chunks.empty()) {
            pthread_spin_unlock(&_to_sync_chunks_lock);
            break;
        }
        int32_t chunk_id = _to_sync_chunks.front();
        _to_sync_chunks.pop();
        pthread_spin_unlock(&_to_sync_chunks_lock);

        if (_chunk_status[chunk_id] & _ck_busy) {
            pthread_spin_lock(&_to_sync_chunks_lock);
            _to_sync_chunks.push(chunk_id);
            pthread_spin_unlock(&_to_sync_chunks_lock);
        } else {
            pthread_spin_lock(&_locks[chunk_id % pwc_locks]);
            chunk_status_t status = _chunk_status_vol[chunk_id];
            if (status & _ck_busy) {
                pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);
                pthread_spin_lock(&_to_sync_chunks_lock);
                _to_sync_chunks.push(chunk_id);
                pthread_spin_unlock(&_to_sync_chunks_lock);
            } else if (!(status & _ck_to_sync)) {
                pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);
            } else {
                status &= ~_ck_to_sync;
                status |= _ck_syncing;
                _chunk_status[chunk_id] = status;
                pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);
                _syncing_pool.push([=]() { __sync_chunk(chunk_id); });
                _num_syncing_chunks++;
            }
        }
        
        num_try++;
    }

    lg_exit();
}

template <class logger_t> void pwc_manager_tmpl<logger_t>::__free_all() {
    lg_init();
    constexpr chunk_status_t _ck_busy = _ck_loading | _ck_syncing | 
                                          _ck_reading | _ck_writing | _ck_to_sync;
    volatile chunk_status_t *_chunk_status_vol = reinterpret_cast<volatile chunk_status_t *>(_chunk_status);
    volatile chunk_t *_cached_chunks_vol = reinterpret_cast<volatile chunk_t *>(_cached_chunks);
    
    int32_t *cache_to_evict = (int32_t *) malloc(_max_cached_chunks * sizeof(int32_t));
    long num_cache_to_evict = 0;

    for (int32_t ptr = 0; ptr < _max_cached_chunks; ptr++) {
        TRY_LOCK_THEN_FREE_CHUNK(_cached_chunks[ptr].id, _cached_chunks_vol[ptr].id);
        cache_to_evict[num_cache_to_evict] = ptr;
        num_cache_to_evict++;
    }

    while (num_cache_to_evict) {
        int still_using = 0;
        for (long j = 0; j < num_cache_to_evict; j++) {
            int32_t ptr = cache_to_evict[j];
            TRY_LOCK_THEN_FREE_CHUNK(_cached_chunks[ptr].id, _cached_chunks_vol[ptr].id);
            cache_to_evict[still_using] = ptr;
            still_using++;
        }
        num_cache_to_evict = still_using;
    }
    free(cache_to_evict);

    lg_exit();

    #undef TRY_LOCK_THEN_FREE_CHUNK
}

template <class logger_t> pwc_manager_tmpl<logger_t>::pwc_manager_tmpl(long loading_threads, long syncing_threads, long max_cached_chunks) {
    #if ENABLE_PROFILING
    this->logger = new logger_t();
    #endif

    for (long i = 0; i < pwc_locks; i++) {
        pthread_spin_init(&this->_locks[i], PTHREAD_PROCESS_SHARED);
    }
    pthread_spin_init(&this->_cached_chunks_lock, PTHREAD_PROCESS_SHARED);
    pthread_spin_init(&this->_to_sync_chunks_lock, PTHREAD_PROCESS_SHARED);

    _num_chunks = 0;
    _max_cached_chunks = 0;
    _cached_chunks = (chunk_t *) malloc((_ck_cache_id_mask + 1) * sizeof(chunk_t));
    _chunk_status = (chunk_status_t *) malloc((_ck_cache_id_mask + 1) * sizeof(chunk_status_t));

    set_num_threads(loading_threads, syncing_threads);
    set_max_cached_chunks(max_cached_chunks);

    _last_cache = 0;
    _num_loading_chunks.store(0);
    _num_syncing_chunks.store(0);

    _num_deleted_ids = 0;
    _deleted_ids = (int32_t *) malloc((_ck_cache_id_mask + 1) * sizeof(int32_t));
    pthread_spin_init(&this->_deleted_ids_lock, PTHREAD_PROCESS_SHARED);
    
    #if ENABLE_PROFILING
    logger->_construcion_done = 1;
    #endif
}

template <class logger_t> pwc_manager_tmpl<logger_t>::~pwc_manager_tmpl() {
    _syncing_pool.wait_work();
    _loading_pool.wait_work();
    __free_all();
    for (long i = 0; i < pwc_locks; i++) {
        pthread_spin_destroy(&this->_locks[i]);
    }
    pthread_spin_destroy(&this->_cached_chunks_lock);
    pthread_spin_destroy(&this->_to_sync_chunks_lock);

    free(_cached_chunks);
    free(_chunk_status);

    free(_deleted_ids);
    pthread_spin_destroy(&this->_deleted_ids_lock);

    #if ENABLE_PROFILING
    if (this->logger) delete this->logger;
    #endif
}

template <class logger_t> pwc_manager_tmpl<logger_t>::pwc_manager_tmpl() : 
                          pwc_manager_tmpl(pwc_default_loading_threads, pwc_default_syncing_threads, pwc_default_max_cached_chunks) {}

#if MULTI_SSD
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#endif

template <class logger_t> int pwc_manager_tmpl<logger_t>::_set_prefix(const char *dirname, const char *basis_prefix) {
    #if MULTI_SSD
    int first_time_set = this->_pfx[0] == '\0';
    int ret = snprintf(this->_dir, sizeof(this->_dir), ".%s", dirname);
    if (ret >= sizeof(this->_dir)) {
        this->_dir[sizeof(this->_dir) - 1] = '\0';
        lg_warn("dir truncated");
    }
    ret = snprintf(this->_pfx, sizeof(this->_pfx), ".%s_", basis_prefix);
    if (ret >= sizeof(this->_pfx)) {
        this->_pfx[sizeof(this->_pfx) - 1] = '\0';
        lg_warn("pfx truncated");
    }
    
    for (int i = 0; i < hw::ssd_num; i++) {
        struct stat st;
        char pdirname[256];
        snprintf(pdirname, 256, ".%s/%s", dirname, hw::ssd_name_list[i]);
        if (stat(pdirname, &st) != 0 || !S_ISDIR(st.st_mode)) {
            lg_err("directory %s not found, aborting", pdirname);
            abort();
        }
    }

    #if ENABLE_PROFILING
    if (logger->construcion_done() && !first_time_set) 
        lg_info("pfx and dir set to %s and %s", this->_pfx, this->_dir);
    #endif
    #else
    int first_time_set = this->_prefix[0] == '\0';
    int ret = snprintf(this->_prefix, sizeof(this->_prefix), ".%s/.%s_", dirname, basis_prefix);
    if (ret >= sizeof(this->_prefix)) {
        this->_prefix[sizeof(this->_prefix) - 1] = '\0';
        lg_warn("prefix truncated");
        return -1;
    }
    
    char pdirname[256];
    snprintf(pdirname, 256, ".%s", dirname); 
    if (mkdir(pdirname, 0755) && errno != EEXIST) {
        lg_warn("mkdir %s failed, %s", pdirname, strerror(errno));
        return -1;
    }

    #if ENABLE_PROFILING
    if (logger->construcion_done() && !first_time_set) lg_info("prefix set to %s", this->_prefix);
    #endif
    #endif

    return 0;
}

template <class logger_t> int pwc_manager_tmpl<logger_t>::set_dirname(const char *dirname) {
    char prefix[9];
    for (long i = 0; i < 8; i++) {
        char c = (char) ((_pool->basis_hash >> (i * 8)) & 0x3f);
        if (c < 26) {
            prefix[i] = 'a' + c;
        } else if (c < 52) {
            prefix[i] = 'A' + c - 26;
        } else if (c < 62) {
            prefix[i] = '0' + c - 52;
        } else if (c == 62) {
            prefix[i] = '&';
        } else {
            prefix[i] = '@';
        }
    }
    prefix[8] = '\0';

    return this->_set_prefix(dirname, prefix);
}

template <class logger_t> int pwc_manager_tmpl<logger_t>::set_pool(Pool_hd_t *pool) {
    this->_pool = pool;
    this->set_dirname("pool");
    return 0;
}

template <class logger_t> long pwc_manager_tmpl<logger_t>::num_vec() const {
    long ret = 0;
    
    #pragma omp parallel for reduction(+:ret) num_threads(_loading_threads)
    for (long i = 0; i < _num_chunks; i++) {
        long tmp = this->chunk_size(i);
        ret += tmp == _ck_size_mask ? 0 : tmp;
    }

    return ret;
}

template <class logger_t> long pwc_manager_tmpl<logger_t>::create_chunk() {
    lg_init();
    volatile int *_num_deleted_ids_vol_ptr = reinterpret_cast<volatile int *>(&_num_deleted_ids);
    volatile long *_num_chunks_vol_ptr = reinterpret_cast<volatile long *>(&_num_chunks);

    if (_num_deleted_ids) {
        pthread_spin_lock(&_deleted_ids_lock);
        if (*_num_deleted_ids_vol_ptr) {
            long ret = _deleted_ids[--_num_deleted_ids];
            _chunk_status[ret] = 0;
            pthread_spin_unlock(&_deleted_ids_lock);
            lg_exit();
            return ret;
        }
        pthread_spin_unlock(&_deleted_ids_lock);
    }
    for (;;) {
        long chunk_id = _num_chunks;
        pthread_spin_lock(&_locks[chunk_id % pwc_locks]);
        if (*_num_chunks_vol_ptr == chunk_id) {
            // successfully locked
            _chunk_status[chunk_id] = 0;
            _num_chunks++;
            pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);
            lg_exit();
            return chunk_id;
        }
        pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);
    }
}
    
template <class logger_t> int pwc_manager_tmpl<logger_t>::prefetch(long chunk_id) {
    lg_init();
    if (chunk_id >= _num_chunks || chunk_id < 0) { lg_exit(); return 0; }
    if (_chunk_status[chunk_id] == _ck_size_mask) { lg_exit(); return 0; }
    
    constexpr chunk_status_t _ck_already_loaded = _ck_caching | _ck_loading;
    volatile chunk_status_t *_chunk_status_vol = reinterpret_cast<volatile chunk_status_t*>(_chunk_status);

    if (_chunk_status[chunk_id] & _ck_already_loaded) { lg_exit(); return 0; }

    pthread_spin_lock(&_locks[chunk_id % pwc_locks]);
    chunk_status_t status = _chunk_status_vol[chunk_id];
    if (!(status & _ck_already_loaded) && (status != _ck_size_mask) && !(status & (_ck_writing | _ck_reading))) {
        _chunk_status[chunk_id] |= _ck_loading;
        _loading_pool.push([=]() { __load_chunk(chunk_id); });
        _num_loading_chunks++;
    }
    pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);

    lg_exit();
    return 0;
}

template <class logger_t> chunk_t *pwc_manager_tmpl<logger_t>::fetch(long chunk_id) {
    lg_init();
    constexpr chunk_status_t _ck_already_loaded = _ck_caching | _ck_loading;
    constexpr chunk_status_t _ck_busy = _ck_loading | _ck_syncing | 
                                        _ck_reading | _ck_writing;
    volatile chunk_status_t *_chunk_status_vol = reinterpret_cast<volatile chunk_status_t*>(_chunk_status);
    
    if (_chunk_status[chunk_id] == _ck_size_mask) { lg_exit(); return NULL; }

    #if ENABLE_PROFILING
    int first_try = 0x2;
    if (logger_t::_log_prefix == pwc_logger_t::_log_prefix) {
        ev_pwc_fetch.fetch_add(1);
    }
    #endif

    for (;;) {
        #if ENABLE_PROFILING
        if (logger_t::_log_prefix == pwc_logger_t::_log_prefix) first_try >>= 1;
        #endif
        // I choose to poll instead of using condition variable now
        if (!(_chunk_status_vol[chunk_id] & _ck_already_loaded)) {
            pthread_spin_lock(&_locks[chunk_id % pwc_locks]);
            chunk_status_t status = _chunk_status_vol[chunk_id];
            if (!(status & _ck_already_loaded) && !(status & _ck_busy)) {
                if (status == _ck_size_mask) {
                    pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);
                    lg_exit();
                    return NULL;
                }
                _chunk_status[chunk_id] |= _ck_loading;
                _loading_pool.push([=]() { __load_chunk(chunk_id); });
                _num_loading_chunks++;
            }
            pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);
        }
        if ((_chunk_status_vol[chunk_id] & (_ck_busy | _ck_caching)) != _ck_caching) continue;
        pthread_spin_lock(&_locks[chunk_id % pwc_locks]);
        chunk_status_t status = _chunk_status_vol[chunk_id];
        if ((status & (_ck_busy | _ck_caching)) != _ck_caching) {
            pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);
            continue;
        }
        if (status == _ck_size_mask) {
            pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);
            lg_exit();
            return NULL;
        }
        _chunk_status[chunk_id] |= _ck_writing;
        pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);
        lg_exit();
        #if ENABLE_PROFILING
        if (logger_t::_log_prefix == pwc_logger_t::_log_prefix && first_try) ev_pwc_cache_hit.fetch_add(1);
        #endif
        return &_cached_chunks[_chunk_status[chunk_id] & _ck_cache_id_mask];
    }
}

template <class logger_t> int pwc_manager_tmpl<logger_t>::release(long chunk_id) {
    lg_init();
    _chunk_status[chunk_id] &= ~(_ck_writing | _ck_reading);
    lg_exit();

    return 0;
}

template <class logger_t> int pwc_manager_tmpl<logger_t>::release_sync(long chunk_id) {
    lg_init();
    if (_chunk_status[chunk_id] & _ck_to_sync) {
        _chunk_status[chunk_id] &= ~(_ck_writing | _ck_reading);
        lg_exit();
        return 0;
    }

    pthread_spin_lock(&_to_sync_chunks_lock);
    if (_num_syncing_chunks.load() < pwc_max_parallel_sync_chunks) {
        chunk_status_t new_status = _chunk_status[chunk_id];
        new_status &= ~(_ck_writing | _ck_reading | _ck_to_sync);
        new_status |= _ck_syncing;
        _chunk_status[chunk_id] = new_status;
        _syncing_pool.push([=]() { __sync_chunk(chunk_id); });
        _num_syncing_chunks++;
    } else {
        _chunk_status[chunk_id] |= _ck_to_sync;
        _to_sync_chunks.push(chunk_id);
        _chunk_status[chunk_id] &= ~(_ck_writing | _ck_reading);
    }
    pthread_spin_unlock(&_to_sync_chunks_lock);

    lg_exit();
    return 0;
}

template <class logger_t> int pwc_manager_tmpl<logger_t>::sync_release(long chunk_id) {
    lg_init();

    pthread_spin_lock(&_locks[chunk_id % pwc_locks]);
    chunk_status_t new_status = _chunk_status[chunk_id];
    new_status &= ~(_ck_writing | _ck_reading | _ck_to_sync);
    new_status |= _ck_syncing;
    _chunk_status[chunk_id] = new_status;
    pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);
    _num_syncing_chunks++;
    __sync_chunk(chunk_id);

    lg_exit();
    return 0;
}

template <class logger_t> int pwc_manager_tmpl<logger_t>::release_del(long chunk_id) {
    lg_init();
    if ((_chunk_status[chunk_id] & _ck_syncing) == 0) {
        if (_chunk_status[chunk_id] & _ck_caching) {
            pthread_spin_lock(&_cached_chunks_lock);
            _cached_chunks[_chunk_status[chunk_id] & _ck_cache_id_mask].size = 0;
            _cached_chunks[_chunk_status[chunk_id] & _ck_cache_id_mask].id = -1;
            pthread_spin_unlock(&_cached_chunks_lock);
        }
        char chunk_filename[256];
        #if MULTI_SSD
        snprintf(chunk_filename, sizeof(chunk_filename), "%s/%s/%s%06lx", _dir, hw::ssd_name(chunk_id), _pfx, chunk_id);
        #else
        snprintf(chunk_filename, sizeof(chunk_filename), "%s%06lx", _prefix, chunk_id);
        #endif
        remove(chunk_filename);

        pthread_spin_lock(&_locks[chunk_id % pwc_locks]);
        _chunk_status[chunk_id] = _ck_size_mask;
        pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);

        pthread_spin_lock(&_deleted_ids_lock);
        _deleted_ids[_num_deleted_ids++] = chunk_id;
        pthread_spin_unlock(&_deleted_ids_lock);

        lg_exit();
        return 0;
    } 

    _loading_pool.push([this, chunk_id]() {
        volatile chunk_status_t *_chunk_status_vol = reinterpret_cast<volatile chunk_status_t*>(_chunk_status);
        
        do {} while (_chunk_status_vol[chunk_id] & _ck_syncing);

        if (_chunk_status[chunk_id] & _ck_caching) {
            pthread_spin_lock(&_cached_chunks_lock);
            _cached_chunks[_chunk_status[chunk_id] & _ck_cache_id_mask].size = 0;
            _cached_chunks[_chunk_status[chunk_id] & _ck_cache_id_mask].id = -1;
            pthread_spin_unlock(&_cached_chunks_lock);
        }
        char chunk_filename[256];
        #if MULTI_SSD
        snprintf(chunk_filename, sizeof(chunk_filename), "%s/%s/%s%06lx", _dir, hw::ssd_name(chunk_id), _pfx, chunk_id);
        #else
        snprintf(chunk_filename, sizeof(chunk_filename), "%s%06lx", _prefix, chunk_id);
        #endif
        remove(chunk_filename);

        pthread_spin_lock(&_locks[chunk_id % pwc_locks]);
        _chunk_status[chunk_id] = _ck_size_mask;
        pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);

        pthread_spin_lock(&_deleted_ids_lock);
        _deleted_ids[_num_deleted_ids++] = chunk_id;
        pthread_spin_unlock(&_deleted_ids_lock);
    });

    lg_exit();
    return 0;
}

template <class logger_t> int pwc_manager_tmpl<logger_t>::wait_work() {
    lg_init();
    _syncing_pool.wait_work();
    _loading_pool.wait_work();

    long queue_size = _to_sync_chunks.size();
    if (queue_size) lg_warn("%d chunks to sync after wait done?", queue_size);

    if (_num_loading_chunks.load() || _num_syncing_chunks.load()) {
        lg_warn("wait_work: %d loading and %d syncing after wait done?", 
                _num_loading_chunks.load(), _num_syncing_chunks.load());
        lg_exit();
        return -1;
    } 
    lg_exit();
    return 0;
}

#endif
#ifndef __BGJ_HD_H
#define __BGJ_HD_H

#include "./pool_hd.h"

#if ENABLE_PROFILING
struct bwc_logger_t : public pwc_logger_t {
    inline int log_level() override { return _ll; }
    inline uint64_t log_prefix() override { return _log_prefix; }

    inline void init(const char *keyfunc) override {}
    inline void exit(const char *keyfunc) override {}
    inline void report(const char *keyfunc) override {}

    //private:
    static constexpr int _ll = BWC_LOG_LEVEL;
    static constexpr uint64_t _log_prefix = 'b' + ('w' << 8) + ('c' << 16);
};

struct swc_logger_t : public pwc_logger_t {
    inline int log_level() override { return _ll; }
    inline uint64_t log_prefix() override { return _log_prefix; }

    inline void init(const char *keyfunc) override {}
    inline void exit(const char *keyfunc) override {}
    inline void report(const char *keyfunc) override {}

    //private:
    static constexpr int _ll = SWC_LOG_LEVEL;
    static constexpr uint64_t _log_prefix = 's' + ('w' << 8) + ('c' << 16);
};
#endif


template <class logger_t>
struct bwc_manager_tmpl : private pwc_manager_tmpl<logger_t> {
    public:
    static constexpr long bwc_default_loading_threads = BWC_DEFAULT_LOADING_THREADS;
    static constexpr long bwc_default_syncing_threads = BWC_DEFAULT_SYNCING_THREADS;
    static constexpr long bwc_default_max_cached_chunks = BWC_DEFAULT_MAX_CACHED_CHUNKS;
    static constexpr long bwc_max_parallel_sync_chunks = BWC_MAX_PARALLEL_SYNC_CHUNKS;
    static constexpr long bwc_max_buckets = BWC_MAX_BUCKETS;
    static constexpr long bwc_auto_prefetch_for_write = 64;
    static constexpr long bwc_auto_prefetch_for_read = 8;
    static constexpr long bwc_auto_prefetch_for_read_depth = 64;
    static constexpr long bwc_bucket_locks = 256;

    bwc_manager_tmpl(Pool_hd_t *p);
    ~bwc_manager_tmpl();

    using pwc_manager_tmpl<logger_t>::set_dirname;
    using pwc_manager_tmpl<logger_t>::set_num_threads;
    using pwc_manager_tmpl<logger_t>::set_max_cached_chunks;
    using pwc_manager_tmpl<logger_t>::max_cached_chunks;


    // create an empty bucket, return bucket id
    long push_bucket();
    // get a bucket from ready list, return bucket id, return -1 if no bucket is ready 
    long pop_bucket();
    // tell the manager that all write/read to the bucket is done so it's ready for use/del
    void bucket_finalize(long bucket_id);
    // return the number of buckets that are ready
    long num_ready();
    
    // make sure the bucket is in prepare, otherwise return NULL
    chunk_t *fetch_for_write(long bucket_id);
    // make sure the bucket is ready, otherwise return NULL
    // also return NULL if the bucket is empty
    chunk_t *fetch_for_read(long bucket_id);
    // when the chunk gets full, sync automatically
    void write_done(chunk_t *chunk, long bucket_id);
    // when the bucket is empty, delete automatically
    void read_done(chunk_t *chunk, long bucket_id);
    long bucket_num_chunks(long bucket_id);

    #if ENABLE_PROFILING
    using pwc_manager_tmpl<logger_t>::logger;
    using pwc_manager_tmpl<logger_t>::ev_f4w;
    using pwc_manager_tmpl<logger_t>::ev_f4r;
    using pwc_manager_tmpl<logger_t>::ev_f4w_hit;
    using pwc_manager_tmpl<logger_t>::ev_f4r_hit;
    using pwc_manager_tmpl<logger_t>::ev_ssd_ld;
    using pwc_manager_tmpl<logger_t>::ev_ssd_st;
    #endif

    private:
    struct l0_bucket_t {
        static constexpr long init_alloc_chunks = 256;

        static constexpr uint32_t _bk_writing   = 0x80000000;
        static constexpr uint32_t _bk_ready     = 0x40000000;
        static constexpr uint32_t _bk_reading   = 0x20000000;
        static constexpr uint32_t _bk_caching   = 0x10000000;
        
        uint32_t status = 0;
        int32_t num_chunks = 0;
        int32_t *chunk_ids = NULL;
        chunk_t *writing_chunk = NULL;

        inline int init() {
            int ret = 0;
            if (chunk_ids == NULL) {
                chunk_ids = (int32_t *) malloc(l0_bucket_t::init_alloc_chunks * sizeof(int32_t));
                if (chunk_ids == NULL) {
                    ret = -1;
                    fprintf(stderr, "[Error] l0_bucket_t::init: allocation fail\n");
                }
            }
            status = _bk_writing;
            num_chunks = 0;
            writing_chunk = NULL;
            alloc_size = l0_bucket_t::init_alloc_chunks;
            return ret;
        }

        inline int add_chunk(int32_t chunk_id) {
            if (num_chunks == alloc_size) {
                chunk_ids = (int32_t *)realloc(chunk_ids, sizeof(int32_t) * alloc_size * 2);
                if (chunk_ids == NULL) {
                    fprintf(stderr, "[Error] bwc_manager_t::l0_bucket_t::add_chunk: realloc failed\n");
                    return -1;
                }
                alloc_size *= 2;
            }
            chunk_ids[num_chunks++] = chunk_id;
            return 0;
        }
        private:
        int32_t alloc_size;
    };

    int32_t _num_buckets;
    l0_bucket_t _bucket[bwc_max_buckets];
    int32_t _num_deleted_buckets;   
    int32_t _deleted_bucket_ids[bwc_max_buckets];
    pthread_spinlock_t _bucket_lock[bwc_bucket_locks];

    int32_t _num_wl, _num_wp;
    chunk_t *_writing_prefetch_chunks[bwc_auto_prefetch_for_write];
    pthread_spinlock_t _bwc_wp_lock;

    int32_t _num_ready_buckets;
    int32_t _num_prefetched_bucket;
    int32_t _ready_bucket_id[bwc_max_buckets];
    int32_t _prefetched_bucket_id[bwc_auto_prefetch_for_read];
    pthread_spinlock_t _bwc_lock;

    void __prefetch_for_writing();
    void __prefetch_for_reading(int32_t bucket_id);

    using typename pwc_manager_tmpl<logger_t>::chunk_status_t;
    using pwc_manager_tmpl<logger_t>::_ck_writing;
    using pwc_manager_tmpl<logger_t>::_ck_reading;
    using pwc_manager_tmpl<logger_t>::_ck_loading;
    using pwc_manager_tmpl<logger_t>::_ck_syncing;
    using pwc_manager_tmpl<logger_t>::_ck_caching;
    using pwc_manager_tmpl<logger_t>::_ck_to_sync;
    using pwc_manager_tmpl<logger_t>::_ck_size_mask;
    using pwc_manager_tmpl<logger_t>::_ck_cache_id_mask;
    using pwc_manager_tmpl<logger_t>::_locks;
    using pwc_manager_tmpl<logger_t>::pwc_locks;
    using pwc_manager_tmpl<logger_t>::create_chunk;
    using pwc_manager_tmpl<logger_t>::prefetch;
    using pwc_manager_tmpl<logger_t>::release_del;
    using pwc_manager_tmpl<logger_t>::release_sync;
    using pwc_manager_tmpl<logger_t>::sync_release;
    using pwc_manager_tmpl<logger_t>::__load_chunk;
    using pwc_manager_tmpl<logger_t>::_num_chunks;
    using pwc_manager_tmpl<logger_t>::_chunk_status;
    using pwc_manager_tmpl<logger_t>::_cached_chunks;
    using pwc_manager_tmpl<logger_t>::_loading_pool;
    using pwc_manager_tmpl<logger_t>::_num_loading_chunks;
    using pwc_manager_tmpl<logger_t>::_loading_threads;
    using pwc_manager_tmpl<logger_t>::_syncing_threads;
    using pwc_manager_tmpl<logger_t>::_max_cached_chunks;
};

/// @brief solution with cache, receive solutions from Reducer_t and from here
///        Bucketer_t will get solutions and insert them into the pool.
template <class logger_t>
struct swc_manager_tmpl : private pwc_manager_tmpl<logger_t> {
    public:
    static constexpr long swc_default_loading_threads = SWC_DEFAULT_LOADING_THREADS;
    static constexpr long swc_default_syncing_threads = SWC_DEFAULT_SYNCING_THREADS;
    static constexpr long swc_default_max_cached_chunks = SWC_DEFAULT_MAX_CACHED_CHUNKS;
    static constexpr long swc_max_parallel_sync_chunks = SWC_MAX_PARALLEL_SYNC_CHUNKS;
    static constexpr long swc_auto_prefetch_for_write = 8;
    static constexpr long swc_auto_prefetch_for_read = 64;
    static constexpr long swc_max_ready_chunks = pwc_manager_tmpl<logger_t>::_ck_cache_id_mask + 1;
    static constexpr long swc_max_writing_chunks = swc_default_max_cached_chunks;
    static constexpr long swc_auto_finalize = 1;

    swc_manager_tmpl(Pool_hd_t *p);
    ~swc_manager_tmpl();

    using pwc_manager_tmpl<logger_t>::num_vec;
    using pwc_manager_tmpl<logger_t>::set_dirname;
    using pwc_manager_tmpl<logger_t>::set_num_threads;
    using pwc_manager_tmpl<logger_t>::set_max_cached_chunks;
    using pwc_manager_tmpl<logger_t>::max_cached_chunks;

    chunk_t *fetch_for_write();
    // return NULL if no solution is ready
    chunk_t *fetch_for_read();
    // release only
    void write_done(chunk_t *chunk);
    // delete the chunk after read
    void read_done(chunk_t *chunk);
    // release and tell the manager that all write to the chunk is done so it's ready for use
    void chunk_finalize(chunk_t *chunk);
    // return the number of chunks that are ready
    long num_ready();
    // return a rough number of solutions that are ready
    long ready_nvecs_estimate();
    // return the number of chunks that are active
    long num_using();
    // finalize all writing chunks, aborted
    // long finalize_all_writing();


    #if ENABLE_PROFILING
    using pwc_manager_tmpl<logger_t>::logger;
    using pwc_manager_tmpl<logger_t>::ev_f4w;
    using pwc_manager_tmpl<logger_t>::ev_f4r;
    using pwc_manager_tmpl<logger_t>::ev_f4w_hit;
    using pwc_manager_tmpl<logger_t>::ev_f4r_hit;
    using pwc_manager_tmpl<logger_t>::ev_ssd_ld;
    using pwc_manager_tmpl<logger_t>::ev_ssd_st;
    #endif

    private:
    int32_t _num_ready;
    int32_t _num_writing;
    int32_t _num_rp, _num_rl;
    int32_t _num_wp, _num_wl;
    
    int32_t *_ready_chunks;
    chunk_t *_writing_chunks[swc_max_writing_chunks];
    chunk_t *_writing_prefetch_chunks[swc_auto_prefetch_for_write];
    chunk_t *_reading_prefetch_chunks[swc_auto_prefetch_for_read];
    
    pthread_spinlock_t _swc_lock;

    void __prefetch_for_writing();
    void __prefetch_for_reading();

    friend int ut_checker_t::batch(long tid, long table_size, long current_hold);

    using typename pwc_manager_tmpl<logger_t>::chunk_status_t;
    using pwc_manager_tmpl<logger_t>::_ck_writing;
    using pwc_manager_tmpl<logger_t>::_ck_reading;
    using pwc_manager_tmpl<logger_t>::_ck_loading;
    using pwc_manager_tmpl<logger_t>::_ck_caching;
    using pwc_manager_tmpl<logger_t>::_ck_size_mask;
    using pwc_manager_tmpl<logger_t>::_ck_cache_id_mask;
    using pwc_manager_tmpl<logger_t>::_locks;
    using pwc_manager_tmpl<logger_t>::pwc_locks;
    using pwc_manager_tmpl<logger_t>::__load_chunk;
    using pwc_manager_tmpl<logger_t>::create_chunk;
    using pwc_manager_tmpl<logger_t>::release_del;
    using pwc_manager_tmpl<logger_t>::release_sync;
    using pwc_manager_tmpl<logger_t>::sync_release;
    using pwc_manager_tmpl<logger_t>::_num_chunks;
    using pwc_manager_tmpl<logger_t>::_num_deleted_ids;
    using pwc_manager_tmpl<logger_t>::_chunk_status;
    using pwc_manager_tmpl<logger_t>::_cached_chunks;
    using pwc_manager_tmpl<logger_t>::_num_loading_chunks;
    using pwc_manager_tmpl<logger_t>::_loading_threads;
    using pwc_manager_tmpl<logger_t>::_syncing_threads;
    using pwc_manager_tmpl<logger_t>::_max_cached_chunks;
    using pwc_manager_tmpl<logger_t>::chunk_size;
};

#endif
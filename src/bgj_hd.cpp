#include "../include/bgj_hd.h"

#include <sys/time.h>
#include <unistd.h>

#include <omp.h>

template <class logger_t> bwc_manager_tmpl<logger_t>::bwc_manager_tmpl(Pool_hd_t *p) : 
                          pwc_manager_tmpl<logger_t>(bwc_default_loading_threads, bwc_default_syncing_threads, bwc_default_max_cached_chunks) {
    _num_buckets = 0;
    _num_deleted_buckets = 0;
    _num_wl = 0;
    _num_wp = 0;
    _num_ready_buckets = 0;
    _num_prefetched_bucket = 0;
    
    for (long i = 0; i < bwc_bucket_locks; i++) {
        pthread_spin_init(&_bucket_lock[i], PTHREAD_PROCESS_SHARED);
    }

    pthread_spin_init(&_bwc_wp_lock, PTHREAD_PROCESS_SHARED);
    pthread_spin_init(&_bwc_lock, PTHREAD_PROCESS_SHARED);
    
    this->_pool = p;
    this->set_dirname("bucket");

    lg_info("manager initialized, (%d, %d) threads for I/O, #caching = %d", 
            _loading_threads, _syncing_threads, _max_cached_chunks);
}

template <class logger_t> bwc_manager_tmpl<logger_t>::~bwc_manager_tmpl() {
    this->_syncing_pool.wait_work();
    this->_loading_pool.wait_work();

    if (_num_wl) lg_err("still %d(4w) chunks loading? ignored.", _num_wl);
    
    for (int32_t i = 0; i < _num_wp; i++) release_del(_writing_prefetch_chunks[i]->id);
    pthread_spin_destroy(&_bwc_wp_lock);

    for (int32_t i = 0; i < _num_buckets; i++) {
        if (_bucket[i].status) {
            for (int32_t j = 0; j < _bucket[i].num_chunks; j++) {
                _chunk_status[_bucket[i].chunk_ids[j]] &= ~ (_ck_writing | _ck_reading | _ck_to_sync); 
            }
        }
        if (_bucket[i].chunk_ids) free(_bucket[i].chunk_ids);
        if (_bucket[i].writing_chunk && _bucket[i].writing_chunk != (chunk_t *) -1) release_del(_bucket[i].writing_chunk->id);
    }

    for (int32_t i = 0; i < bwc_bucket_locks; i++) {
        pthread_spin_destroy(&_bucket_lock[i]);
    }
    pthread_spin_destroy(&_bwc_lock);

    this->_syncing_pool.wait_work();
    this->_loading_pool.wait_work();

    /// delete all files
    char chunk_filename[256];
    for (long i = 0; i < _num_chunks; i++) {
        #if MULTI_SSD
        snprintf(chunk_filename, sizeof(chunk_filename), "%s/%s/%s%06lx", this->_dir, hw::ssd_name(i), this->_pfx, i);
        #else
        snprintf(chunk_filename, sizeof(chunk_filename), "%s%06lx", this->_prefix, i);
        #endif
        remove(chunk_filename);
    }
}

template <class logger_t> void bwc_manager_tmpl<logger_t>::__prefetch_for_writing() {
    lg_init();
    volatile int32_t *_num_wp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wp);
    volatile int32_t *_num_wl_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wl);

    // almost the same as swc_manager_t::__prefetch_for_writing
    if (_num_wl + _num_wp >= bwc_auto_prefetch_for_write) {
        lg_exit();
        return;
    }

    pthread_spin_lock(&_bwc_wp_lock);
    int to_prefetch = bwc_auto_prefetch_for_write - *_num_wl_ptr_vol - *_num_wp_ptr_vol;
    if (to_prefetch <= 0) {
        pthread_spin_unlock(&_bwc_wp_lock);
        lg_exit();
        return;
    }
    _num_wl += to_prefetch;
    pthread_spin_unlock(&_bwc_wp_lock);

    int32_t num_fail = 0;
    for (int32_t i = 0; i < to_prefetch; i++) {
        int32_t id = create_chunk();
        pthread_spin_lock(&_locks[id % pwc_locks]);
        if (_chunk_status[id]) {
            num_fail++;
        } else {
            _chunk_status[id] |= _ck_loading | _ck_writing;
            this->_loading_pool.push([=]() {
                __load_chunk(id);
                pthread_spin_lock(&_bwc_wp_lock);
                _writing_prefetch_chunks[_num_wp++] = &_cached_chunks[_chunk_status[id] & _ck_cache_id_mask];
                _num_wl--;
                pthread_spin_unlock(&_bwc_wp_lock);
            });
            _num_loading_chunks++;
        }
        pthread_spin_unlock(&_locks[id % pwc_locks]);
    }

    if (num_fail) {
        lg_err("%d of %d new chunk already in use?", num_fail, to_prefetch);
        pthread_spin_lock(&_bwc_wp_lock);
        _num_wl -= num_fail;
        pthread_spin_unlock(&_bwc_wp_lock);
    }
    lg_exit();
}

template <class logger_t> void bwc_manager_tmpl<logger_t>::__prefetch_for_reading(int32_t bucket_id) {
    lg_init();
    volatile uint32_t *status_ptr_vol = reinterpret_cast<volatile uint32_t*>(&_bucket[bucket_id].status);
    volatile  int32_t *num_pb_ptr_vol = reinterpret_cast<volatile  int32_t*>(&_num_prefetched_bucket);

    /// try to convert it to a hard prefetch bucket
    if (_bucket[bucket_id].status == l0_bucket_t::_bk_reading) {
        if (_num_prefetched_bucket < bwc_auto_prefetch_for_read) {
            pthread_spin_lock(&_bwc_lock);
            if (*num_pb_ptr_vol < bwc_auto_prefetch_for_read) {
                _prefetched_bucket_id[num_pb_ptr_vol[0]++] = bucket_id;
                pthread_spin_unlock(&_bwc_lock);

                pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                if (status_ptr_vol[0] == l0_bucket_t::_bk_reading) {
                    status_ptr_vol[0] |= l0_bucket_t::_bk_caching;
                    pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                } else {
                    pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                    pthread_spin_lock(&_bwc_lock);
                    for (int32_t i = 0; i < _num_prefetched_bucket; i++) {
                        if (_prefetched_bucket_id[i] == bucket_id) {
                            _prefetched_bucket_id[i] = _prefetched_bucket_id[--_num_prefetched_bucket];
                            break;
                        }
                    }
                    pthread_spin_unlock(&_bwc_lock);
                }
            } else pthread_spin_unlock(&_bwc_lock);
        }
    }
    

    if (_bucket[bucket_id].status & l0_bucket_t::_bk_caching) {
        pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        for (int32_t i = _bucket[bucket_id].num_chunks - 1; i >= 0 && 
                    i >= _bucket[bucket_id].num_chunks - bwc_auto_prefetch_for_read_depth; i--) {
            if (_chunk_status[_bucket[bucket_id].chunk_ids[i]] & _ck_writing) continue;
            pthread_spin_lock(&_locks[_bucket[bucket_id].chunk_ids[i] % pwc_locks]);
            _chunk_status[_bucket[bucket_id].chunk_ids[i]] |= _ck_writing;
            pthread_spin_unlock(&_locks[_bucket[bucket_id].chunk_ids[i] % pwc_locks]);
        }
        pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    }

    pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    for (int32_t i = _bucket[bucket_id].num_chunks - 1; i >= 0 && 
                i >= _bucket[bucket_id].num_chunks - bwc_auto_prefetch_for_read_depth; i--) {
        int32_t chunk_id = _bucket[bucket_id].chunk_ids[i];
        if (_chunk_status[chunk_id] & (_ck_caching | _ck_loading)) continue;
        if (chunk_id >= _num_chunks || chunk_id < 0 || _chunk_status[chunk_id] == _ck_size_mask) {
            lg_err("invalid chunk id %d(%x)", chunk_id, _chunk_status[chunk_id]);
            continue;
        }
        
        volatile chunk_status_t *_chunk_status_vol = reinterpret_cast<volatile chunk_status_t*>(_chunk_status);

        pthread_spin_lock(&_locks[chunk_id % pwc_locks]);
        chunk_status_t status = _chunk_status_vol[chunk_id];
        if (!(status & (_ck_caching | _ck_loading)) && (status != _ck_size_mask)) {
            _chunk_status[chunk_id] |= _ck_loading;
            _loading_pool.push([=]() { __load_chunk(chunk_id); });
            _num_loading_chunks++;
        }
        pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);
    }
    pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);

    lg_exit();
}

template <class logger_t> long bwc_manager_tmpl<logger_t>::push_bucket() {
    lg_init();
    volatile int32_t *_num_wp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wp);

    int32_t ret = -1;
    pthread_spin_lock(&_bwc_lock);
    if (_num_deleted_buckets) {
        ret = _deleted_bucket_ids[--_num_deleted_buckets];
    } else if (_num_buckets < bwc_max_buckets) {
        ret = _num_buckets++;
    } else lg_err("no place for new buckets(%ld)", bwc_max_buckets);
    pthread_spin_unlock(&_bwc_lock);

    if (ret >= 0) {
        _bucket[ret].init();

        if (_num_wp) {
            pthread_spin_lock(&_bwc_wp_lock);
            if (*_num_wp_ptr_vol) _bucket[ret].writing_chunk = _writing_prefetch_chunks[--_num_wp];
            pthread_spin_unlock(&_bwc_wp_lock);
            if (_bucket[ret].writing_chunk) {
                _bucket[ret].add_chunk(_bucket[ret].writing_chunk->id);
            }
        }

        __prefetch_for_writing();
    }

    lg_exit();
    return ret;
}

template <class logger_t> long bwc_manager_tmpl<logger_t>::pop_bucket() {
    lg_init();
    int32_t ret = -1;

    pthread_spin_lock(&_bwc_lock);
    for (int32_t i = 0; i < _num_prefetched_bucket; i++) {
        if (_bucket[_prefetched_bucket_id[i]].status & l0_bucket_t::_bk_ready) {
            ret = _prefetched_bucket_id[i];
            _bucket[_prefetched_bucket_id[i]].status = 
            (_bucket[_prefetched_bucket_id[i]].status & ~l0_bucket_t::_bk_ready) | l0_bucket_t::_bk_reading;
            break;
        }
    }
    if (ret == -1 && _num_ready_buckets) {
        ret = _ready_bucket_id[--_num_ready_buckets];
        _bucket[ret].status = (_bucket[ret].status & ~l0_bucket_t::_bk_ready) | l0_bucket_t::_bk_reading;        
        if (_num_prefetched_bucket < bwc_auto_prefetch_for_read) {
            _prefetched_bucket_id[_num_prefetched_bucket++] = ret;
            _bucket[ret].status |= l0_bucket_t::_bk_caching;
        }
    }
    pthread_spin_unlock(&_bwc_lock);

    if (ret != -1) __prefetch_for_reading(ret);

    lg_exit();
    return ret;
}

template <class logger_t> void bwc_manager_tmpl<logger_t>::bucket_finalize(long bucket_id) {
    lg_init();

    volatile int32_t *_num_wp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wp);
    volatile uint32_t *_chunk_status_vol = reinterpret_cast<volatile uint32_t*>(_chunk_status);
    chunk_t *volatile *writing_chunk_ptr_vol = reinterpret_cast<chunk_t *volatile *>(&_bucket[bucket_id].writing_chunk);

    if (_bucket[bucket_id].status & l0_bucket_t::_bk_reading) {
        pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        int32_t num_chunks = _bucket[bucket_id].num_chunks;
        _bucket[bucket_id].num_chunks = 0;
        pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);

        pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        _bucket[bucket_id].status = 0;
        pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        
        pthread_spin_lock(&_bwc_lock);
        for (int32_t i = 0; i < _num_prefetched_bucket; i++) {
            if (_prefetched_bucket_id[i] == bucket_id) 
                _prefetched_bucket_id[i] = _prefetched_bucket_id[--_num_prefetched_bucket];
        }
        pthread_spin_unlock(&_bwc_lock);

        /// clean all chunks in the chunk list
        struct timeval start, end;
        gettimeofday(&start, NULL);
        while (num_chunks) {
            for (int32_t i = num_chunks - 1; i >= 0; i--) {
                int32_t id = _bucket[bucket_id].chunk_ids[i];
                if (_chunk_status[id] & (_ck_syncing | _ck_loading)) continue;
                pthread_spin_lock(&_locks[id % pwc_locks]);
                if ((_chunk_status_vol[id] & (_ck_syncing | _ck_loading)) == 0) {
                    _chunk_status[id] |= _ck_writing | _ck_loading;
                } else {
                    pthread_spin_unlock(&_locks[id % pwc_locks]);
                    continue;
                }
                pthread_spin_unlock(&_locks[id % pwc_locks]);
                _bucket[bucket_id].chunk_ids[i] = _bucket[bucket_id].chunk_ids[--num_chunks];
                release_del(id);
            }

            gettimeofday(&end, NULL);
            if (end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1e-6 > 10.0) {
                lg_err("%d chunks still loading/syncing after 10 seconds, discarded", num_chunks);
                for (int32_t i = num_chunks - 1; i >= 0; i--) 
                    _chunk_status[_bucket[bucket_id].chunk_ids[i]] &= ~(_ck_writing | _ck_reading);
                break;
            }
        }

        pthread_spin_lock(&_bwc_lock);
        _deleted_bucket_ids[_num_deleted_buckets++] = bucket_id;
        pthread_spin_unlock(&_bwc_lock);
    } else if (_bucket[bucket_id].status & l0_bucket_t::_bk_writing) {
        for (;;) {
            if (_bucket[bucket_id].writing_chunk == (chunk_t *) -1) continue;

            pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
            chunk_t *tmp = *writing_chunk_ptr_vol;
            if (tmp != (chunk_t *) -1) {
                _bucket[bucket_id].writing_chunk = (chunk_t *) -1;
                pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                if (tmp != NULL) {
                    if (tmp->size) {
                        release_sync(tmp->id);
                    } else if (_bucket[bucket_id].num_chunks > 0) {
                        if (_bucket[bucket_id].chunk_ids[_bucket[bucket_id].num_chunks - 1] == tmp->id) {
                            _bucket[bucket_id].num_chunks--;
                            release_del(tmp->id);
                        } else lg_err("last empty chunk %d not in the list of bucket %d", tmp->id, bucket_id);
                    } else lg_err("empty chunk %d not in the list of bucket %d", tmp->id, bucket_id);
                }
                break;
            }
            pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        }

        _bucket[bucket_id].status = l0_bucket_t::_bk_ready;
        pthread_spin_lock(&_bwc_lock);
        if (_num_ready_buckets < bwc_max_buckets) {
            _ready_bucket_id[_num_ready_buckets++] = bucket_id;
        } else {
            lg_err("ready list full(%d), bucket %d discarded", _num_ready_buckets, bucket_id);
            _bucket[bucket_id].status = l0_bucket_t::_bk_reading;
        }
        pthread_spin_unlock(&_bwc_lock);

        if (_bucket[bucket_id].status == l0_bucket_t::_bk_ready) {
            __prefetch_for_reading(bucket_id);
        } else {
            bucket_finalize(bucket_id);
        }
    } else lg_err("input bucket %d status(%x) not in reading or writing", bucket_id, _bucket[bucket_id].status);
    
    lg_exit();
    return;
}

template <class logger_t> long bwc_manager_tmpl<logger_t>::num_ready() {
    pthread_spin_lock(&_bwc_lock);
    long ret = _num_ready_buckets;
    for (int32_t i = 0; i < _num_prefetched_bucket; i++) {
        if (_bucket[_prefetched_bucket_id[i]].status & l0_bucket_t::_bk_ready) ret++;
    }
    pthread_spin_unlock(&_bwc_lock);

    return ret;
}

template <class logger_t> chunk_t *bwc_manager_tmpl<logger_t>::fetch_for_write(long bucket_id) {
    lg_init();
    constexpr uint32_t _bk_all_status = l0_bucket_t::_bk_writing | l0_bucket_t::_bk_ready |
                                        l0_bucket_t::_bk_reading | l0_bucket_t::_bk_caching;

    volatile int32_t *_num_wp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wp);
    chunk_t *volatile *writing_chunk_ptr_vol = reinterpret_cast<chunk_t * volatile *>(&_bucket[bucket_id].writing_chunk);
    
    #if ENABLE_PROFILING
    ev_f4w.fetch_add(1);
    int first_try = 0x2;
    #endif

    for (;;) {
        #if ENABLE_PROFILING
        first_try >>= 1;
        #endif
        if ((_bucket[bucket_id].status & _bk_all_status) != l0_bucket_t::_bk_writing) {
            lg_err("wrong input bucket(%ld) status(%x)", bucket_id, _bucket[bucket_id].status);
            lg_exit();
            return NULL;
        }

        asm volatile("" ::: "memory");

        if (_bucket[bucket_id].writing_chunk == (chunk_t *) -1) continue;

        if (_bucket[bucket_id].writing_chunk == NULL) {
            pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
            if (*writing_chunk_ptr_vol == NULL) {
                _bucket[bucket_id].writing_chunk = (chunk_t *) -1;
                pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                chunk_t *ret = NULL;
                for (;;) {
                    if (_num_wp) {
                        pthread_spin_lock(&_bwc_wp_lock);
                        if (*_num_wp_ptr_vol) ret = _writing_prefetch_chunks[--_num_wp];
                        pthread_spin_unlock(&_bwc_wp_lock);
                    }
                    __prefetch_for_writing();
                    if (ret) break;
                    #if ENABLE_PROFILING
                    first_try >>= 1;
                    #endif
                }
                pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                _bucket[bucket_id].add_chunk(ret->id);
                pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                lg_exit();
                #if ENABLE_PROFILING
                if (first_try) ev_f4w_hit.fetch_add(1);
                #endif
                return ret;
            }
            pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);            
        }
        

        pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        chunk_t *writing_chunk = *writing_chunk_ptr_vol;
        if (writing_chunk != NULL && writing_chunk != (chunk_t *) -1) {
            chunk_t *ret = writing_chunk;
            _bucket[bucket_id].writing_chunk = (chunk_t *) -1;
            pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
            lg_exit();
            #if ENABLE_PROFILING
            if (first_try) ev_f4w_hit.fetch_add(1);
            #endif
            return ret;
        }
        pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    }
}

template <class logger_t> chunk_t *bwc_manager_tmpl<logger_t>::fetch_for_read(long bucket_id) {
    lg_init();
    volatile uint32_t *_chunk_status_vol = reinterpret_cast<volatile uint32_t*>(_chunk_status);
    volatile int32_t *num_chunks_ptr_vol = reinterpret_cast<volatile int32_t*>(&_bucket[bucket_id].num_chunks);
    volatile uint32_t *status_ptr_vol = reinterpret_cast<volatile uint32_t*>(&_bucket[bucket_id].status);
    volatile int32_t *chunk_ids_vol = reinterpret_cast<volatile int32_t*>(_bucket[bucket_id].chunk_ids);

    constexpr uint32_t _bk_all_status = l0_bucket_t::_bk_writing | l0_bucket_t::_bk_ready |
                                        l0_bucket_t::_bk_reading;

    #if ENABLE_PROFILING
    ev_f4r.fetch_add(1);
    int first_try = 0x2;
    #endif

    for (;;) {
        #if ENABLE_PROFILING
        first_try >>= 1;
        #endif
        uint32_t status = _bucket[bucket_id].status;
        uint32_t num_chunks = _bucket[bucket_id].num_chunks;

        if ((status & _bk_all_status) != l0_bucket_t::_bk_reading) {
            lg_err("wrong input bucket(%ld) status(%x)", bucket_id, status);
            return NULL;
        }

        if (num_chunks == 0) {
            lg_exit();
            return NULL;
        }

        int32_t ret = -1;
        for (int32_t i = num_chunks - 1; i >= 0 && i >= num_chunks - bwc_auto_prefetch_for_read_depth; i--) {
            int32_t id = _bucket[bucket_id].chunk_ids[i];
            if ((_chunk_status[id] & (_ck_reading | _ck_caching)) == _ck_caching) {
                pthread_spin_lock(&_locks[id % pwc_locks]);
                if ((_chunk_status_vol[id] & (_ck_reading | _ck_caching)) == _ck_caching) _chunk_status[id] |= _ck_writing;
                else {
                    pthread_spin_unlock(&_locks[id % pwc_locks]);
                    continue;
                }
                pthread_spin_unlock(&_locks[id % pwc_locks]);

                pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                if (i < *num_chunks_ptr_vol && status == *status_ptr_vol && id == chunk_ids_vol[i]) {
                    chunk_ids_vol[i] = chunk_ids_vol[--num_chunks_ptr_vol[0]];
                    ret = id;
                }
                pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                if (ret != -1) break;
            }
        }

        __prefetch_for_reading(bucket_id);
        
        if (ret != -1) {
            lg_exit();
            #if ENABLE_PROFILING
            if (first_try) ev_f4r_hit.fetch_add(1);
            #endif
            return &_cached_chunks[_chunk_status[ret] & _ck_cache_id_mask];
        }
    }
}

template <class logger_t> void bwc_manager_tmpl<logger_t>::write_done(chunk_t *chunk, long bucket_id) {
    lg_init();
    volatile int32_t *_num_wp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wp);

    if (chunk->size == Pool_hd_t::chunk_max_nvecs) {
        release_sync(chunk->id);

        chunk_t *new_chunk = NULL;
        if (_num_wp) {
            pthread_spin_lock(&_bwc_wp_lock);
            if (*_num_wp_ptr_vol) {
                new_chunk = _writing_prefetch_chunks[--_num_wp_ptr_vol[0]];
            }
            pthread_spin_unlock(&_bwc_wp_lock);
        }
        __prefetch_for_writing();
        if (new_chunk) {
            pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
            _bucket[bucket_id].add_chunk(new_chunk->id);
            pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
            _bucket[bucket_id].writing_chunk = new_chunk;
        } else _bucket[bucket_id].writing_chunk = NULL;
    } else {
        _bucket[bucket_id].writing_chunk = chunk;
    }
    lg_exit();
}

template <class logger_t> void bwc_manager_tmpl<logger_t>::read_done(chunk_t *chunk, long bucket_id) {
    lg_init();
    release_del(chunk->id);
    lg_exit();
}

template <class logger_t> long bwc_manager_tmpl<logger_t>::bucket_num_chunks(long bucket_id) {
    pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    long ret = _bucket[bucket_id].num_chunks;
    pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    return ret;
}

template <class logger_t> swc_manager_tmpl<logger_t>::swc_manager_tmpl(Pool_hd_t *p) : 
                          pwc_manager_tmpl<logger_t>(swc_default_loading_threads, swc_default_syncing_threads, swc_default_max_cached_chunks) {
    _num_ready = 0;
    _num_writing = 0;
    _num_rp = 0; 
    _num_rl = 0;
    _num_wp = 0;
    _num_wl = 0;
    _ready_chunks = (int32_t *) malloc(swc_max_ready_chunks * sizeof(int32_t));
    pthread_spin_init(&_swc_lock, PTHREAD_PROCESS_SHARED);

    this->_pool = p;
    this->set_dirname("sol");

    lg_info("manager initialized, (%d, %d) threads for I/O, #caching = %d", 
            _loading_threads, _syncing_threads, _max_cached_chunks);
}

template <class logger_t> swc_manager_tmpl<logger_t>::~swc_manager_tmpl() {
    constexpr chunk_status_t _ck_busy = _ck_writing | _ck_reading | _ck_loading;
    this->_syncing_pool.wait_work();
    this->_loading_pool.wait_work();

    pthread_spin_lock(&_swc_lock);
    if (_num_rl || _num_wl) lg_err("still %d(4r) + %d(4w) chunks loading?", _num_rl, _num_wl);
    int32_t num_ready = _num_ready;
    int32_t num_writing = _num_writing;
    int32_t num_rp = _num_rp;
    int32_t num_wp = _num_wp;
    _num_ready = 0;
    _num_writing = 0;
    _num_rp = 0;
    _num_wp = 0;
    _num_rl = swc_auto_prefetch_for_read;
    _num_wl = swc_auto_prefetch_for_write;
    pthread_spin_unlock(&_swc_lock);

    for (long i = 0; i < num_rp; i++) release_del(_reading_prefetch_chunks[i]->id);
    for (long i = 0; i < num_wp; i++) release_del(_writing_prefetch_chunks[i]->id);
    for (long i = 0; i < num_writing; i++) release_del(_writing_chunks[i]->id);
    for (long i = 0; i < num_ready; i++) {
        int32_t id = _ready_chunks[i];
        if (_chunk_status[id] & _ck_busy) lg_err("ready chunk %d status %x", id, _chunk_status[id]);
        pthread_spin_lock(&_locks[id % pwc_locks]);
        _chunk_status[_ready_chunks[i]] |= _ck_writing;
        pthread_spin_unlock(&_locks[id % pwc_locks]);
        release_del(id);
    }

    free(_ready_chunks);
    pthread_spin_destroy(&_swc_lock);

    this->_syncing_pool.wait_work();
    this->_loading_pool.wait_work();
    
    /// delete all files
    char chunk_filename[256];
    
    for (long i = 0; i < _num_chunks; i++) {
        #if MULTI_SSD
        snprintf(chunk_filename, sizeof(chunk_filename), "%s/%s/%s%06lx", this->_dir, hw::ssd_name(i), this->_pfx, i);
        #else
        snprintf(chunk_filename, sizeof(chunk_filename), "%s%06lx", this->_prefix, i);
        #endif
        remove(chunk_filename);
    }
}

template <class logger_t> void swc_manager_tmpl<logger_t>::__prefetch_for_writing() {
    lg_init();
    volatile int32_t *_num_wp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wp);
    volatile int32_t *_num_wl_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wl);

    if (_num_writing + _num_wl + _num_wp >= swc_max_writing_chunks) { 
        lg_exit(); 
        return; 
    }
    if (_num_wl + _num_wp >= swc_auto_prefetch_for_write) { 
        lg_exit(); 
        return; 
    }

    pthread_spin_lock(&_swc_lock);
    int to_prefetch = swc_auto_prefetch_for_write - _num_wl_ptr_vol[0] - _num_wp_ptr_vol[0];
    if (to_prefetch > swc_max_writing_chunks - _num_writing + _num_wl + _num_wp)
        to_prefetch = swc_max_writing_chunks - _num_writing + _num_wl + _num_wp;
    if (to_prefetch <= 0) {
        pthread_spin_unlock(&_swc_lock);
        lg_exit();
        return;
    }
    _num_wl += to_prefetch;
    pthread_spin_unlock(&_swc_lock);

    int32_t num_fail = 0;
    for (int32_t i = 0; i < to_prefetch; i++) {
        int32_t id = create_chunk();
        pthread_spin_lock(&_locks[id % pwc_locks]);
        if (_chunk_status[id]) {
            num_fail++;
        } else {
            _chunk_status[id] |= _ck_loading | _ck_writing;
            this->_loading_pool.push([=]() {
                __load_chunk(id);
                pthread_spin_lock(&_swc_lock);
                _writing_prefetch_chunks[_num_wp++] = &_cached_chunks[_chunk_status[id] & _ck_cache_id_mask];
                _num_wl--;
                pthread_spin_unlock(&_swc_lock);
            });
            _num_loading_chunks++;
        }
        pthread_spin_unlock(&_locks[id % pwc_locks]);
    }

    if (num_fail) {
        lg_err("%d of %d new chunk already in use?", num_fail, to_prefetch);
        pthread_spin_lock(&_swc_lock);
        _num_wl -= num_fail;
        pthread_spin_unlock(&_swc_lock);
    }
    lg_exit();
}

template <class logger_t> void swc_manager_tmpl<logger_t>::__prefetch_for_reading() {
    lg_init();
    volatile int32_t *_num_rp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_rp);
    volatile int32_t *_num_rl_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_rl);

    if (_num_ready == 0 || _num_rp + _num_rl >= swc_auto_prefetch_for_read) {
        lg_exit();
        return;
    }

    constexpr chunk_status_t _ck_busy = _ck_reading | _ck_writing | _ck_loading;
    constexpr int32_t max_search = 64;
    int32_t num_possible_ids = 0;
    int32_t possible_ids[max_search];
    do {
        int exp_prefetch = swc_auto_prefetch_for_read - _num_rp - _num_rl;
        if (2 * exp_prefetch > _num_ready || exp_prefetch <= 0) break;
        for (int32_t i = _num_ready - 1; i >= 0 && i >= _num_ready - max_search; i--) {
            int id = _ready_chunks[i];
            if (id >= _num_chunks || id < 0) continue;
            if ((_chunk_status[id] & (_ck_caching | _ck_busy )) == _ck_caching) {
                possible_ids[num_possible_ids++] = i;
                if (num_possible_ids >= 2 * exp_prefetch || num_possible_ids >= max_search) break;
            }
        }
    } while (0);

    pthread_spin_lock(&_swc_lock);
    int to_prefetch = swc_auto_prefetch_for_read - _num_rp_ptr_vol[0] - _num_rl_ptr_vol[0];
    if (to_prefetch > _num_ready) to_prefetch = _num_ready;
    if (to_prefetch <= 0) {
        pthread_spin_unlock(&_swc_lock);
        lg_exit();
        return;
    }
    _num_rl += to_prefetch;

    int32_t to_prefetch_ids[swc_auto_prefetch_for_read];
    do {
        int32_t i = 0;
        while (num_possible_ids) {
            int32_t ptr = possible_ids[--num_possible_ids];
            if (ptr >= _num_ready) continue;
            if ((_chunk_status[_ready_chunks[ptr]] & (_ck_busy | _ck_caching)) == _ck_caching) {
                to_prefetch_ids[i++] = _ready_chunks[ptr];
                _ready_chunks[ptr] = _ready_chunks[--_num_ready];
                if (i >= to_prefetch) break;
            }
        }

        _num_ready -= to_prefetch - i;
        for (int32_t j = 0; j < to_prefetch - i; j++) {
            to_prefetch_ids[i + j] = _ready_chunks[_num_ready + j];
        }
    } while (0);
    pthread_spin_unlock(&_swc_lock);

    int32_t num_fail = 0;
    for (int32_t i = 0; i < to_prefetch; i++) {
        int32_t id = to_prefetch_ids[i];
        if (id < 0 || id >= _num_chunks) {
            lg_err("invalid chunk id %d, %d/%d", id, i, to_prefetch);
            num_fail++;
            continue;
        }
        pthread_spin_lock(&_locks[id % pwc_locks]);
        if (_chunk_status[id] & _ck_busy) {
            lg_err("chunk(%d) in ready list is busy(%x)", id, _chunk_status[id]);
            num_fail++;
        } else if (_chunk_status[id] == _ck_size_mask) {
            lg_err("chunk(%d) in ready list is deleted", id);
            num_fail++;
        } else if (_chunk_status[id] & _ck_caching) {
            _chunk_status[id] |= _ck_writing;
            pthread_spin_unlock(&_locks[id % pwc_locks]);

            pthread_spin_lock(&_swc_lock);
            _reading_prefetch_chunks[_num_rp++] = &_cached_chunks[_chunk_status[id] & _ck_cache_id_mask];
            _num_rl--;
            pthread_spin_unlock(&_swc_lock);
            continue;
        } else {
            _chunk_status[id] |= _ck_loading | _ck_writing;
            this->_loading_pool.push([=]() {
                __load_chunk(id);
                pthread_spin_lock(&_swc_lock);
                _reading_prefetch_chunks[_num_rp++] = &_cached_chunks[_chunk_status[id] & _ck_cache_id_mask];
                _num_rl--;
                pthread_spin_unlock(&_swc_lock);
            });
            _num_loading_chunks++;
        }
        pthread_spin_unlock(&_locks[id % pwc_locks]);
    }

    if (num_fail) {
        pthread_spin_lock(&_swc_lock);
        _num_rl -= num_fail;
        pthread_spin_unlock(&_swc_lock);
    }

    lg_exit();
}

template <class logger_t> void swc_manager_tmpl<logger_t>::chunk_finalize(chunk_t *chunk) {
    lg_init();
    volatile int32_t *_num_rp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_rp);
    volatile int32_t *_num_rl_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_rl);
    volatile int32_t *_num_ready_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_ready);

    constexpr chunk_status_t _ck_busy = _ck_writing | _ck_reading | _ck_loading;

    if (_num_rp + _num_rl < swc_auto_prefetch_for_read) {
        pthread_spin_lock(&_swc_lock);
        if (_num_rp_ptr_vol[0] + _num_rl_ptr_vol[0] < swc_auto_prefetch_for_read) {
            _reading_prefetch_chunks[_num_rp++] = chunk;
            pthread_spin_unlock(&_swc_lock);
            lg_exit();
            return;
        } 
        pthread_spin_unlock(&_swc_lock);
    }

    int id = chunk->id;
    release_sync(chunk->id);

    if (_num_ready < swc_max_ready_chunks) {
        pthread_spin_lock(&_swc_lock);
        if (_num_ready_ptr_vol[0] < swc_max_ready_chunks) {
            _ready_chunks[_num_ready++] = id;
            pthread_spin_unlock(&_swc_lock);
            lg_exit();
            return;
        }
        pthread_spin_unlock(&_swc_lock);
    }
    
    pthread_spin_lock(&_locks[id % pwc_locks]);
    _chunk_status[id] |= _ck_writing;
    pthread_spin_lock(&_locks[id % pwc_locks]);
    release_del(id);
    lg_err("ready list full(%d), new chunk discarded", _num_ready);
    lg_exit();
    return;
}

template <class logger_t> chunk_t *swc_manager_tmpl<logger_t>::fetch_for_write() {
    lg_init();
    volatile int32_t *_num_wp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wp);
    volatile int32_t *_num_writing_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_writing);
    
    chunk_t *ret = NULL;

    #if ENABLE_PROFILING
    ev_f4w.fetch_add(1);
    int first_try = 0x2;
    #endif

    while (ret == NULL) {
        #if ENABLE_PROFILING
        first_try >>= 1;
        #endif
        if (_num_writing || _num_wp) {
            pthread_spin_lock(&_swc_lock);
            if (*_num_writing_ptr_vol) ret = _writing_chunks[--_num_writing];
            else if (*_num_wp_ptr_vol) ret = _writing_prefetch_chunks[--_num_wp];
            pthread_spin_unlock(&_swc_lock);
        }

        __prefetch_for_writing();
    }

    #if ENABLE_PROFILING
    if (first_try) ev_f4w_hit.fetch_add(1);
    #endif

    lg_exit();
    return ret;
}

template <class logger_t> chunk_t *swc_manager_tmpl<logger_t>::fetch_for_read() {
    lg_init();
    volatile int32_t *_num_rp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_rp);

    #if ENABLE_PROFILING
    ev_f4r.fetch_add(1);
    int first_try = 0x2;
    #endif

    chunk_t *ret = NULL;

    while (ret == NULL) {
        #if ENABLE_PROFILING
        first_try >>= 1;
        #endif
        if (_num_rp) {
            pthread_spin_lock(&_swc_lock);
            if (*_num_rp_ptr_vol) ret = _reading_prefetch_chunks[--_num_rp];
            pthread_spin_unlock(&_swc_lock);
        }

        if (_num_ready) __prefetch_for_reading();
        if (_num_rl == 0 && _num_rp == 0 && _num_ready == 0) break;
    }

    #if ENABLE_PROFILING
    if (first_try) ev_f4r_hit.fetch_add(1);
    #endif

    lg_exit();
    return ret;
}

template <class logger_t> void swc_manager_tmpl<logger_t>::write_done(chunk_t *chunk) {
    lg_init();
    if (chunk->size < Pool_hd_t::chunk_max_nvecs || !swc_auto_finalize) {
        pthread_spin_lock(&_swc_lock);
        if (_num_writing < swc_max_writing_chunks) {
            _writing_chunks[_num_writing++] = chunk;
            pthread_spin_unlock(&_swc_lock);
            lg_exit();
            return;
        }
        pthread_spin_unlock(&_swc_lock);
    }

    chunk_finalize(chunk);
    lg_exit();
}

template <class logger_t> void swc_manager_tmpl<logger_t>::read_done(chunk_t *chunk) {
    lg_init();
    release_del(chunk->id);
    lg_exit();
}

template <class logger_t> long swc_manager_tmpl<logger_t>::num_ready() {
    pthread_spin_lock(&_swc_lock);
    long ret = _num_ready + _num_rp + _num_rl;
    pthread_spin_unlock(&_swc_lock);
    return ret;
}

template <class logger_t> long swc_manager_tmpl<logger_t>::ready_nvecs_estimate() {
    long ret = 0;

    pthread_spin_lock(&_swc_lock);
    for (int i = 0; i < _num_rp; i++) ret += _reading_prefetch_chunks[i]->size;
    for (int i = 0; i < _num_ready; i++) {
        int size = 0;
        int id = _ready_chunks[i];
        if (id >= 0 && id < _num_chunks) size = chunk_size(id);
        if (size <= Pool_hd_t::chunk_max_nvecs) ret += size;
    }
    pthread_spin_unlock(&_swc_lock);

    return ret;
}

template <class logger_t> long swc_manager_tmpl<logger_t>::num_using() {
    pthread_spin_lock(&_swc_lock);
    long ret = _num_chunks - _num_deleted_ids - _num_writing;
    pthread_spin_unlock(&_swc_lock);

    return ret;
}

#if 0
template <class logger_t> long swc_manager_tmpl<logger_t>::finalize_all_writing() {
    lg_init();
    int to_finalize_malloc_size = _num_writing + swc_auto_prefetch_for_write;
    int num_to_finalize = 0;
    chunk_t **to_finalize = (chunk_t **) malloc(to_finalize_malloc_size * sizeof(chunk_t *));
    pthread_spin_lock(&_swc_lock);
    if (_num_writing + swc_auto_prefetch_for_write > to_finalize_malloc_size) {
        lg_err("# writing chunks increased while calling finalize_all_writing, no chunk finalized");
        pthread_spin_unlock(&_swc_lock);
        free(to_finalize);
        lg_exit();
        return -1;
    }
    for (int i = 0; i < _num_writing; i++) to_finalize[num_to_finalize++] = _writing_chunks[i];
    _num_writing = 0;
    for (int i = 0; i < _num_wp; i++) to_finalize[num_to_finalize++] = _writing_prefetch_chunks[i];
    _num_wp = 0;
    pthread_spin_unlock(&_swc_lock);

    for (int i = 0; i < num_to_finalize; i++) {
        if (to_finalize[i]->size == 0) release_del(to_finalize[i]->id);
        else chunk_finalize(to_finalize[i]);
    }

    lg_exit();
    return 0;
}
#endif

#if ENABLE_PROFILING
template struct bwc_manager_tmpl<bwc_logger_t>;
template struct swc_manager_tmpl<swc_logger_t>;
#else
template struct bwc_manager_tmpl<int>;
template struct swc_manager_tmpl<int>;
#endif
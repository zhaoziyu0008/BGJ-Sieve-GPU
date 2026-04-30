/// @file pool_hd.cpp
/// @brief Implementation of host-only pool operations

#include "../include/pool_hd.h"
#include "../include/bgj_hd.h"
#include "../include/utils.h"
#include "../include/vec.h"

#include <omp.h>


// this is used only for basis consistency check
// we choose Daniel J. Bernstein's 33 hash now
static uint64_t _basis_hash(Lattice_QP *L) {
    uint64_t h = 5381;
    h = (h << 5) + h + L->NumRows();
    h = (h << 5) + h + L->NumCols();
    for (long i = 0; i < L->NumRows(); i++) {
        const char *b_hi_str = (char *) L->get_b().hi[i];
        const char *b_lo_str = (char *) L->get_b().lo[i]; 
        for (long j = 0; j < L->NumCols() * 8; j++) {
            h = (h << 5) + h + b_hi_str[j];
            h = (h << 5) + h + b_lo_str[j];
        }
    }

    return h;
}

#if ENABLE_PROFILING
generic_logger_t::generic_logger_t() {
    this->log_thread = NULL;
    this->start();
}

generic_logger_t::~generic_logger_t() {
    this->stop();
    /// if (_log_out != stdout && _log_out != stderr) fclose(_log_out);
    /// if (_log_err != stdout && _log_err != stderr) fclose(_log_err);
}
#endif

///////////////// pool with cache manager /////////////////

#if ENABLE_PROFILING
template struct pwc_manager_tmpl<pwc_logger_t>;
#else
template struct pwc_manager_tmpl<int>;
#endif

///////////////// pool operations /////////////////

Pool_hd_t::Pool_hd_t() {
    _start_ck_allocator();
    #if ENABLE_PROFILING
    this->logger = new pool_logger_t();
    #endif
    this->pwc_manager = new pwc_manager_t();
    this->pwc_manager->set_pool(this);
    this->uid_table = new UidTable();
    this->set_num_threads(1);
    this->ESD = 0;
    this->_boost_data = new boost_data_t();

    return;
}

Pool_hd_t::Pool_hd_t(Lattice_QP *L) {
    _start_ck_allocator();
    #if ENABLE_PROFILING
    this->logger = new pool_logger_t();
    #endif
    this->pwc_manager = new pwc_manager_t();
    this->set_basis(L);
    this->pwc_manager->set_pool(this);
    this->uid_table = new UidTable();
    this->set_num_threads(1);
    this->ESD = 0;
    this->_boost_data = new boost_data_t();

    return;
}

Pool_hd_t::~Pool_hd_t() {
    delete this->pwc_manager;
    delete this->uid_table;
    delete this->_boost_data;
    if (_b_local) FREE_MAT(_b_local);
    if (_b_dual) FREE_VEC(_b_dual);
    #if ENABLE_PROFILING
    delete this->logger;
    #endif

    return;
}

int Pool_hd_t::set_basis(Lattice_QP *L) {
    if (L->get_gso_status() != GSO_COMPUTED_QP) L->compute_gso_QP();
    this->basis = L;
    this->basis_hash = _basis_hash(L);

    return 0;
}

int Pool_hd_t::set_num_threads(long num_threads) {
    _num_threads = num_threads;
    uid_table->set_num_threads(num_threads);
    
    return 0;
}

int Pool_hd_t::set_sieving_context(long ind_l, long ind_r) {
    // I don't want to check the validity of l and r here
    this->CSD = ind_r - ind_l;
    this->index_l = ind_l;
    this->index_r = ind_r;
    
    this->uid_table->reset_hash_function(this->CSD);    
    this->_update_b_local();

    return 0;
}

int Pool_hd_t::set_boost_depth(long esd) {
    if (esd < 0 || esd > boost_data_t::max_boost_dim || esd > this->index_l) {
        fprintf(stdout, "[Warning] Pool_hd_t::set_boost_depth: esd out of range, fixed\n");
        if (esd < 0) esd = 0;
        if (esd > boost_data_t::max_boost_dim) esd = boost_data_t::max_boost_dim;
        if (esd > this->index_l) esd = this->index_l;
    }
    this->ESD = esd;
    this->_update_boost_data();

    return 0;
}

int Pool_hd_t::_update_b_local(float ratio) {
    if (_b_local) FREE_MAT(_b_local);
    if (_b_dual) FREE_VEC(_b_dual);
    _b_local = (float **) NEW_MAT(CSD, vec_nbytes, sizeof(float));
    _b_dual = (int8_t *) NEW_VEC(CSD * vec_nbytes, sizeof(int8_t));
    double **tmp1 = (double **) NEW_MAT(CSD, CSD, sizeof(double));
    double **tmp2 = (double **) NEW_MAT(CSD, CSD, sizeof(double));
    if (!_b_local || !_b_dual || !tmp1 || !tmp2) {
        fprintf(stdout, "[Error] Pool_hd_t::_update_b_local: memory allocation failed\n");
        return -1;
    }

    this->_ratio = -1.0;
    double **mu = this->basis->get_miu().hi;
    double *B = this->basis->get_B().hi;
    for (long j = 0; j < this->CSD; j++){
        double x = sqrt(B[j + this->index_l]);
        for (long i = j; i < this->CSD; i++){
            tmp1[j][i] = mu[i + this->index_l][j + this->index_l] * x;
        }
        this->_ratio = x > this->_ratio ? x : this->_ratio;
    }

    for (long i = 0; i < this->CSD; i++) {
        tmp2[i][i] = 1.0 / tmp1[i][i];
        mul_avx2(tmp1[i], tmp2[i][i], this->CSD);
    }
    for (long i = this->CSD - 1; i > 0; i--) {
        for (long j = 0; j < i; j++) {
            red_avx2(tmp2[j], tmp2[i], tmp1[j][i], this->CSD);
        }
    }

    for (long i = 0; i < this->CSD; i++) {
        mul_avx2(tmp1[i], 1.0 / tmp2[i][i], this->CSD);
    }
    for (long i = 0; i < this->CSD; i++) {
        for (long j = 0; j < i; j++) {
            double x = tmp1[i][j];
            tmp1[i][j] = tmp1[j][i];
            tmp1[j][i] = x;
        }
    }
    for (long i = 1; i < this->CSD; i++) {
        double x = 1.0 / tmp2[i][i];
        for (long j = 0; j < i; j++) {
            double y = round(tmp2[j][i] * x);
            if (fabs(y) > 1e-3) {
                red_avx2(tmp2[j], tmp2[i], y, this->CSD);
                red_avx2(tmp1[i], tmp1[j], -y, this->CSD);
            }
        }
    }

    this->_ratio = 254.0 / this->_ratio * (CSD < 80 ? pow(1.01, CSD - 80) : 1.0);
    if (ratio) this->_ratio = ratio;

    double mdd = 0.0;
    for (long i = 0; i < this->CSD; i++) {
        mul_avx2(tmp1[i], this->_ratio, this->CSD);
        mul_avx2(tmp2[i], 1.0 / this->_ratio, CSD);
        for (long j = 0; j < this->CSD; j++) _b_local[i][j] = tmp1[i][j];
        mdd = tmp2[i][i] > mdd ? tmp2[i][i] : mdd;
    }
    this->_dhalf = floor(127.0 / mdd);
    this->_dshift = 31 - __builtin_clz(this->_dhalf);
    this->_dhalf = 1 << (this->_dshift-1);
    if (this->_dhalf < 1) {
        fprintf(stdout, "[Error] Pool_hd_t::_update_b_local: _dhalf < 1, the basis too coarse?");
    }
    mdd = ((this->_dhalf << 1) + 0.0);

    for (long i = 0; i < this->CSD; i++) {
        mul_avx2(tmp2[i], mdd, this->CSD);
        for (long j = 0; j < this->CSD; j++) {
            _b_dual[i * vec_nbytes + j] = round(tmp2[i][j]);
        }
    }

    FREE_MAT(tmp1);
    FREE_MAT(tmp2);

    this->_gh2 = this->basis->gh(this->index_l, this->index_r);
    this->_gh2 *= this->_gh2;

    return 0;
}

float **Pool_hd_t::_compute_b_local(long ind_l, long ind_r) {
    long _tmp_index_l = this->index_l;
    long _tmp_index_r = this->index_r;
    long _tmp_CSD = this->CSD;
    float **_tmp_b_local = this->_b_local;
    int8_t *_tmp_b_dual = this->_b_dual;
    float _tmp_ratio = this->_ratio;
    int32_t _tmp_dhalf = this->_dhalf;
    int32_t _tmp_dshift = this->_dshift;
    double _tmp_gh2 = this->_gh2;
    this->_b_local = NULL;
    this->_b_dual = NULL;
    
    this->index_l = ind_l;
    this->index_r = ind_r;
    this->CSD = ind_r - ind_l;
    this->_update_b_local(_tmp_ratio);

    FREE_VEC((void *) _b_dual);
    float **ret = _b_local;

    this->index_l = _tmp_index_l;
    this->index_r = _tmp_index_r;
    this->CSD = _tmp_CSD;
    this->_gh2 = _tmp_gh2;
    this->_b_local = _tmp_b_local;
    this->_b_dual = _tmp_b_dual;
    this->_ratio = _tmp_ratio;
    this->_dhalf = _tmp_dhalf;
    this->_dshift = _tmp_dshift;
    
    if ((ind_r == index_r) && (ind_l <= index_l)) {
        bool pass = true;
        long b = this->index_l - ind_l;
        for (long i = 0; i < this->CSD; i++) {
            for (long j = 0; j <= i; j++) {
                if (*((uint32_t *)(&ret[i+b][j+b])) != *((uint32_t *)(&this->_b_local[i][j]))) {
                    pass = false;
                    break;
                }
            }
        }
        if (!pass) printf("[Error] Pool_hd_t::_compute_b_local: consistency check fail, "
                          "unexpected things may happen, ignored.\n");
    }

    return ret;
}

int Pool_hd_t::_update_boost_data() {
    Lattice_QP *L_full = this->basis->b_loc_QP(index_l - ESD, index_r);
    Lattice_QP *L_loc_target = new Lattice_QP(CSD, CSD);
    for (long i = 0; i < CSD; i++) {
        for (long j = 0; j < CSD; j++) L_loc_target->get_b().hi[i][j] = this->_b_local[i][j] / this->_ratio;
    }
    L_full->trans_to(ESD, ESD + CSD, L_loc_target);
    for (long i = ESD; i < ESD + CSD; i++) {
        for (long j = ESD - 1; j >= 0; j--) {
            long q = round(L_full->get_b().hi[i][j] / L_full->get_b().hi[j][j]);
            red(L_full->get_b().hi[i], L_full->get_b().lo[i], L_full->get_b().hi[j], L_full->get_b().lo[j], NTL::quad_float(q), ESD);
        }
    }
    delete L_loc_target;

    for (long i = 0; i < ESD; i++) {
        for (long j = 0; j <= i; j++) {
            _boost_data->evec[i * ESD + j] = L_full->get_b().hi[i][j] * _ratio;
        }
    }
    for (long i = ESD; i < ESD + CSD; i++) {
        for (long j = 0; j < ESD; j++) {
            _boost_data->evec[i * ESD + j] = L_full->get_b().hi[i][j] * _ratio;
        }
    }

    for (long i = 0; i < ESD; i++) {
        double x = 1.0 / L_full->gh(i, L_full->NumRows());
        x = x * x * this->_gh2;
        _boost_data->igh[i] = x;
        _boost_data->inorm[i] = 1.0 / _boost_data->evec[i * ESD + i];
    }

    delete L_full;

    return 0;
}

int Pool_hd_t::_sampling(int8_t *dst_vec, uint16_t *dst_score, int32_t *dst_norm, uint64_t *dst_u, DGS1d *R) {
    float res[vec_nbytes] __attribute__ ((aligned (32)));
    float ext[boost_data_t::max_boost_dim] __attribute__ ((aligned (32)));
    int coeff[vec_nbytes];
    double sigma2 = _b_local[CSD/2][CSD/2] * _b_local[CSD/2][CSD/2] / 64.0;
    long cnt = 0;
    do {
        set_zero_avx2(res, CSD);
        set_zero_avx2(ext, ESD);
        for (long i = CSD - 1; i >= 0; i--) {
            do {
                // -O3 should optimize it ?
                coeff[i] = R->discrete_gaussian(-res[i]/_b_local[i][i], 
                                                sigma2/(_b_local[i][i] * _b_local[i][i]) + 0.1);
            } while (fabsf(res[i] + coeff[i] * _b_local[i][i]) > 127.4f);
            red_avx2(res, _b_local[i], -coeff[i], CSD);
            if (ESD) {
                for (int j = 0; j < ESD; j++) ext[j] += coeff[i] * _boost_data->evec[(i + ESD) * ESD + j];
            }
        }
        *dst_u = 0;
        for (long i = 0; i < CSD; i++){
            *dst_u += coeff[i] * uid_table->coeff(i);
        }
        cnt++;
        if (cnt > 30) {
            fprintf(stderr, "[Error] Pool_hd_t::_sampling: sampling always get collision, aborted.\n");
            return -1;
        }
    } while (!uid_table->insert(*dst_u));
    for (long i = 0; i < CSD; i++) dst_vec[i] = roundf(res[i]);
    *dst_norm = roundf(0.5 * dot_avx2(res, res, CSD));
    if (ESD) {
        float sc2 = 2.0 * dst_norm[0];
        float n2 = sc2;
        for (long i = ESD - 1; i >= 0; i--) {
            float c = roundf(ext[i] * _boost_data->inorm[i]);
            for (long j = 0; j <= i; j++) ext[j] -= c * _boost_data->evec[i * ESD + j];
            n2 += ext[i] * ext[i];
            sc2 = sc2 < n2 * _boost_data->igh[i] ? sc2 : n2 * _boost_data->igh[i];
        }
        int hsc2i = 0.25 * sc2;
        dst_score[0] = hsc2i > 65535 ? 65535 : hsc2i;
    } else *dst_score = (dst_norm[0] >> 1) > 65535 ? 65535 : (dst_norm[0] >> 1);
    return 0;
}

struct sampling_iterator_t {
    using chunk_status_t = pwc_manager_t::chunk_status_t;

    static constexpr long min_empty_per_load = 512;
    static constexpr chunk_status_t _ck_busy = pwc_manager_t::_ck_loading | pwc_manager_t::_ck_syncing | 
                                               pwc_manager_t::_ck_reading | pwc_manager_t::_ck_writing;
    
    sampling_iterator_t(Pool_hd_t *pool, long to_add) {
        this->num_threads = pool->_num_threads;
        this->pwc_manager = pool->pwc_manager;
        this->prefetch_ahead = 4 * pool->_num_threads;

        this->init(to_add);
    }

    ~sampling_iterator_t() {
        free(task_list);
        pthread_spin_destroy(&task_lock);
    }

    void init(long to_add) {
        num_task = 0;
        task_list = (uint64_t *) malloc(sizeof(uint64_t) * (to_add / min_empty_per_load + 1));
        
        for (long i = 0;; i++) {
            long id = i;
            if (i >= pwc_manager->num_chunks()) id = pwc_manager->create_chunk();
            
            if (pwc_manager->chunk_status(id) & _ck_busy) continue;
            long chunk_empty = Pool_hd_t::chunk_max_nvecs - pwc_manager->chunk_size(id);

            if (chunk_empty >= to_add) {
                task_list[num_task++] = (id << 32) | to_add;
                break;
            } else if (chunk_empty >= min_empty_per_load) {
                task_list[num_task++] = (id << 32) | chunk_empty;
                to_add -= chunk_empty;
            } else continue;
        }

        pthread_spin_init(&task_lock, PTHREAD_PROCESS_SHARED);

        for (long i = 0; i < prefetch_ahead && i < num_task; i++) {
            pwc_manager->prefetch(task_list[num_task - 1 - i] >> 32);
        }
    }

    int pop(chunk_t *&chunk, int32_t &to_fill) {
        pthread_spin_lock(&task_lock);
        uint64_t task;
        if (num_task > 0) {
            task = task_list[--num_task];
            if (num_task >= prefetch_ahead) pwc_manager->prefetch(task_list[num_task - prefetch_ahead] >> 32);
        } else {
            pthread_spin_unlock(&task_lock);
            return -1;
        }
        pthread_spin_unlock(&task_lock);

        chunk = pwc_manager->fetch(task >> 32);
        to_fill = task & 0xffffffff;

        return 0;
    }

    pwc_manager_t *pwc_manager;
    
    long num_task;
    uint64_t *task_list;
    pthread_spinlock_t task_lock;

    long num_threads;
    long prefetch_ahead;
};

int Pool_hd_t::sampling(long N) {
    lg_init();
    long to_add = N - pwc_manager->num_vec();

    if (to_add <= 0) {
        lg_info("nothing to do");
        lg_exit();
        return 0;
    }

    sampling_iterator_t iter(this, to_add);    
    
    bool sampling_failed = false;
    #pragma omp parallel for num_threads(_num_threads) reduction(+:score_stat[:65536])
    for (long thread = 0; thread < _num_threads; thread++) {
        DGS1d R(thread + rand());
        int32_t to_fill;
        chunk_t *chunk;

        for (;;) {
            if (sampling_failed) break;
            if (iter.pop(chunk, to_fill) == -1) break;
            
            chunk->size += to_fill;
            for (long i = 0; i < chunk_max_nvecs; i++) {
                if (chunk->score[i]) continue;

                if (_sampling(chunk->vec + CSD * i, chunk->score + i, chunk->norm + i, chunk->u + i, &R) == -1) {
                    sampling_failed = true;
                    chunk->size -= to_fill;
                    break;
                } else score_stat[chunk->score[i]]++;

                if (--to_fill <= 0) break;
            }

            pwc_manager->release_sync(chunk->id);
        }
    }

    if (sampling_failed) {
        lg_err("failed, %ld / %ld vectors in the pool now", pwc_manager->num_vec(), N);
        lg_exit();
        return -1;
    }
    
    lg_exit();
    return 0;
}

struct shrink_iterator_t {
    static constexpr int pfch_ahead = 8;
    static constexpr int dst_need_filt = 1;
    static constexpr int src_need_filt = 2;
    static constexpr uint8_t chunk_filt = 0x10;
    static constexpr uint8_t chunk_busy = 0x20;
    static constexpr uint8_t chunk_zero = 0x40;
    static constexpr uint8_t chunk_full = 0x80;

    shrink_iterator_t(pwc_manager_t *pwc_manager, long num_chunk) {
        this->pwc = pwc_manager;
        this->max_full = -1;
        this->min_zero = num_chunk;
        this->task_status = (uint8_t *) malloc(num_chunk + 2);
        this->task_chunks = (chunk_t **) calloc(num_chunk, sizeof(chunk_t *));
        memset(task_status, 0, num_chunk + 2);
        task_status[0] = chunk_full;
        task_status[num_chunk + 1] = chunk_zero;
        task_status++;
        pthread_spin_init(&lock, PTHREAD_PROCESS_SHARED);
        for (int i = 0; i < pfch_ahead; i++) {
            if (i < num_chunk) pwc->prefetch(i);
        }
        for (int i = num_chunk - 1; i >= num_chunk - pfch_ahead; i--) {
            if (i >= 0) pwc->prefetch(i);
        }
    }

    ~shrink_iterator_t() {
        free(task_status - 1);
        free(task_chunks);
        pthread_spin_destroy(&lock);
    }

    int pop(chunk_t *&dst, chunk_t *&src) {
        int ret = 0, dst_id = -1, src_id = -1;
        dst = NULL;
        src = NULL;

        pthread_spin_lock(&lock);
        for (int i = max_full + 1; i < min_zero; i++) {
            if (task_status[i] & (chunk_full | chunk_busy)) continue;
            dst_id = i;
            task_status[i] |= chunk_busy;
            if ((task_status[i] & chunk_filt) == 0) ret |= dst_need_filt;
            break;
        }
        if (dst_id == -1) {
            pthread_spin_unlock(&lock);
            return -1;
        }
        for (int i = min_zero - 1; i > dst_id; i--) {
            if (task_status[i] & (chunk_zero | chunk_busy)) continue;
            src_id = i;
            task_status[i] |= chunk_busy;
            if ((task_status[i] & chunk_filt) == 0) ret |= src_need_filt;
            break;
        }
        if (src_id == -1 && (task_status[dst_id] & chunk_filt)) {
            task_status[dst_id] &= ~chunk_busy;
            if (max_full + 2 == min_zero && dst_id == max_full + 1) pwc->release_sync(dst_id);
            pthread_spin_unlock(&lock);
            return -1;
        }
        pthread_spin_unlock(&lock);
        
        if (task_chunks[dst_id] == NULL) { 
            dst = pwc->fetch(dst_id); 
            if (dst_id + pfch_ahead < min_zero) pwc->prefetch(dst_id + pfch_ahead);
        } else { dst = task_chunks[dst_id]; task_chunks[dst_id] = NULL; }
        if (src_id != -1) {
            if (task_chunks[src_id] == NULL) {
                src = pwc->fetch(src_id);
                if (src_id - pfch_ahead > max_full) pwc->prefetch(src_id - pfch_ahead);
            } else { src = task_chunks[src_id]; task_chunks[src_id] = NULL; }
        }        
        
        return ret;
    }

    int release(chunk_t *dst, chunk_t *src) {
        task_chunks[dst->id] = dst;
        if (src) task_chunks[src->id] = src;
        
        pthread_spin_lock(&lock);
        int old_max_full = max_full;
        int old_min_zero = min_zero;
        
        task_status[dst->id] = chunk_filt;
        if (dst->size == Pool_hd_t::chunk_max_nvecs) {
            task_status[dst->id] |= chunk_full;
            for (int i = max_full + 1; i < min_zero; i++) {
                if (!(task_status[i] & chunk_full)) break;
                max_full = i;
            }
        }
        if (src) {
            task_status[src->id] = chunk_filt;
            if (src->size == 0) {
                task_status[src->id] |= chunk_zero;
                for (int i = min_zero - 1; i > max_full; i--) {
                    if (!(task_status[i] & chunk_zero)) break;
                    min_zero = i;
                }
            }
        }
        int new_min_zero = min_zero;
        int new_max_full = max_full;
        pthread_spin_unlock(&lock);

        for (int i = old_max_full + 1; i <= new_max_full; i++) { pwc->release_sync(i); task_chunks[i] = NULL; }
        for (int i = new_min_zero; i < old_min_zero; i++) { pwc->release_del(i); task_chunks[i] = NULL; }

        return 0;
    }

    int res_chunk() {
        if (max_full + 2 < min_zero) {
            printf("[Error] shrink_iterator_t::res_chunk: called before all tasks finished, ignored\n");
        }
        return min_zero;
    }

    pthread_spinlock_t lock;

    int max_full, min_zero;
    uint8_t *task_status;
    chunk_t **task_chunks;
    pwc_manager_t *pwc;
};

int Pool_hd_t::shrink(long N) {
    lg_init();
    pwc_manager->wait_work();

    long num_vec = pwc_manager->num_vec();
    long num_chunk = pwc_manager->num_chunks();
    if (num_vec < N) {
        lg_info("nothing to do");
        lg_exit();
        return -2;
    }
    if (pwc_manager->num_empty()) {
        lg_err("pool with cache has deleted chunks? nothing done");
        lg_exit();
        return -1;
    }

    #if ENABLE_PROFILING
    logger->ev_total_chunks = num_chunk;
    logger->ev_vec_nbytes = num_vec * (CSD + 14.0);
    #endif

    ut_checker_t ut_checker(ut_checker_t::type_shrink, this, this->uid_table, NULL);
    
    int score_th = 1;
    long th_max_del = -N;
    for (; score_th < 65536; score_th++) {
        th_max_del += score_stat[score_th];
        if (th_max_del >= 0) break;
    }
    if (th_max_del < 0) lg_err("th_max_del < 0, something must be wrong, ignored");

    score_stat[score_th] -= th_max_del;
    for (int i = score_th + 1; i < 65536; i++) {
        score_stat[i] = 0;
    }
    
    std::atomic<int32_t> th_to_del(th_max_del);

    shrink_iterator_t iter(pwc_manager, num_chunk);
    
    #pragma omp parallel for num_threads(_num_threads)
    for (long thread = 0; thread < _num_threads; thread++) {
        for (;;) {
            chunk_t *dst, *src;
            int task = iter.pop(dst, src);
            if (task == -1) break;

            uint64_t uid_to_rm[chunk_max_nvecs * 2];
            int num_uid_to_rm = 0;

            if (task & shrink_iterator_t::dst_need_filt) {
                for (int i = 0; i < chunk_max_nvecs; i++) {
                    int to_remove = dst->score[i] > score_th ? 1 : 0;
                    if (dst->score[i] == score_th) {
                        int32_t remaining = th_to_del.fetch_sub(1, std::memory_order_relaxed);
                        if (remaining > 0) to_remove = 1;
                    }
                    if (to_remove) {
                        dst->score[i] = 0;
                        dst->norm[i] = 0;
                        dst->size--;
                        uid_to_rm[num_uid_to_rm++] = dst->u[i];
                    }
                }

                _normalize_chunk(dst, CSD);
            }

            if (task & shrink_iterator_t::src_need_filt) {
                for (int i = 0; i < chunk_max_nvecs; i++) {
                    int to_remove = src->score[i] > score_th ? 1 : 0;
                    if (src->score[i] == score_th) {
                        int32_t remaining = th_to_del.fetch_sub(1, std::memory_order_relaxed);
                        if (remaining > 0) to_remove = 1;
                    }
                    if (to_remove) {
                        src->score[i] = 0;
                        src->norm[i] = 0;
                        src->size--;
                        uid_to_rm[num_uid_to_rm++] = src->u[i];
                    }
                }

                _normalize_chunk(src, CSD);
            }

            if (num_uid_to_rm) ut_checker.task_commit(uid_to_rm, num_uid_to_rm);

            if (src) {
                int to_move = (chunk_max_nvecs - dst->size) < src->size ?
                          (chunk_max_nvecs - dst->size) : src->size;

                memcpy(&dst->score[dst->size], &src->score[src->size - to_move], sizeof(uint16_t) * to_move);
                memcpy(&dst->norm[dst->size], &src->norm[src->size - to_move], sizeof(int32_t) * to_move);
                memcpy(&dst->u[dst->size], &src->u[src->size - to_move], sizeof(uint64_t) * to_move);
                memcpy(&dst->vec[CSD * dst->size], &src->vec[CSD * (src->size - to_move)], CSD * to_move);
                memset(&src->score[src->size - to_move], 0, sizeof(uint16_t) * to_move);
                memset(&src->norm[src->size - to_move], 0, sizeof(int32_t) * to_move);
                dst->size += to_move;
                src->size -= to_move;
            }
            
            iter.release(dst, src);
        }
    }

    ut_checker.input_done();

    pwc_manager->wait_work();

    int res_chunk = iter.res_chunk();
    if (num_chunk - res_chunk != pwc_manager->num_empty()) {
        lg_err("verification failed, something must be wrong, ignored");
    }
    pthread_spin_lock(&pwc_manager->_deleted_ids_lock);
    pwc_manager->_num_chunks = res_chunk;
    pwc_manager->_num_deleted_ids = 0;
    pthread_spin_unlock(&pwc_manager->_deleted_ids_lock);

    #if ENABLE_PROFILING
    logger->ev_curr_chunks = res_chunk;
    logger->num_thread = _num_threads;
    #endif

    lg_report();
    lg_exit();
    return 0;
}

int Pool_hd_t::store() {
    pwc_manager->wait_work();
    return 0;
}

ut_checker_t::ut_checker_t(long type, Pool_hd_t *p, UidTable *uid_table, pwc_manager_t *pwc_manager) {
    #if ENABLE_PROFILING
    this->logger = p->logger;
    this->logger->ev_batch_num.store(0);
    this->logger->ev_batch_us.store(0);
    this->logger->ev_batch_rm_ssum.store(0);
    this->logger->ev_batch_check_ssum.store(0);
    this->logger->ev_max_table_size.store(0);
    this->logger->ev_ld_frags.store(0);
    this->logger->ev_st_frags.store(0);
    this->logger->ev_batch_start.tv_sec = 0;
    this->logger->ev_batch_start.tv_usec = 0;
    #endif

    this->CSD = p->CSD;

    this->_type = type;
    this->_uid_table = uid_table;
    if (type == type_sieve)  this->_swc_manager = (swc_manager_t *)pwc_manager;
    if (type == type_others) this->_pwc_manager = pwc_manager;
    
    pthread_spin_init(&_ut_lock, PTHREAD_PROCESS_SHARED);
    pthread_barrier_init(&_ut_barrier, NULL, 1);
    this->set_num_threads(default_num_threads);
    {
        long exp_max_holding = (type == type_shrink || type == type_check) ? default_max_uids : default_max_chunks;
        if (type == type_sieve) {
            long swc_limit = _swc_manager->max_cached_chunks() * 0.9;
            if (exp_max_holding > swc_limit) exp_max_holding = swc_limit;
        }
        if (type == type_others) {
            long pwc_limit = _pwc_manager->max_cached_chunks() * 0.9;
            if (exp_max_holding > pwc_limit) exp_max_holding = pwc_limit;
        }
        this->set_max_holding(exp_max_holding);
    }
    {
        long uid_num = uid_table->size();
        long exp_uid_size = 40L * uid_num;
        double ratio = default_batch_ratio;
        if (exp_uid_size > UT_TABLE_DRAM_SLIMIT) ratio = 0.025;
        if (exp_uid_size > UT_TABLE_DRAM_SLIMIT * 2.0) ratio = 0.05;
        long eeb = uid_num * ratio / Pool_hd_t::chunk_max_nvecs;
        if (exp_uid_size > UT_TABLE_DRAM_SLIMIT * 2.0) {
            if (eeb >= _max_holding * 0.9) eeb = _max_holding * 0.9;
        } else if (exp_uid_size > UT_TABLE_DRAM_SLIMIT) {
            if (eeb >= _max_holding * 0.7) eeb = _max_holding * 0.7;
        } else {
            if (eeb >= _max_holding * 0.5) eeb = _max_holding * 0.5;
        }
        this->set_exp_batch(eeb);
    }

    #if ENABLE_PROFILING
    p->logger->exp_batch = this->_exp_batch;
    p->logger->max_batch = this->_max_holding;
    p->logger->dbg("ut_checker initialized, #threads = %ld, expect batch size = %ld, max_holding = %ld", 
                    _num_threads, _exp_batch, _max_holding);
    #endif
}

ut_checker_t::~ut_checker_t() {
    this->wait_work();
    if (_type == type_shrink || _type == type_check) {
        for (long i = 0; i < _max_holding; i++) {
            if (_to_check[i].u) free(_to_check[i].u);
            if (_in_check[i].u) free(_in_check[i].u);
        }
    }
    if (_type == type_sieve) {
        for (long i = 0; i < _max_holding; i++) {
            if (_red_to_rm[i].u) free(_red_to_rm[i].u);
            if (_red_in_rm[i].u) free(_red_in_rm[i].u);
        }
        if (_red_to_rm) free(_red_to_rm - 1);
        if (_red_in_rm) free(_red_in_rm - 1);
    }
    if (_to_check) free(_to_check - 1);
    if (_in_check) free(_in_check - 1);
    pthread_spin_destroy(&_ut_lock);
    pthread_barrier_destroy(&_ut_barrier);
}

int ut_checker_t::set_num_threads(long num_threads) { 
    this->_num_threads = num_threads;
    pthread_barrier_destroy(&_ut_barrier);
    pthread_barrier_init(&_ut_barrier, NULL, num_threads);
    _ut_pool.resize(num_threads);
    return 0;
}

int ut_checker_t::set_max_holding(int max_holding) {
    this->_max_holding = max_holding;
    _to_check = (chunk_t *) realloc(_to_check, sizeof(chunk_t) * (max_holding + 1));
    _in_check = (chunk_t *) realloc(_in_check, sizeof(chunk_t) * (max_holding + 1));
    _to_check[0].size = Pool_hd_t::chunk_max_nvecs;
    _in_check[0].size = Pool_hd_t::chunk_max_nvecs;
    _to_check++;
    _in_check++; 
    memset(_to_check, 0, sizeof(chunk_t) * max_holding);
    memset(_in_check, 0, sizeof(chunk_t) * max_holding);
    if (_type == type_sieve) {
        _red_to_rm = (chunk_t *) realloc(_red_to_rm, sizeof(chunk_t) * (max_holding + 1));
        _red_in_rm = (chunk_t *) realloc(_red_in_rm, sizeof(chunk_t) * (max_holding + 1));
        _red_to_rm[0].size = Pool_hd_t::chunk_max_nvecs;
        _red_in_rm[0].size = Pool_hd_t::chunk_max_nvecs;
        _red_to_rm++;
        _red_in_rm++;
        memset(_red_to_rm, 0, sizeof(chunk_t) * max_holding);
        memset(_red_in_rm, 0, sizeof(chunk_t) * max_holding);
    }
    return 0;
}

int ut_checker_t::set_exp_batch(long exp_batch) {
    this->_exp_batch = exp_batch > _max_holding ? _max_holding : exp_batch;
    return 0;
}

int ut_checker_t::task_commit(chunk_t *chunk) {
    pthread_spin_lock(&_ut_lock);
    while (_num_to_check + _num_in_check >= _max_holding) {
        pthread_spin_unlock(&_ut_lock);
        
        {
            std::unique_lock<std::mutex> lock(_ut_mutex);
            _ut_cv.wait(lock);
        }

        pthread_spin_lock(&_ut_lock);
    }

    _to_check[_num_to_check++] = *chunk;

    pthread_spin_unlock(&_ut_lock);

    if (_num_to_check >= _exp_batch || _need_batch) trigger_batch();

    return 0;
}

int ut_checker_t::task_commit(uint64_t uid) {
    pthread_spin_lock(&_ut_lock);
    if (_to_check[_num_to_check - 1].size == Pool_hd_t::chunk_max_nvecs) {
        while (_num_to_check + _num_in_check >= _max_holding) {
            pthread_spin_unlock(&_ut_lock);
            
            {
                std::unique_lock<std::mutex> lock(_ut_mutex);
                _ut_cv.wait(lock);
            }

            pthread_spin_lock(&_ut_lock);
        }

        _num_to_check++;
        if (_to_check[_num_to_check - 1].u == NULL) {
            _to_check[_num_to_check - 1].u = (uint64_t *) malloc(Pool_hd_t::chunk_max_nvecs * sizeof(uint64_t));
            if (_to_check[_num_to_check - 1].u == NULL) {
                fprintf(stderr, "[Error] ut_checker_t::task_commit: fail to allocate memory for uid\n");
                return -2;
            }
        }
        _to_check[_num_to_check - 1].size = 0;
    }

    _to_check[_num_to_check - 1].u[_to_check[_num_to_check - 1].size++] = uid;
    pthread_spin_unlock(&_ut_lock);

    if ((_num_to_check >= _exp_batch || _need_batch) && _to_check[_num_to_check - 1].size == Pool_hd_t::chunk_max_nvecs) {
        trigger_batch();
    }

    return 0;
}

int ut_checker_t::task_commit(uint64_t *uids, long num) {
    if (_type == type_sieve) {
        pthread_spin_lock(&_ut_lock);
        int to_malloc = (num + _red_to_rm[_num_red_to_rm - 1].size) / Pool_hd_t::chunk_max_nvecs;
        if (to_malloc) {
            while (_num_red_to_rm + _num_red_in_rm + to_malloc >= _max_holding) {
                pthread_spin_unlock(&_ut_lock);
                
                {
                    std::unique_lock<std::mutex> lock(_ut_mutex);
                    _ut_cv.wait(lock);
                }

                pthread_spin_lock(&_ut_lock);
                to_malloc = (num + _red_to_rm[_num_red_to_rm - 1].size) / Pool_hd_t::chunk_max_nvecs;
            }

            for (int i = 0; i < to_malloc; i++) {
                if (_red_to_rm[_num_red_to_rm + i].u == NULL) {
                    _red_to_rm[_num_red_to_rm + i].u = (uint64_t *) malloc(Pool_hd_t::chunk_max_nvecs * sizeof(uint64_t));
                    if (_red_to_rm[_num_red_to_rm + i].u == NULL) {
                        pthread_spin_unlock(&_ut_lock);
                        lg_err("ut_checker failed to allocate memory for uid");
                        return -2;
                    }
                    _red_to_rm[_num_red_to_rm + i].size = 0;
                }
            }
        }

        int try_trigger = 0;
        while (num > 0) {
            if (_red_to_rm[_num_red_to_rm - 1].size == Pool_hd_t::chunk_max_nvecs) {
                _num_red_to_rm++;
                try_trigger = 1;
            }
            int nn =  num < Pool_hd_t::chunk_max_nvecs - _red_to_rm[_num_red_to_rm - 1].size ? 
                      num : Pool_hd_t::chunk_max_nvecs - _red_to_rm[_num_red_to_rm - 1].size;
            memcpy(_red_to_rm[_num_red_to_rm - 1].u + _red_to_rm[_num_red_to_rm - 1].size, uids, 8 * nn);
            _red_to_rm[_num_red_to_rm - 1].size += nn;
            uids += nn;
            num -= nn;
        }

        pthread_spin_unlock(&_ut_lock);

        if ((_num_red_to_rm >= _exp_batch || _need_batch) && try_trigger) {
            trigger_batch();
        }

        return 0;
    }
    
    pthread_spin_lock(&_ut_lock);
    int to_malloc = (num + _to_check[_num_to_check - 1].size) / Pool_hd_t::chunk_max_nvecs;
    if (to_malloc) {
        while (_num_to_check + _num_in_check + to_malloc >= _max_holding) {
            pthread_spin_unlock(&_ut_lock);
            
            {
                std::unique_lock<std::mutex> lock(_ut_mutex);
                _ut_cv.wait(lock);
            }

            pthread_spin_lock(&_ut_lock);
        }

        for (int i = 0; i < to_malloc; i++) {
            if (_to_check[_num_to_check + i].u == NULL) {
                _to_check[_num_to_check + i].u = (uint64_t *) malloc(Pool_hd_t::chunk_max_nvecs * sizeof(uint64_t));
                if (_to_check[_num_to_check + i].u == NULL) {
                    pthread_spin_unlock(&_ut_lock);
                    lg_err("ut_checker failed to allocate memory for uid");
                    return -2;
                }
                _to_check[_num_to_check + i].size = 0;
            }
        }
    }

    int try_trigger = 0;
    while (num > 0) {
        int nn =  num < Pool_hd_t::chunk_max_nvecs - _to_check[_num_to_check - 1].size ? 
                  num : Pool_hd_t::chunk_max_nvecs - _to_check[_num_to_check - 1].size;
        memcpy(_to_check[_num_to_check - 1].u + _to_check[_num_to_check - 1].size, uids, 8 * nn);
        _to_check[_num_to_check - 1].size += nn;
        if (_to_check[_num_to_check - 1].size == Pool_hd_t::chunk_max_nvecs) {
            _num_to_check++;
            try_trigger = 1;
        }
        uids += nn;
        num -= nn;
    }

    pthread_spin_unlock(&_ut_lock);

    if ((_num_to_check >= _exp_batch || _need_batch) && try_trigger) {
        trigger_batch();
    }

    return 0;
}

int ut_checker_t::trigger_batch() {
    std::unique_lock<std::mutex> lock(_ut_mutex);

    if (_running_threads) {
        _input_done = 1;
        _need_batch = 1;
        return 0;
    }

    _need_batch = 0;
    _running_threads = _num_threads;

    pthread_spin_lock(&_ut_lock);
    if (_type == type_sieve) {
        if (_num_to_check == 0 && _num_red_to_rm == 0) {
            pthread_spin_unlock(&_ut_lock);
            _running_threads = 0;
            return 0;
        }
        if (_num_to_check) {
            std::swap(_to_check, _in_check);
            _num_in_check = _num_to_check;
            _num_to_check = 0;
        }

        if (_num_red_to_rm) {
            std::swap(_red_to_rm, _red_in_rm);
            int ic_holds = _num_red_in_rm;
            _num_red_in_rm = _num_red_to_rm;
            _num_red_to_rm = 0;
            for (int i = _num_red_in_rm; i < _max_holding; i++) {
                if (_red_in_rm[i].u) {
                    uint64_t *tmp = _red_in_rm[i].u;
                    _red_in_rm[i].u = NULL;
                    _red_to_rm[ic_holds].u = tmp;
                    _red_to_rm[ic_holds++].size = 0;
                }
            }
        }
    } else {
        if (_num_to_check == 0) {
            pthread_spin_unlock(&_ut_lock);
            _running_threads = 0;
            return 0;
        }
        std::swap(_to_check, _in_check);
        int ic_holds = _num_in_check;
        _num_in_check = _num_to_check;
        _num_to_check = 0;
        if (_type == type_check || _type == type_shrink) {
            for (int i = _num_in_check; i < _max_holding; i++) {
                if (_in_check[i].u) {
                    uint64_t *tmp = _in_check[i].u;
                    _in_check[i].u = NULL;
                    _to_check[ic_holds].u = tmp;
                    _to_check[ic_holds++].size = 0;
                }
            }
        }
        
        if (_type == type_others && _max_available_id) {
            for (int i = _num_in_check - 1; i >= 0; i--) {
                if (_in_check[i].id > _max_available_id[0]) {
                    _num_in_check--;
                    _to_check[_num_to_check++] = _in_check[i];
                    _in_check[i] = _in_check[_num_in_check];
                }
            }
        }
    }
    pthread_spin_unlock(&_ut_lock);

    long table_size = _uid_table->size();
    long curr_hold = _uid_table->hold();
    #if ENABLE_PROFILING
    {
        long table_frag_size = (table_size / UidTable::ut_split + 1) * 24;
        long max_avail_frags = table_dram_slimit / table_frag_size > UidTable::ut_split ? 
                                UidTable::ut_split : table_dram_slimit / table_frag_size;
        lg_dbg("ut_checker new batch, #to_check = %d, #to_rm = %d, #avail_frag = %d(%.2f GB)", 
                _num_in_check, _num_red_in_rm, max_avail_frags, max_avail_frags * table_frag_size * 1e-9);

        if (_type == type_sieve) {
            gettimeofday(&logger->ev_batch_start, NULL);
            logger->ev_max_table_size = std::max(logger->ev_max_table_size.load(), (uint64_t)table_size);
        }
    }
    #endif
    for (long i = 0; i < _num_threads; i++) {
        _ut_pool.push([this, table_size, curr_hold, i]() {
            batch(i, table_size, curr_hold);
        });
    }

    return 0;
}

int ut_checker_t::input_done() {
    _input_done = 1;
    trigger_batch();

    return 0;
}

int ut_checker_t::batch(long tid, long table_size, long current_hold) {
    long table_frag_size = (table_size / UidTable::ut_split + 1) * 24;
    long max_avail_frags = table_dram_slimit / table_frag_size > UidTable::ut_split ? 
                           UidTable::ut_split : table_dram_slimit / table_frag_size;
    long batch_start_pt = 0;
    {   
        long curr_hits = 0;
        for (long i = 0; i < max_avail_frags; i++) if (_uid_table->_tables[i]) curr_hits++;

        long max_hits = curr_hits;
        for (long i = 0; i < UidTable::ut_split - 1; i++) {
            if (_uid_table->_tables[i]) curr_hits--;
            if (_uid_table->_tables[(i + max_avail_frags) % UidTable::ut_split]) curr_hits++;
            if (curr_hits > max_hits) {
                max_hits = curr_hits;
                batch_start_pt = i + 1;
            }
        }
    }

    uint64_t num_check = 0, num_notin = 0;
    for (int i = tid; i < _num_in_check; i += _num_threads) {
        num_check += _in_check[i].size;
    }
    num_notin = num_check;
    
    for (long ep = 0; ep < (UidTable::ut_split - 1) / max_avail_frags + 1; ep++) {
        const long start = (batch_start_pt + ep * max_avail_frags) % UidTable::ut_split;
        const long end = (batch_start_pt + ((ep + 1) * max_avail_frags < UidTable::ut_split ?
                                            (ep + 1) * max_avail_frags : UidTable::ut_split)) % UidTable::ut_split;
        
        long to_load[UidTable::ut_split] = {};
        long num_to_load = 0;
        for (long i = start;; i = (i + 1) % UidTable::ut_split) {
            if (_uid_table->_tables[i] == NULL) to_load[num_to_load++] = i;
            if (i == (end + UidTable::ut_split - 1) % UidTable::ut_split) break;
        }
        long to_dump[UidTable::ut_split] = {};
        long num_to_dump = 0;
        for (long i = start;;) {
            i = (i - 1 + UidTable::ut_split) % UidTable::ut_split;
            if (num_to_dump >= current_hold + num_to_load - max_avail_frags) break;
            if (_uid_table->_tables[i]) to_dump[num_to_dump++] = i;
        }
        current_hold = current_hold + num_to_load - num_to_dump;
        pthread_barrier_wait(&_ut_barrier);

        for (long i = tid; i < num_to_dump + num_to_load; i += _num_threads) {
            if (i < num_to_dump) {
                _uid_table->dump_table(to_dump[i]);
                #if ENABLE_PROFILING
                logger->ev_st_frags++;
                #endif
            } else {
                _uid_table->load_table(to_load[i - num_to_dump]);
                #if ENABLE_PROFILING
                logger->ev_ld_frags++;
                #endif
            }
        }

        pthread_barrier_wait(&_ut_barrier);

        if (_type == type_shrink && end > start) num_notin -= real_work<type_shrink, 0>(tid, start, end);
        if (_type == type_shrink && end <= start) num_notin -= real_work<type_shrink, 1>(tid, start, end);
        if (_type == type_check && end > start) num_notin -= real_work<type_check, 0>(tid, start, end);
        if (_type == type_check && end <= start) num_notin -= real_work<type_check, 1>(tid, start, end);
        if (_type == type_sieve && end > start) num_notin -= real_work<type_sieve, 0>(tid, start, end);
        if (_type == type_sieve && end <= start) num_notin -= real_work<type_sieve, 1>(tid, start, end);
        if (_type == type_others && end > start) num_notin -= real_work<type_others, 0>(tid, start, end);
        if (_type == type_others && end <= start) num_notin -= real_work<type_others, 1>(tid, start, end);
    }

    pthread_barrier_wait(&_ut_barrier);
    if (_type == type_shrink || _type == type_check) {
        for (long i = tid; i < _num_in_check; i += _num_threads) {
            if (_in_check[i].u) _in_check[i].size = 0;
        }
    }

    if (_type == type_sieve) {
        for (long i = tid; i < _num_red_in_rm; i += _num_threads) {
            if (_red_in_rm[i].u) _red_in_rm[i].size = 0;
        }
        {   
            for (long i = tid; i < _num_in_check; i += _num_threads) {
                _normalize_chunk(&_in_check[i], CSD);
            }
            int tid_total_chunks = (_num_in_check - tid - 1) / _num_threads + 1;
            int dst_id = 0;
            int src_id = tid_total_chunks - 1;
            while (src_id > dst_id) {
                chunk_t *src = &_in_check[src_id * _num_threads + tid];
                chunk_t *dst = &_in_check[dst_id * _num_threads + tid];
                int to_move = (Pool_hd_t::chunk_max_nvecs - dst->size) < src->size ?
                              (Pool_hd_t::chunk_max_nvecs - dst->size) : src->size;
                memcpy(&dst->score[dst->size], &src->score[src->size - to_move], sizeof(uint16_t) * to_move);
                memcpy(&dst->norm[dst->size], &src->norm[src->size - to_move], sizeof(int32_t) * to_move);
                memcpy(&dst->u[dst->size], &src->u[src->size - to_move], sizeof(uint64_t) * to_move);
                memcpy(&dst->vec[CSD * dst->size], &src->vec[CSD * (src->size - to_move)], CSD * to_move);
                memset(&src->score[src->size - to_move], 0, sizeof(uint16_t) * to_move);
                memset(&src->norm[src->size - to_move], 0, sizeof(int32_t) * to_move);
                dst->size += to_move;
                src->size -= to_move;
                if (dst->size == Pool_hd_t::chunk_max_nvecs) dst_id++;
                if (src->size == 0) src_id--;
            }
            pthread_spin_lock(&_stuck_lock);
            total_check[0] += num_check;
            total_notin[0] += num_notin;
            pthread_spin_unlock(&_stuck_lock);
        }
        for (long i = tid; i < _num_in_check; i += _num_threads) {
            chunk_t *real_chunk = &_swc_manager->_cached_chunks[_swc_manager->_chunk_status[_in_check[i].id] & 
                                                                pwc_manager_t::_ck_cache_id_mask];
            real_chunk->size = _in_check[i].size;
            if (real_chunk->size) _swc_manager->chunk_finalize(real_chunk);
            else _swc_manager->write_done(real_chunk);
        }
    }
    
    if (_type == type_others) {
        uint32_t _score_stat[65536] = {};
        for (long i = tid; i < _num_in_check; i += _num_threads) {
            _pwc_manager->_cached_chunks[_pwc_manager->_chunk_status[_in_check[i].id] & 
                                         pwc_manager_t::_ck_cache_id_mask].size = _in_check[i].size;
            for (int j = 0; j < Pool_hd_t::chunk_max_nvecs; j++) {
                _score_stat[_in_check[i].score[j]]++;
            }
            _pwc_manager->release_sync(_in_check[i].id);
        }
        pthread_spin_lock(&_score_stat_lock);
        for (int i = 0; i < 65536; i++) score_stat[i] += _score_stat[i];
        pthread_spin_unlock(&_score_stat_lock);
    }
    
    {
        std::unique_lock<std::mutex> lock(_ut_mutex);
        if (--_running_threads == 0) {
            lg_dbg("ut_checker batch done, #to_check = %d, #to_rm = %d", _num_to_check, _num_red_to_rm);
            #if ENABLE_PROFILING
            if (_type == type_sieve) {
                struct timeval now;
                gettimeofday(&now, NULL);
                logger->ev_batch_num++;
                logger->ev_batch_us += (now.tv_sec - logger->ev_batch_start.tv_sec) * 1000000ULL + 
                                    (now.tv_usec - logger->ev_batch_start.tv_usec);
                logger->ev_batch_check_ssum += _num_in_check;
                logger->ev_batch_rm_ssum += _num_red_in_rm;
            }
            #endif
            if (_type != type_check && _type != type_shrink) _num_in_check = 0;
            if (_type == type_shrink || _type == type_check) {
                pthread_spin_lock(&_ut_lock);
                for (int i = _num_to_check; i < _max_holding; i++) {
                    if (_num_in_check == 0) break;
                    if (_to_check[i].u == NULL) {
                        _to_check[i].u = _in_check[--_num_in_check].u;
                        _in_check[_num_in_check].u = NULL;
                    }
                }
                pthread_spin_unlock(&_ut_lock);
            }
            if (_type == type_sieve) {
                pthread_spin_lock(&_ut_lock);
                for (int i = _num_red_to_rm; i < _max_holding; i++) {
                    if (_num_red_in_rm == 0) break;
                    if (_red_to_rm[i].u == NULL) {
                        _red_to_rm[i].u = _red_in_rm[--_num_red_in_rm].u;
                        _red_in_rm[_num_red_in_rm].u = NULL;
                    }
                }
                pthread_spin_unlock(&_ut_lock);
            }
            if (_input_done) {
                _input_done = 0;
                lock.unlock();
                trigger_batch();
            }
            _ut_cv.notify_all();
        }
    }

    return 0;
}

template <int type, int cross>
long ut_checker_t::real_work(long tid, long start, long end) {
    long num_in = 0;

    if (type == type_sieve) {
        for (long i = tid; i < _num_red_in_rm; i += _num_threads) {
            for (int j = 0; j < _red_in_rm[i].size; j++) {
                uint64_t uid = _red_in_rm[i].u[j];
                int pos = _uid_table->normalize(uid) % UidTable::ut_split;
                if (cross ? (pos >= start || pos < end) : (pos >= start && pos < end)) {
                    int ret = _uid_table->erase(uid);
                    if (ret != 1) {
                        lg_err("ut_checker failed to erase uid %lx(%d)", uid, ret);
                    }
                }
            }
        }
    }
    for (long i = tid; i < _num_in_check; i += _num_threads) {
        for (int j = 0; j < Pool_hd_t::chunk_max_nvecs; j++) {
            bool need_work = (type == type_sieve || type == type_others) ? _in_check[i].score[j] : (j < _in_check[i].size);
            if (need_work) {
                uint64_t uid = _in_check[i].u[j];
                int pos = _uid_table->normalize(uid) % UidTable::ut_split;
                if (cross ? (pos >= start || pos < end) : (pos >= start && pos < end)) {
                    if (type == type_shrink) {
                        int ret = _uid_table->erase(uid);
                        if (ret != 1) lg_err("ut_checker failed to erase uid %lx(%d)", uid, ret);
                    }
                    if (type == type_check) {
                        int ret = _uid_table->check(uid);
                        if (ret != 1) {
                            if (ret == 0) _check_fail++;
                            else lg_err("ut_checker failed to check uid %lx(%d)", uid, ret);
                        }
                    }
                    if (type == type_sieve || type == type_others) {
                        int ret = _uid_table->insert(uid);
                        if (ret != 1) {
                            if (ret == 0) {
                                _in_check[i].score[j] = 0;
                                _in_check[i].norm[j] = 0;
                                _in_check[i].size--;
                                num_in++;
                            } else {
                                lg_err("ut_checker failed to insert uid %lx(%d)", uid, ret);
                            }
                        }
                    }
                }
            }
        }
    }

    return num_in;
}

long ut_checker_t::wait_work() {
    std::unique_lock<std::mutex> lock(_ut_mutex);
    _ut_cv.wait(lock, [this] { return _running_threads == 0 ; });

    return _type == type_check ? _check_fail.load() : 0;
}

#include <immintrin.h>

extern int _normalize_chunk(chunk_t *chunk, int CSD) {
    int chunk_modified = 0;

    constexpr int strict_check = 1;
    const __m256i msk = _mm256_set1_epi32(0xffffffff);
    const __m256i zero = _mm256_setzero_si256();
    int src = Pool_hd_t::chunk_max_nvecs - 1;
    int ptr = Pool_hd_t::chunk_max_nvecs - 16;
    
    for (; src >= chunk->size; ptr -= 16) {
        if (ptr < 0 && strict_check) break;
        __m256i score = _mm256_loadu_si256((__m256i *)&chunk->score[ptr]);
        if (_mm256_testz_si256(score, msk)) src -= 16;
        else break;
    }

    for (; src >= chunk->size; ptr -= 16) {
        if (ptr < 0 && strict_check) break;
        __m256i score = _mm256_loadu_si256((__m256i *)&chunk->score[ptr]);
        uint32_t cmp = _mm256_movemask_epi8(_mm256_cmpeq_epi16(score, zero));
        while (cmp) {
            int r = __builtin_clz(cmp);
            cmp ^= 0xc0000000 >> r;
            int dst = ptr + 15 - (r >> 1);
            if (dst != src) {
                chunk_modified = 1;
                memcpy(&chunk->vec[dst * CSD], &chunk->vec[src * CSD], CSD);
                chunk->score[dst] = chunk->score[src];
                chunk->norm[dst] = chunk->norm[src];
                chunk->u[dst] = chunk->u[src];
                chunk->norm[src] = 0;
                chunk->score[src] = 0;
            }
            src--;
        }
    }

    if (src >= chunk->size && strict_check) {
        fprintf(stderr, "[Warning] _normalize_chunk: wrong input format\n");
        sleep(100000);
    }

    return chunk_modified;
}

extern int pop_check_vec_err(int8_t *src1, int8_t *src2, int CSD) {
    __mmask64 m0 = CSD >= 64 ? 0xffffffff : ((1UL << CSD) - 1);
    __mmask64 m1 = CSD < 64 ? 0 : (CSD > 128 ? 0xffffffff : ((1UL << (CSD - 64)) - 1));
    __mmask64 m2 = CSD < 128 ? 0 : ((1UL << (CSD - 128)) - 1);
    __m512i x0 = _mm512_maskz_loadu_epi8(m0, (__m512i *) src1);
    __m512i x1 = _mm512_maskz_loadu_epi8(m1, (__m512i *) src1 + 64);
    __m512i x2 = _mm512_maskz_loadu_epi8(m2, (__m512i *) src1 + 128);
    __m512i y0 = _mm512_maskz_loadu_epi8(m0, (__m512i *) src2);
    __m512i y1 = _mm512_maskz_loadu_epi8(m1, (__m512i *) src2 + 64);
    __m512i y2 = _mm512_maskz_loadu_epi8(m2, (__m512i *) src2 + 128);
    __m512i d0 = _mm512_abs_epi8(_mm512_sub_epi8(x0, y0));
    __m512i d1 = _mm512_abs_epi8(_mm512_sub_epi8(x1, y1));
    __m512i d2 = _mm512_abs_epi8(_mm512_sub_epi8(x2, y2));
    __m512i dm = _mm512_max_epi8(_mm512_max_epi8(d0, d1), d2);
    #ifndef _mm256_reduce_max_epi8
    #define _mm256_reduce_max_epi8(_v) ({ \
        __m128i _v128_low = _mm256_castsi256_si128(_v); \
        __m128i _v128_high = _mm256_extracti128_si256(_v, 1); \
        __m128i _max128 = _mm_max_epi8(_v128_low, _v128_high); \
        _max128 = _mm_max_epi8(_max128, _mm_srli_si128(_max128, 8)); \
        _max128 = _mm_max_epi8(_max128, _mm_srli_si128(_max128, 4)); \
        _max128 = _mm_max_epi8(_max128, _mm_srli_si128(_max128, 2)); \
        _max128 = _mm_max_epi8(_max128, _mm_srli_si128(_max128, 1)); \
        _mm_extract_epi8(_max128, 0); \
    })
    #endif
    int vec_err = _mm256_reduce_max_epi8(_mm256_max_epi8(_mm512_castsi512_si256(dm), _mm512_extracti64x4_epi64(dm, 1)));
    vec_err = vec_err > 3 ? 3 : vec_err;

    return vec_err;
}
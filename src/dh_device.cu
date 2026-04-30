#include "../include/bgj_hd.h"
#include "../include/pool_hd_device.h"
#include "../include/bgj_hd_device.h"
#include "../include/dh_device.h"
#include "../include/utils.h"

float dot_avx2(float *src1, float *src2, long n);
double dot_avx2(double *src1, double *src2, long n);
void red_avx2(float *dst, float *src, float q, long n);
void copy_avx2(float *dst, float *src, long n);

static inline long ceil256(long n) {
    return ((n + 255L) / 256L) * 256L;
}


int _prepare_device_prop(int &num_devices, cudaDeviceProp device_props[]);

int Pool_hd_t::dh_insert(long target_index, double eta, double max_time, long *pos, double target_length) {
    lg_init();
    if (target_index < 0 || target_index > index_l || target_index < index_l - boost_data_t::max_boost_dim) {
        lg_warn("index(%ld) out of range [%ld, %ld], nothing done.\n", target_index, index_l - 
                boost_data_t::max_boost_dim >= 0 ? index_l - boost_data_t::max_boost_dim : 0, index_l);
        lg_exit();
        return -2;
    }

    int num_devices;
    cudaDeviceProp device_props[MAX_NUM_DEVICE];
    if (_prepare_device_prop(num_devices, device_props)) {
        lg_err("no device detected, nothing done.");
        lg_exit();
        return -1;
    }

    pwc_manager->wait_work();
    if (pwc_manager->max_cached_chunks() > pwc_manager_t::pwc_default_max_cached_chunks) {
        pwc_manager->set_max_cached_chunks(pwc_manager_t::pwc_default_max_cached_chunks);
    }

    const int old_ESD = ESD;
    const int new_ESD = index_l - target_index;

    this->set_boost_depth(new_ESD);

    bwc_manager_t *bwc_manager = new bwc_manager_t(this);
    dh_bucketer_t    *bucketer = new dh_bucketer_t(this, bwc_manager);
    dh_reducer_t      *reducer = new dh_reducer_t(this, bwc_manager, eta, 1);

    reducer->set_bucketer(bucketer);
    bucketer->set_reducer(reducer);

    #if ENABLE_PROFILING
    bwc_manager->logger->set_log_out(this->pwc_manager->logger->log_out());
    bwc_manager->logger->set_log_err(this->pwc_manager->logger->log_err());
    reducer->logger->set_log_out(this->logger->log_out());
    reducer->logger->set_log_err(this->logger->log_err());
    bucketer->logger->set_log_out(this->logger->log_out());
    bucketer->logger->set_log_err(this->logger->log_err());
    #endif

    bucketer->auto_bgj_params_set();
    reducer->auto_bgj_params_set();

    std::thread bucketer_thread([&]() { bucketer->run(max_time); });
    std::thread reducer_thread([&]() { reducer->run(target_length); });

    bucketer_thread.join();
    reducer_thread.join();

    const long ESD8 = (new_ESD + 7) / 8 * 8 < 32 ? 32 : (new_ESD + 7) / 8 * 8;
    int *res = reducer->get_result();

    delete reducer;
    delete bucketer;
    delete bwc_manager;

    lg_info("min_lift search done");

    float best_ratio = 0x1p30f;
    int remove_pos = -1, insert_pos = -1;
    for (int i = 0; i < ESD; i++) {
        if (target_length != 0.0 && ((float *)res)[i] * pow(eta, i) > 
           (target_length * _ratio * 0.9995f) * (target_length * _ratio * 0.9995f)) continue;
        float ratio = ((float *)res)[i] / (_boost_data->evec[i * ESD + i] * _boost_data->evec[i * ESD + i]);
        if (ratio * pow(eta, i) < best_ratio && (ratio < 0.9995f)) {
            best_ratio = ratio;
            insert_pos = index_l - ESD + i;
        }
    }

    for (int i = CSD - 1; i >= 0 && insert_pos != -1; i--) {
        int *coeff = &res[2 * ESD8 + (insert_pos - index_l + ESD) * 256];
        if (coeff[ESD8 + i] == 1) {
            remove_pos = index_l + i;
            break;
        }
        if (coeff[ESD8 + i] == -1) {
            remove_pos = index_l + i;
            for (int j = 0; j < 256; j++) coeff[j] = -coeff[j];
            break;
        }
    }

    if (insert_pos == -1 || remove_pos == -1) {
        if (insert_pos != -1 && remove_pos == -1) 
            lg_err("all coeff is not 1 or -1, please check");
        free(res);
        this->set_boost_depth(old_ESD);
        this->shrink_left();
        if (pos) pos[0] = -1;
        lg_exit();
        return -1;
    } else {
        lg_info("insert_pos = %d, remove_pos = %d", insert_pos, remove_pos);
        if (pos) pos[0] = insert_pos;
    }

    typedef insert_traits traits;

    local_data_t *local_data[MAX_NUM_DEVICE];
    for (int i = 0; i < num_devices; i++) {
        local_data[i] = (local_data_t *) calloc(256 + 4, 4);
        ((int *)local_data[i])[256] = insert_pos;
        ((int *)local_data[i])[257] = remove_pos;
        ((int *)local_data[i])[258] = old_ESD;
        ((int *)local_data[i])[259] = 1;
        for (int j = 0; j < ESD; j++) 
            ((int *)local_data[i])[j] = res[2 * ESD8 + (insert_pos - index_l + ESD) * 256 + j];
        for (int j = 0; j < CSD; j++) 
            ((int *)local_data[i])[boost_data_t::max_boost_dim + j] = res[2 * ESD8 + (insert_pos - index_l + ESD) * 256 + ESD8 + j];
    }
    free(res);

    pwc_manager->wait_work();
    if (CSD > 120) {
        long target_cached_chunks = PWC_DEFAULT_MAX_CACHED_CHUNKS + 
                                    BWC_DEFAULT_MAX_CACHED_CHUNKS +
                                    SWC_DEFAULT_MAX_CACHED_CHUNKS;
        if (pwc_manager->max_cached_chunks() != target_cached_chunks) {
            pwc_manager->set_max_cached_chunks(target_cached_chunks);
        }
    }

    stream_task_template<traits>(num_devices, device_props, local_data);
    for (int i = 0; i < num_devices; i++) CHECK_CUDA_ERR(cudaFree(local_data[i]));
    
    lg_exit();
    return 0;
}

int Pool_hd_t::dh_final(long target_index, double eta, double max_time, double target_length) {
    lg_init();
    if (target_index < 0 || target_index > index_l || target_index < index_l - boost_data_t::max_boost_dim) {
        lg_warn("index(%ld) out of range [%ld, %ld], nothing done.\n", target_index, index_l - 
                boost_data_t::max_boost_dim >= 0 ? index_l - boost_data_t::max_boost_dim : 0, index_l);
        return -2;
    }

    pwc_manager->wait_work();
    if (pwc_manager->max_cached_chunks() > pwc_manager_t::pwc_default_max_cached_chunks) {
        pwc_manager->set_max_cached_chunks(pwc_manager_t::pwc_default_max_cached_chunks);
    }

    const int old_ESD = ESD;
    const int new_ESD = index_l - target_index;

    this->set_boost_depth(new_ESD);

    bwc_manager_t *bwc_manager = new bwc_manager_t(this);
    dh_bucketer_t    *bucketer = new dh_bucketer_t(this, bwc_manager);
    dh_reducer_t      *reducer = new dh_reducer_t(this, bwc_manager, eta, 0);

    reducer->set_bucketer(bucketer);
    bucketer->set_reducer(reducer);

    #if ENABLE_PROFILING
    bwc_manager->logger->set_log_out(this->pwc_manager->logger->log_out());
    bwc_manager->logger->set_log_err(this->pwc_manager->logger->log_err());
    reducer->logger->set_log_out(this->logger->log_out());
    reducer->logger->set_log_err(this->logger->log_err());
    bucketer->logger->set_log_out(this->logger->log_out());
    bucketer->logger->set_log_err(this->logger->log_err());
    #endif

    bucketer->auto_bgj_params_set();
    reducer->auto_bgj_params_set();

    std::thread bucketer_thread([&]() { bucketer->run(max_time); });
    std::thread reducer_thread([&]() { reducer->run(target_length); });

    bucketer_thread.join();
    reducer_thread.join();

    const long ESD8 = (new_ESD + 7) / 8 * 8 < 32 ? 32 : (new_ESD + 7) / 8 * 8;
    int *res = reducer->get_result();

    delete reducer;
    delete bucketer;
    delete bwc_manager;

    typedef min_lift_traits traits;

    VEC_QP v_QP = NEW_VEC_QP(basis->NumCols());
    MAT_QP b_QP = basis->get_b();
    MAT_QP b_trans_QP = traits::b_trans_QP(this);

    for (int i = 0; i < ESD; i++) {
        int pos = index_l - ESD + i;
        if ((target_index >= 0 && target_index != pos) || (target_index < 0 && index_l - pos > -target_index)) continue;
        if (target_length != 0.0 && ((float *)res)[i] * pow(eta, i) > 
           (target_length * _ratio * 0.9995f) * (target_length * _ratio * 0.9995f)) continue;
        float old_scaled_norm = _boost_data->evec[i * ESD + i] * _boost_data->evec[i * ESD + i];
        float new_scaled_norm = ((float *)res)[i];
        if (new_scaled_norm < 0.9995f * old_scaled_norm) {
            int *coeff = &res[ESD8 * 2 + i * 256];
            int has_one = 0;
            for (int j = ESD8; j < ESD8 + CSD; j++) if (coeff[j] == 1 || coeff[j] == -1) has_one = 1;
            for (int j = 0; j < basis->NumCols(); j++) {
                v_QP.hi[j] = 0.0;
                v_QP.lo[j] = 0.0;
            }
            for (int j = 0; j < ESD; j++) {
                red(v_QP.hi, v_QP.lo, b_trans_QP.hi[j], b_trans_QP.lo[j], NTL::quad_float(-coeff[j]), basis->NumCols());
            }
            for (int j = 0; j < CSD; j++) {
                red(v_QP.hi, v_QP.lo, b_trans_QP.hi[ESD+j], b_trans_QP.lo[ESD+j], NTL::quad_float(-coeff[ESD8 + j]), basis->NumCols());
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

    free(res);
    FREE_VEC_QP(v_QP);
    FREE_MAT_QP(b_trans_QP);

    lg_exit();

    return 0;
}


int dhb_buffer_t::center_sampling(float *dst, Pool_hd_t *p, int num, int ESD8) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dis(-1024.0f, 1024.0f);

    int ESD = p->ESD;
    float *inorm = p->_boost_data->inorm;
    float *evec = p->_boost_data->evec;

    for (int i = 0; i < num; i++) {
        for (int j = 0; j < ESD8; j++) dst[i * ESD8 + j] = j < ESD ? dis(gen) : 0.0f;
        for (int l = ESD - 1; l >= 0; l--) {
            float q = roundf(dst[i * ESD8 + l] * inorm[l]);
            for (int j = 0; j <= l; j++) dst[i * ESD8 + j] -= q * evec[l * ESD + j];
        }
    }

    return 0;
}

int dhb_buffer_t::dh_data_prepare(dh_data_t *dst, Pool_hd_t *p, int CSD16, int ESD8) {
    dst->dhalf = p->_dhalf;
    dst->dshift = p->_dshift;

    // b_dual
    int8_t *ip = dst->b_dual;
    for (int row = 0; row < CSD16; row += 16) {
        for (int col = row; col < CSD16; col += 16) {
            for (int i = row; i < row + 16; i++) {
                for (int j = col; j < col + 16; j++) *ip++ = (i < p->CSD && j < p->CSD) ? p->_b_dual[i * Pool_hd_t::vec_nbytes + j] : 0;
            }
        }
    }

    // b_head
    float *fsrc = p->_boost_data->evec + p->ESD * p->ESD;
    for (int i = 0; i < CSD16; i++) {
        for (int j = 0; j < ESD8; j++) {
            dst->b_head[j * CSD16 + i] = (i < p->CSD && j < p->ESD) ? fsrc[i * p->ESD + j] : 0.0f;
        }
    }

    return 0;
}

// only called in v_dual_sampling
static float _compute_detn(float **D, float **Dt, long nlist, long d) {
    for (long i = 0; i < nlist; i++) {
        for (long j = 0; j < d; j++) Dt[j][i] = D[i][j];
    }

    float *dst_vec = (float *) NEW_VEC(d, sizeof(float));
    for (long i = 0; i < d; i++) {
        dst_vec[i] = dot_avx2(Dt[i], Dt[i], nlist);
        if (dst_vec[i] == 0.0f || ((i >= 1) && dst_vec[i] < 1e-6 * dst_vec[i-1])) {
            FREE_VEC(dst_vec);
            return 0.0f;
        }
        float r = 1.0 / dst_vec[i];
        for (long j = i + 1; j < d; j++) {
            float x = dot_avx2(Dt[i], Dt[j], nlist);
            red_avx2(Dt[j], Dt[i], x * r, nlist);
        }
    }

    float log_ret = 0.0;
    for (long i = 0; i < d; i++) log_ret += log(dst_vec[i]);
    log_ret /= d;
    FREE_VEC(dst_vec);
    return exp(log_ret);
}

int dhr_buffer_t::v_dual_sampling(float *dst, Pool_hd_t *p, int num, int ESD8) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, RAND_MAX);
    constexpr int expect_num_short_vec = 2000;
    constexpr int max_iter = 1000;

    const int ESD = p->ESD;

    Lattice_QP *Lp = p->basis->b_loc_QP(p->index_l - p->ESD, p->index_l);
    for (int i = 0; i < p->ESD; i++) {
        for (int j = 0; j <= i; j++) {
            NTL::quad_float val(Lp->get_b().hi[i][j], Lp->get_b().lo[i][j]);
            val *= p->_ratio;
            Lp->get_b().hi[i][j] = val.hi;
            Lp->get_b().lo[i][j] = val.lo;
        }
    }
    Lattice_QP *Ld = Lp->dual_QP();


    const float R = pow(0.85 * expect_num_short_vec, 1.0 / ESD) * Ld->gh();
    float *sv_store = (float *)  NEW_VEC(ESD8 * expect_num_short_vec, sizeof(float));
    float **sv_ptrs = (float **) calloc(expect_num_short_vec, sizeof(float *));
    float *sv_norms = (float *)  calloc(expect_num_short_vec, sizeof(float));
    int      sv_num = 0;

    do {
        const float R2 = R * R;
        float **b_fp    = (float **) NEW_MAT(ESD, ESD8, sizeof(float));
        float **miu_loc = (float **) NEW_MAT(ESD, ESD8 + 8, sizeof(float));
        float  *B_loc   = (float *)  NEW_VEC(ESD8 + 8, sizeof(float));
        for (long i = 0; i < ESD; i++) {
            for (long j = 0; j < ESD - i - 1; j++) miu_loc[i][j] = Ld->get_miu().hi[ESD - 1 - j][i];
            for (long j = i; j < ESD; j++) b_fp[i][j] = Ld->get_b().hi[i][j];
        }
        for (long i = 0; i < ESD; i++) B_loc[i] = Ld->get_B().hi[i];

        __attribute__ ((aligned (32))) float u[256] = {};
        __attribute__ ((aligned (32))) float l[256] = {};
        __attribute__ ((aligned (32))) float c[256] = {};
        __attribute__ ((aligned (32))) float g[256] = {};
        long t = 0;
        long t_max = 0;
        double kk = 5000. * (1.0 + (p->CSD - 145) / 40.0);
        if (kk < 2750.0) kk = 2750.0;
        if (kk > 5000.0) kk = 5000.0;
        for (long i = 0; i <= ESD; i++) g[i] = R2 * pow(1 - i / (float)ESD, ESD * ESD / kk);

        for (;;) {
            l[t] = l[t + 1] + (u[ESD - 1 - t] + c[t]) * (u[ESD - 1 - t] + c[t]) * B_loc[t];
            if (l[t] < g[t]) {
                if (t > 0) {
                    t--;
                    c[t] = dot_avx2(u, miu_loc[t], ESD - t - 1);
                    u[ESD - 1 - t] = round(-c[t]);
                } else {
                    if (l[0] != 0.0) {
                        sv_ptrs[sv_num] = sv_store + sv_num * ESD8;
                        for (long i = 0; i < ESD; i++) {
                            if (u[ESD - 1 - i] != 0.0) 
                                red_avx2(sv_ptrs[sv_num], b_fp[i], -u[ESD - 1 -i], ESD);
                        }
                        sv_norms[sv_num] = l[0];
                        if (++sv_num >= expect_num_short_vec) break;
                    }
                    
                    if (t >= t_max){
                        if (t > t_max) {
                            t_max = t;
                            if (t >= ESD) break;
                        }
                        u[ESD - 1 - t] += 1.0;
                    } else {
                        float c_tmp = round(-c[t]);
                        if (-c[t] >= c_tmp) {
                            if (u[ESD - 1 - t] <= c_tmp) {
                                u[ESD - 1 - t] = 2 * c_tmp - u[ESD - 1 - t] + 1;
                            } else {
                                u[ESD - 1 - t] = 2 * c_tmp - u[ESD - 1 - t];
                            }
                        } else {
                            if (u[ESD - 1 - t] < c_tmp) {
                                u[ESD - 1 - t] = 2 * c_tmp - u[ESD - 1 - t];
                            } else {
                                u[ESD - 1 - t] = 2 * c_tmp - u[ESD - 1 - t] - 1;
                            }
                        }
                    }
                }
            } else {
                t++;
                if (t >= t_max) {
                    if (t > t_max) {
                        t_max = t;
                        if (t >= ESD) break;
                    }
                    u[ESD - 1 - t] += 1.0;
                } else {
                    float c_tmp = round(-c[t]);
                    if (-c[t] >= c_tmp) {
                        if (u[ESD - 1 - t] <= c_tmp) {
                            u[ESD - 1 - t] = 2 * c_tmp - u[ESD - 1 - t] + 1;
                        }else{
                            u[ESD - 1 - t] = 2 * c_tmp - u[ESD - 1 - t];
                        }
                    }else{
                        if (u[ESD - 1 - t] < c_tmp) {
                            u[ESD - 1 - t] = 2 * c_tmp - u[ESD - 1 - t];
                        } else {
                            u[ESD - 1 - t] = 2 * c_tmp - u[ESD - 1 - t] - 1;
                        }
                    }
                }
            }
        }
        
        for (long i = 0; i < sv_num; i++) {
            for (long j = i; j > 0; j--) {
                if (sv_norms[j] < sv_norms[j-1]) {
                    float tmp = sv_norms[j];
                    sv_norms[j] = sv_norms[j-1];
                    sv_norms[j-1] = tmp;
                    float *tmpp = sv_ptrs[j];
                    sv_ptrs[j] = sv_ptrs[j-1];
                    sv_ptrs[j-1] = tmpp;
                } else break;
            }
        }
        
        FREE_MAT(miu_loc);
        FREE_MAT(b_fp);
        FREE_VEC(B_loc);
    } while (0);

    int  *select_ind = (int *)    malloc(num * sizeof(int));
    float **dp_table = (float **) NEW_MAT(num, num, sizeof(float));
    float    *dp_sum = (float *)  NEW_VEC(num, sizeof(float));
    for (long i = 0; i < num; i++) {
        select_ind[i] = i;
        if (sv_ptrs[i][i] == 0.0f && i < ESD) {
            for (long j = num; j < sv_num; j++) {
                if (sv_ptrs[j][i] != 0.0f) {
                    int dup = 0;
                    for (long k = 0; k < i; k++) if (j == select_ind[k]) dup = 1;
                    if (!dup) {
                        select_ind[i] = j;
                        break;
                    }
                }
            }
        }
        if (sv_ptrs[select_ind[i]][i] == 0.0f && i < ESD) {
            fprintf(stderr, "[Error] gen_dual_vec_list: dual vectors always in a subspace? ignored.\n");
        }
    }
    float **D = (float **)  NEW_MAT(num, ESD8, sizeof(float));
    float **Dt = (float **) NEW_MAT(ESD, num, sizeof(float));
    for (long i = 0; i < num; i++) copy_avx2(D[i], sv_ptrs[select_ind[i]], ESD);
    for (long i = 0; i < num; i++) {
        for (long j = 0; j <= i; j++) {
            dp_table[i][j] = dot_avx2(D[i], D[j], ESD);
            dp_table[j][i] = dp_table[i][j];
        }
    }
    for (long i = 0; i < num; i++) {
        for (long j = 0; j < num; j++) dp_sum[i] += dp_table[i][j] * dp_table[i][j];
    }
    
    long iter = 0;
    double sum_dp = 0.0;
    double sum_norm = 0.0;
    double cond = 0.0;
    for (long i = 0; i < num; i++) cond += dp_table[i][i];
    cond /= _compute_detn(D, Dt, num, ESD);
    for (long i = 0; i < num; i++) sum_dp += dp_sum[i];
    for (long i = 0; i < num; i++) sum_norm += dp_table[i][i];
    
    while (iter < max_iter) {
        long dst_ind;
        long count = 0;
        do {
            dst_ind = dis(gen) % num;
            count++;
        } while (dp_sum[dst_ind] * num < sum_dp && count < 3);
        
        long src = 0;
        long nrem = 5;
        float new_dp[1024];
        float new_sum;
        float new_norm;
        while (nrem --> 0) {
            long pass = 1;
            src += (dis(gen) % 163) + 1;
            if (src >= sv_num) src -= sv_num;
            
            for (long i = 0; i < num; i++) if (select_ind[i] == src) pass = 0;
            if (!pass) continue;

            for (long i = 0; i < num; i++) {
                if (i != dst_ind) {
                    new_dp[i] = dot_avx2(sv_ptrs[src], D[i], ESD);
                } else {
                    new_dp[i] = dot_avx2(sv_ptrs[src], sv_ptrs[src], ESD);
                }
            }
            new_sum = sum_dp;
            new_norm = sum_norm + new_dp[dst_ind] - dp_table[dst_ind][dst_ind];
            for (long i = 0; i < num; i++) {
                if (i == dst_ind) {
                    new_sum += new_dp[i] * new_dp[i] - dp_table[dst_ind][i] * dp_table[dst_ind][i];
                } else {
                    new_sum += 2 * (new_dp[i] * new_dp[i] - dp_table[dst_ind][i] * dp_table[dst_ind][i]);
                }
            }
            if (new_sum > sum_dp) pass = 0;

            if (pass) break;
        }
        int dup = 0;
        for (int i = 0; i < num; i++) {
            if (src == select_ind[i]) {
                dup = 1;
            }
        }
        if (dup) continue;

        iter++;

        long old_dst = select_ind[dst_ind];
        copy_avx2(D[dst_ind], sv_ptrs[src], ESD);
        float new_cond = 0.0;
        for (long i = 0; i < num; i++) new_cond += dp_table[i][i];
        new_cond /= _compute_detn(D, Dt, num, ESD);
        if (new_cond < cond) {
            select_ind[dst_ind] = src;
            for (long i = 0; i < num; i++) {
                dp_sum[i] += new_dp[i] * new_dp[i] - dp_table[dst_ind][i] * dp_table[dst_ind][i];
            }
            dp_sum[dst_ind] = 0.0;
            for (long i = 0; i < num; i++) dp_sum[dst_ind] += new_dp[i] * new_dp[i];
            sum_dp = new_sum;
            cond = new_cond;
            for (long i = 0; i < num; i++) {
                dp_table[dst_ind][i] = new_dp[i];
                dp_table[i][dst_ind] = new_dp[i];
            }
        } else {
            copy_avx2(D[dst_ind], sv_ptrs[old_dst], ESD);
        }
    }

    for (long i = 0; i < num; i++) {
        for (long j = 0; j < ESD8; j++) dst[i * (ESD8 + 1) + j] = D[i][j];
        dst[i * (ESD8 + 1) + ESD8] = 0.0f;
    }

    // free
    FREE_MAT(D);
    FREE_MAT(Dt);
    FREE_MAT(dp_table);
    FREE_VEC(dp_sum);
    free(select_ind);
    free(sv_norms);
    free(sv_ptrs);
    FREE_VEC(sv_store);
    delete Lp;
    delete Ld;

    return 0;
}


dhb_buffer_t::dhb_buffer_t(dh_bucketer_t *bucketer) {
    this->bucketer = bucketer;

    /// fixed during sieving
    this->CSD           = bucketer->_pool->CSD;
    this->CSD16         = Pool_hd_t::vec_nbytes;
    this->ESD           = bucketer->_pool->ESD;
    this->ESD8          = (ESD + 7) / 8 * 8 < 32 ? 32 : (ESD + 7) / 8 * 8;
    this->max_batch     = bucketer->_max_batch;
    this->out_max_size  = DH_BSIZE_RATIO / sqrt(bucketer->_pool->pwc_manager->num_vec()) * traits::taskVecs;
    out_max_size        = (out_max_size + 511) / 512 * 512;
    if (out_max_size > traits::taskVecs) out_max_size = traits::taskVecs;
    this->radius        = bucketer->_pool->basis->gh(bucketer->_pool->index_l - ESD, bucketer->_pool->index_l) * 
                          bucketer->_pool->_ratio * bucketer->_beta;
    this->radius        = radius * radius;                            
    
    /// thread & device info
    int _num_devices;
    _num_devices = hw::gpu_num;
    pthread_spin_init(&gram_lock, PTHREAD_PROCESS_SHARED);
    this->num_devices   = _num_devices;
    this->num_threads   = bucketer->_num_threads;
    this->streams       = (cudaStream_t *) malloc(num_threads * sizeof(cudaStream_t));
    this->used_gram     = (long *) calloc(num_devices, sizeof(long));

    #if ENABLE_PROFILING
    this->logger = bucketer->logger;
    this->logger->num_devices = this->num_devices;
    this->logger->num_threads = this->num_threads;
    this->logger->chunk_nbytes = (14 + CSD) * Pool_hd_t::chunk_max_nvecs;
    #endif

    lg_dbg("#device %ld, ESD8 %ld, out_max_size %ld, radius %.2f", 
            num_devices, ESD8, out_max_size, sqrt(radius) / bucketer->_pool->_ratio);
    
    /// runtime data
    long nbytes_task_vecs = ceil256(num_threads * sizeof(int32_t));
    long nbytes_h_center  = ceil256(max_batch * ESD8 * sizeof(float));
    long nbytes_h_data    = ceil256(sizeof(dh_data_t));
    long nbytes_h_out     = ceil256(max_batch * (out_max_size + 1) * sizeof(uint32_t));
    long nbytes_pinned = nbytes_task_vecs + nbytes_h_center + nbytes_h_data + nbytes_h_out * num_threads;
    nbytes_pinned = ((nbytes_pinned + 4095L) / 4096L) * 4096L;
    char *pinned_buf = NULL;
    if (posix_memalign((void **)&pinned_buf, 4096, nbytes_pinned)) {
        lg_err("posix_memalign failed");
    }
    CHECK_CUDA_ERR(cudaHostRegister(pinned_buf, nbytes_pinned, cudaHostAllocPortable));
    pinned_ram.fetch_add(nbytes_pinned, std::memory_order_relaxed);
    
    task_vecs    = (int32_t *)  (pinned_buf);
    h_center     = (float *)    (pinned_buf + nbytes_task_vecs);
    h_data       = (dh_data_t *)(pinned_buf + nbytes_task_vecs + nbytes_h_center);

    d_vec      = (int8_t **)   malloc(num_threads * sizeof(int8_t *));
    h_out      = (uint32_t **) malloc(num_threads * sizeof(uint32_t *));
    d_out      = (uint32_t **) malloc(num_threads * sizeof(uint32_t *));
    d_center   = (float **)    malloc(num_devices * sizeof(float *));
    d_data     = (dh_data_t **)malloc(num_devices * sizeof(dh_data_t *));
    bml_data   = (local_data_t **)malloc(num_devices * sizeof(local_data_t *));
    d_upk      = (int8_t **)   malloc(num_threads * sizeof(int8_t *));

    dhb_kernel = traits::dhb_kernel_chooser(CSD16, ESD8);
    if (!dhb_kernel) lg_err("dhb_kernel_chooser: unsupported CSD16 = %ld, ESD8 = %ld", CSD16, ESD8);
    bml_kernel = traits::mlf_kernel_chooser(CSD16, ESD8);
    if (!bml_kernel) lg_err("bml_kernel_chooser: unsupported CSD16 = %ld, ESD8 = %ld", CSD16, ESD8);
    for (long i = 0; i < num_threads; i++) {
        h_out[i] = (uint32_t *) (pinned_buf + nbytes_task_vecs + nbytes_h_center + nbytes_h_data + i * nbytes_h_out);
    }

    dh_data_prepare(h_data, bucketer->_pool, CSD16, ESD8);
}

dhb_buffer_t::~dhb_buffer_t() {
    /// thread & device info free
    for (int i = 0; i < num_devices; i++) {
        if (used_gram[i] != 0) lg_err("%ld bytes of GPU memory leak detected", used_gram[i]);
    }
    pthread_spin_destroy(&gram_lock);
    free(streams);
    free(used_gram);

    /// runtime data free
    CHECK_CUDA_ERR(cudaHostUnregister(task_vecs));
    free(task_vecs);
    free(d_vec);
    free(h_out);
    free(d_out);
    free(d_center);
    free(d_data);
    free(bml_data);
    free(d_upk);
}

int dhb_buffer_t::device_init(int tid) {
    int device_ptr = hw::gpu_ptr(tid, num_threads);
    CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[device_ptr]));
    CHECK_CUDA_ERR(cudaStreamCreate(&streams[tid]));

    #if ENABLE_PROFILING
    for (int i = 0; i < traits::taskChunks; i++) {
        CHECK_CUDA_ERR(cudaEventCreate(&logger->h2d_start[tid][i]));
        CHECK_CUDA_ERR(cudaEventCreate(&logger->h2d_stop[tid][i]));
    }
    CHECK_CUDA_ERR(cudaEventCreate(&logger->d2h_start[tid]));
    CHECK_CUDA_ERR(cudaEventCreate(&logger->d2h_stop[tid]));
    CHECK_CUDA_ERR(cudaEventCreate(&logger->kernel_start[tid]));
    #endif

    long _used_gram = 0;

    CHECK_CUDA_ERR(cudaMalloc(&d_vec[tid], traits::taskVecs * CSD16 * sizeof(int8_t)));
    CHECK_CUDA_ERR(cudaMalloc(&d_out[tid], max_batch * (out_max_size + 1L) * sizeof(uint32_t)));
    CHECK_CUDA_ERR(cudaMalloc(&d_upk[tid], traits::taskVecs * CSD16 * sizeof(int8_t)));
    _used_gram += max_batch * (out_max_size + 1L) * 4L + traits::taskVecs * CSD16 * 2;
    if (tid == 0 || hw::gpu_ptr(tid - 1, num_threads) != device_ptr) {
        bml_data[device_ptr] = (local_data_t *)calloc(1, 4);
        ((int *)bml_data[device_ptr])[0] = bucketer->_reducer->_force_one;
        min_lift_traits::_prep_device_local_data(CSD16, ESD8, bml_data[device_ptr], bucketer->_pool);
        CHECK_CUDA_ERR(cudaMalloc(&d_center[device_ptr], max_batch * ESD8 * sizeof(float)));
        CHECK_CUDA_ERR(cudaMalloc(&d_data[device_ptr], sizeof(dh_data_t)));
        CHECK_CUDA_ERR(cudaMemcpyAsync(d_data[device_ptr], h_data, sizeof(dh_data_t), cudaMemcpyHostToDevice, streams[tid]));
        /// prepare const
        {
            float dh_head_val[1176] = {};
            float dh_inorm_val[48] = {};

            for (int i = 0; i < ESD; i++) dh_inorm_val[i] = bucketer->_pool->_boost_data->inorm[i];
            int bias = 0;
            for (int i = ESD8 - 1; i >= 0; i--) {
                for (int j = 0; j <= i; j++)
                    dh_head_val[bias + j] = i < ESD ? bucketer->_pool->_boost_data->evec[i * ESD + j] : 0.0f;
                bias += i + 1;
            }

            set_dh_head(dh_head_val, streams[tid]);
            set_dh_inorm(dh_inorm_val, streams[tid]);
        }
        _used_gram += max_batch * ESD8 * 4L + sizeof(dh_data_t) + sizeof(local_data_t);
    }

    cudaFuncSetAttribute(dhb_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dhb_shmem);
    cudaFuncSetAttribute(_device_unpack<176>, cudaFuncAttributeMaxDynamicSharedMemorySize, utils_t::packshmem);
    cudaFuncSetAttribute(bml_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, min_lift_traits::dynamic_shmem);

    pthread_spin_lock(&gram_lock);
    used_gram[device_ptr] += _used_gram;
    pthread_spin_unlock(&gram_lock);
    if (used_gram[device_ptr] > DHB_GRAM_SLIMIT) {
        lg_err("device %d used %ld bytes of GPU memory, exceeds the slimit %ld", hw::gpu_id_list[device_ptr], used_gram[device_ptr], DHB_GRAM_SLIMIT);
    }

    return 0;
}

int dhb_buffer_t::device_done(int tid) {
    int device_ptr = hw::gpu_ptr(tid, num_threads);
    CHECK_CUDA_ERR(cudaStreamDestroy(streams[tid]));

    #if ENABLE_PROFILING
    for (int i = 0; i < traits::taskChunks; i++) {
        CHECK_CUDA_ERR(cudaEventDestroy(logger->h2d_start[tid][i]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->h2d_stop[tid][i]));
    }
    CHECK_CUDA_ERR(cudaEventDestroy(logger->d2h_start[tid]));
    CHECK_CUDA_ERR(cudaEventDestroy(logger->d2h_stop[tid]));
    CHECK_CUDA_ERR(cudaEventDestroy(logger->kernel_start[tid]));
    #endif

    long _used_gram = 0;
    
    CHECK_CUDA_ERR(cudaFree(d_vec[tid]));
    CHECK_CUDA_ERR(cudaFree(d_out[tid]));
    CHECK_CUDA_ERR(cudaFree(d_upk[tid]));
    _used_gram += max_batch * (out_max_size + 1L) * 4L + traits::taskVecs * CSD16 * 2;
    if (tid == 0 || hw::gpu_ptr(tid - 1, num_threads) != device_ptr) {
        CHECK_CUDA_ERR(cudaFree(bml_data[device_ptr]));
        CHECK_CUDA_ERR(cudaFree(d_center[device_ptr]));
        CHECK_CUDA_ERR(cudaFree(d_data[device_ptr]));
        _used_gram += max_batch * ESD8 * 4L + sizeof(dh_data_t) + sizeof(local_data_t);
    }

    pthread_spin_lock(&gram_lock);
    used_gram[device_ptr] -= _used_gram;
    pthread_spin_unlock(&gram_lock);

    return 0;
}

int dhb_buffer_t::center_prep(int batch) {
    this->curr_batch = batch;

    center_sampling(h_center, bucketer->_pool, batch, ESD8);

    /// copy centers to devices
    for (int i = 0; i < num_devices; i++) {
        CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[i]));
        CHECK_CUDA_ERR(cudaMemcpy(d_center[i], h_center, curr_batch * ESD8 * sizeof(float), cudaMemcpyHostToDevice));
    }

    for (int i = 0; i < num_threads; i++) task_vecs[i] = 0;

    return 0;
}

int dhb_buffer_t::h2d(int tid, chunk_t *chunk) {
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->h2d_start[tid][logger->h2d_count[tid]], streams[tid]));
    #endif
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_vec[tid] + CSD * task_vecs[tid], chunk->vec, 
                    chunk->size * CSD, cudaMemcpyHostToDevice, streams[tid]));
    task_vecs[tid] += chunk->size;
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->h2d_stop[tid][logger->h2d_count[tid]], streams[tid]));
    logger->h2d_count[tid]++;
    #endif

    return 0;
}

int dhb_buffer_t::run(int tid) {
    int device_ptr = hw::gpu_ptr(tid, num_threads);
    if (bucketer->init_round) {
        _device_unpack<176><<<dhb_blocks, dhb_threads, utils_t::packshmem, streams[tid]>>>(
            d_upk[tid], d_vec[tid], CSD, task_vecs[tid]
        );
        bml_kernel<<<dhb_blocks, dhb_threads, min_lift_traits::dynamic_shmem, streams[tid]>>>(
            d_upk[tid], task_vecs[tid], bml_data[device_ptr]
        );
    }
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->kernel_start[tid], streams[tid]));
    #endif
    CHECK_CUDA_ERR(cudaMemsetAsync(d_out[tid], 0, curr_batch * sizeof(uint32_t), streams[tid]));
    dhb_kernel<<<dhb_blocks, dhb_threads, dhb_shmem, streams[tid]>>>(
        d_out[tid], out_max_size, d_center[device_ptr], radius, curr_batch, d_vec[tid], task_vecs[tid], CSD, d_data[device_ptr]
    );
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->d2h_start[tid], streams[tid]));
    #endif
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_out[tid], d_out[tid], curr_batch * (out_max_size + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[tid]));
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->d2h_stop[tid], streams[tid]));
    #endif
    CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));
    #if ENABLE_PROFILING
    {
        float h2d_tt = .0f, d2h_tt, kernel_tt;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&kernel_tt, logger->kernel_start[tid], logger->d2h_start[tid]));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&d2h_tt, logger->d2h_start[tid], logger->d2h_stop[tid]));
        for (int i = 0; i < logger->h2d_count[tid]; i++) {
            float tt;
            CHECK_CUDA_ERR(cudaEventElapsedTime(&tt, logger->h2d_start[tid][i], logger->h2d_stop[tid][i]));
            h2d_tt += tt;
        }
        logger->h2d_count[tid] = 0;
        logger->ev_kernel_us += 1000.f * kernel_tt;
        logger->ev_d2h_us += 1000.f * d2h_tt;
        logger->ev_h2d_us += 1000.f * h2d_tt;
        logger->ev_total_256ops += ceil(task_vecs[tid] / 16.0) * curr_batch / 16.0;
        logger->ev_d2h_nbytes += curr_batch * (out_max_size + 1) * sizeof(uint32_t);
        logger->ev_h2d_nbytes += task_vecs[tid] * CSD * sizeof(int8_t) + task_vecs[tid] * sizeof(int32_t);
    }
    task_vecs[tid] = 0;
    #endif
    int full_count = 0;
    for (int i = 0; i < curr_batch; i++) {
        if (h_out[tid][i] > out_max_size) { h_out[tid][i] = out_max_size; full_count++; }
    }
    if (full_count > 0.5 * curr_batch && CSD > 120) lg_warn("%d / %d buckets full", full_count, curr_batch);
    return 0;
}

int dhb_buffer_t::out(int tid, int bid, int *num, int **entry) {
    if (tid < 0 || tid >= num_threads || bid < 0 || bid >= curr_batch) return -1;

    *num = h_out[tid][bid];
    *entry = (int *)(h_out[tid] + curr_batch + out_max_size * bid);
    return 0;
}


dhr_buffer_t::dhr_buffer_t(dh_reducer_t *reducer, double target_length) {
    this->reducer = reducer;

    /// fixed during dh
    this->CSD = reducer->_pool->CSD;
    this->CSD16 = Pool_hd_t::vec_nbytes;
    this->ESD = reducer->_pool->ESD;
    this->ESD8 = (ESD + 7) / 8 * 8 < 32 ? 32 : (ESD + 7) / 8 * 8;
    this->th = traits::dh_threshold(CSD, ESD);
    double pool_size = reducer->_pool->pwc_manager->num_vec();
    double beta = pow((DH_BSIZE_RATIO / sqrt(3.2 * pow(4./3., CSD * .5))), 1.0 / ESD);
    if (beta > 0.95) beta = 0.95;
    this->buc_max_size = DH_BSIZE_RATIO * sqrt(pool_size) * 
            (6.9367057575991202 - 0.0254642974101921 * ESD - 6.1327667887667809 * beta) * 1.1;
    if (this->buc_max_size > pool_size) this->buc_max_size = pool_size;
    this->buc_max_size = (buc_max_size + 511L) / 512L * 512L;
    this->out_max_size = traits::dhr_out_max_size(CSD);
    this->out_max_size = (out_max_size + 511L) / 512L * 512L;
    this->target_length = target_length;
    this->report_range = reducer->_force_one ? 10 : 8;

    /// thread & device info
    int _num_devices;
    _num_devices = hw::gpu_num;
    pthread_spin_init(&gram_lock, PTHREAD_PROCESS_SHARED);
    this->num_devices = _num_devices;
    this->num_threads = reducer->_num_threads;
    this->used_gram = (long *) calloc(num_devices, sizeof(long));
    this->streams = (cudaStream_t *) malloc(num_threads * sizeof(cudaStream_t));

    #if ENABLE_PROFILING
    this->logger = reducer->logger;
    this->logger->num_devices = this->num_devices;
    this->logger->num_threads = this->num_threads;
    #endif

    lg_dbg("buc_max_size %ld, out_max_size %ld, target_length %.2f, th %d", buc_max_size, out_max_size, target_length, th);

    /// runtime data
    long nbytes_task_vecs = ceil256(num_threads * sizeof(int32_t));
    long nbytes_buc_vecs  = ceil256(num_threads * sizeof(int32_t));
    long nbytes_h_data    = ceil256(sizeof(dh_data_t));
    long nbytes_h_res     = ceil256(ESD8 * (1024 + 4 + 4));
    long nbytes_pinned = nbytes_task_vecs + nbytes_buc_vecs + nbytes_h_data + nbytes_h_res * num_threads;
    nbytes_pinned = ((nbytes_pinned + 4095L) / 4096L) * 4096L;
    char *pinned_buf = NULL;
    if (posix_memalign((void **)&pinned_buf, 4096, nbytes_pinned)) {
        lg_err("posix_memalign failed");
    }
    CHECK_CUDA_ERR(cudaHostRegister(pinned_buf, nbytes_pinned, cudaHostAllocPortable));
    pinned_ram.fetch_add(nbytes_pinned, std::memory_order_relaxed);

    task_vecs    = (int32_t *)  (pinned_buf);
    buc_vecs     = (int32_t *)  (pinned_buf + nbytes_task_vecs);
    h_data       = (dh_data_t *)(pinned_buf + nbytes_task_vecs + nbytes_buc_vecs);

    pthread_spin_init(&min_lock, PTHREAD_PROCESS_SHARED);
    memset(task_vecs, 0, num_threads * sizeof(int32_t));
    memset(buc_vecs, 0, num_threads * sizeof(int32_t));
    d_upk       = (int8_t **) malloc(num_threads * sizeof(int8_t *));
    d_vec16     = (int8_t **) malloc(num_threads * sizeof(int8_t *));
    d_dh        = (int **)  malloc(num_threads * sizeof(int *));
    d_num_out   = (int **) malloc(num_threads * sizeof(int *));
    d_out       = (int **) malloc(num_threads * sizeof(int *));
    local_data  = (local_data_t **) malloc(num_threads * sizeof(local_data_t *));
    d_data      = (dh_data_t **) malloc(num_devices * sizeof(dh_data_t *));

    res         = (int *) malloc(ESD8 * (1024L + 4L + 4L));
    h_res       = (int **) malloc(num_threads * sizeof(int *));
    for (int j = 0; j < ESD8; j++) ((float *)res)[j] = 0x1p40f;
    for (long i = 0; i < num_threads; i++) {
        h_res[i] = (int32_t *)  (pinned_buf + nbytes_task_vecs + nbytes_buc_vecs + nbytes_h_data + i * nbytes_h_res);
        for (int j = 0; j < ESD8; j++) ((float *)h_res[i])[j] = 0x1p40f;
    }
    
    b_trans_QP  = min_lift_traits::b_trans_QP(reducer->_pool);
    v_QP        = (VEC_QP *) malloc(num_threads * sizeof(VEC_QP));
    for (int i = 0; i < num_threads; i++) {
        v_QP[i] = NEW_VEC_QP(reducer->_pool->basis->NumCols());
    }
    dhb_buffer_t::dh_data_prepare(h_data, reducer->_pool, CSD16, ESD8);
    v_dual_sampling(h_data->v_dual, reducer->_pool, hdh_nbits, ESD8);
    
    for (int i = 0; i < num_threads; i++) {
        v2h_kernel[i] = traits::v2h_kernel_chooser(CSD16, ESD8, i % 16);
        if (!v2h_kernel[i]) lg_err("v2h_kernel_chooser: unsupported CSD16 = %d, ESD8 = %d", CSD16, ESD8);
    }
    dhr_kernel = traits::dhr_kernel_chooser();
    if (!dhr_kernel) lg_err("dhr_kernel_chooser: unsupported");
    fpv_kernel = traits::fpv_kernel_chooser(CSD16);
    if (!fpv_kernel) lg_err("fpv_kernel_chooser: unsupported CSD16 = %d", CSD16);
    mlf_kernel = traits::mlf_kernel_chooser(CSD16, ESD8);
    if (!mlf_kernel) lg_err("mlf_kernel_chooser: unsupported CSD16 = %d, ESD8 = %d", CSD16, ESD8);

    gettimeofday(&last_report, NULL);
}

dhr_buffer_t::~dhr_buffer_t() {
    /// thread & device info free
    for (int i = 0; i < num_devices; i++) {
        if (used_gram[i] != 0) lg_err("%ld bytes of GPU memory leak detected", used_gram[i]);
    }
    pthread_spin_destroy(&gram_lock);
    pthread_spin_destroy(&min_lock);
    free(streams);
    free(used_gram);

    /// host free
    for (int i = 0; i < num_threads; i++) {
        FREE_VEC_QP(v_QP[i]);
    }
    free(res);
    free(h_res);
    free(v_QP);
    FREE_MAT_QP(b_trans_QP);
    
    /// runtime data free
    CHECK_CUDA_ERR(cudaHostUnregister(task_vecs));
    free(task_vecs);
    free(d_upk);
    free(d_vec16);
    free(d_dh);
    free(d_num_out);
    free(d_out);
    free(local_data);
    free(d_data);
}

int dhr_buffer_t::device_init(int tid) {
    int device_ptr = hw::gpu_ptr(tid, num_threads);
    CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[device_ptr]));
    CHECK_CUDA_ERR(cudaStreamCreate(&streams[tid]));

    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventCreate(&logger->h2d_start[tid]));
    CHECK_CUDA_ERR(cudaEventCreate(&logger->h2d_stop[tid]));
    CHECK_CUDA_ERR(cudaEventCreate(&logger->dhr_start[tid]));
    CHECK_CUDA_ERR(cudaEventCreate(&logger->dhr_stop[tid]));
    CHECK_CUDA_ERR(cudaEventCreate(&logger->v2h_start[tid]));
    CHECK_CUDA_ERR(cudaEventCreate(&logger->v2h_stop[tid]));
    CHECK_CUDA_ERR(cudaEventCreate(&logger->mlf_start[tid]));
    CHECK_CUDA_ERR(cudaEventCreate(&logger->mlf_stop[tid]));
    #endif

    long _used_gram = 0;

    CHECK_CUDA_ERR(cudaMalloc(&d_upk[tid], traits::taskVecs * CSD16 * sizeof(int8_t)));
    CHECK_CUDA_ERR(cudaMalloc(&d_vec16[tid], buc_max_size * CSD16 * sizeof(int8_t)));
    CHECK_CUDA_ERR(cudaMalloc(&d_dh[tid], buc_max_size * dh_nbits / 8));
    CHECK_CUDA_ERR(cudaMalloc(&d_num_out[tid], sizeof(int)));
    CHECK_CUDA_ERR(cudaMemsetAsync(d_num_out[tid], 0, sizeof(int), streams[tid]));
    CHECK_CUDA_ERR(cudaMalloc(&d_out[tid], out_max_size * 2 * sizeof(uint32_t)));
    local_data[tid] = (local_data_t *)calloc(1, 4);
    ((int *)local_data[tid])[0] = reducer->_force_one;
    min_lift_traits::_prep_device_local_data(CSD16, ESD8, local_data[tid], reducer->_pool);
    _used_gram += traits::taskVecs * CSD16 + buc_max_size * (long)(CSD16 + dh_nbits / 8) + out_max_size * 8 + sizeof(local_data_t);
    if (tid == 0 || hw::gpu_ptr(tid - 1, num_threads) != device_ptr) {
        CHECK_CUDA_ERR(cudaMalloc(&d_data[device_ptr], sizeof(dh_data_t)));
        CHECK_CUDA_ERR(cudaMemcpyAsync(d_data[device_ptr], h_data, sizeof(dh_data_t), cudaMemcpyHostToDevice, streams[tid]));
        _used_gram += sizeof(dh_data_t);
    }

    CHECK_CUDA_ERR(cudaFuncSetAttribute(dhr_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dhr_shmem));
    CHECK_CUDA_ERR(cudaFuncSetAttribute(fpv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 68608));
    CHECK_CUDA_ERR(cudaFuncSetAttribute(mlf_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, min_lift_traits::dynamic_shmem));
    CHECK_CUDA_ERR(cudaFuncSetAttribute(v2h_kernel[tid], cudaFuncAttributeMaxDynamicSharedMemorySize, v2h_shmem));

    pthread_spin_lock(&gram_lock);
    used_gram[device_ptr] += _used_gram;
    pthread_spin_unlock(&gram_lock);
    if (used_gram[device_ptr] > DHR_GRAM_SLIMIT) {
        lg_err("device %d used %ld bytes of GPU memory, exceeds the slimit %ld", hw::gpu_id_list[device_ptr], used_gram[device_ptr], DHR_GRAM_SLIMIT);
    }

    return 0;
}

int dhr_buffer_t::device_done(int tid) {
    int device_ptr = hw::gpu_ptr(tid, num_threads);
    CHECK_CUDA_ERR(cudaStreamDestroy(streams[tid]));

    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventDestroy(logger->h2d_start[tid]));
    CHECK_CUDA_ERR(cudaEventDestroy(logger->h2d_stop[tid]));
    CHECK_CUDA_ERR(cudaEventDestroy(logger->dhr_start[tid]));
    CHECK_CUDA_ERR(cudaEventDestroy(logger->dhr_stop[tid]));
    CHECK_CUDA_ERR(cudaEventDestroy(logger->v2h_start[tid]));
    CHECK_CUDA_ERR(cudaEventDestroy(logger->v2h_stop[tid]));
    CHECK_CUDA_ERR(cudaEventDestroy(logger->mlf_start[tid]));
    CHECK_CUDA_ERR(cudaEventDestroy(logger->mlf_stop[tid]));
    #endif

    long _used_gram = 0;

    CHECK_CUDA_ERR(cudaFree(d_upk[tid]));
    CHECK_CUDA_ERR(cudaFree(d_vec16[tid]));
    CHECK_CUDA_ERR(cudaFree(d_dh[tid]));
    CHECK_CUDA_ERR(cudaFree(d_num_out[tid]));
    CHECK_CUDA_ERR(cudaFree(d_out[tid]));
    CHECK_CUDA_ERR(cudaFree(local_data[tid]));
    _used_gram += traits::taskVecs * CSD16 + buc_max_size * (CSD16 + dh_nbits / 8) + out_max_size * 8 + sizeof(local_data_t);
    if (tid == 0 || hw::gpu_ptr(tid - 1, num_threads) != device_ptr) {
        CHECK_CUDA_ERR(cudaFree(d_data[device_ptr]));
        _used_gram += sizeof(dh_data_t);
    }

    pthread_spin_lock(&gram_lock);
    used_gram[device_ptr] -= _used_gram;
    pthread_spin_unlock(&gram_lock);

    return 0;
}

int dhr_buffer_t::h2d(int tid, chunk_t *chunk, int &used) {
    int buc_full = 0;
    int to_copy = chunk->size - used < traits::taskVecs - task_vecs[tid] ?
                  chunk->size - used : traits::taskVecs - task_vecs[tid];
    if (to_copy + task_vecs[tid] + buc_vecs[tid] >= buc_max_size) {
        to_copy = buc_max_size - buc_vecs[tid] - task_vecs[tid];
        if (to_copy < 0) to_copy = 0;
        buc_full = 1;
    }
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->h2d_start[tid], streams[tid]));
    #endif
    if (to_copy)
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_upk[tid] + task_vecs[tid] * CSD, chunk->vec + used * CSD, 
                                   to_copy * CSD, cudaMemcpyHostToDevice, streams[tid]));
    task_vecs[tid] += to_copy;
    used = buc_full ? -1 : used + to_copy;
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->h2d_stop[tid], streams[tid]));
    #endif

    CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));
    #if ENABLE_PROFILING
    {
        float tt;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&tt, logger->h2d_start[tid], logger->h2d_stop[tid]));
        logger->ev_h2d_us += 1000.f * tt;
    }
    #endif

    return to_copy;
}

int dhr_buffer_t::upk(int tid) {
    int device_ptr = hw::gpu_ptr(tid, num_threads);
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->v2h_start[tid], streams[tid]));
    #endif
    v2h_kernel[tid]<<<v2h_blocks, v2h_threads, v2h_shmem, streams[tid]>>>(
        (uint32_t *)(d_dh[tid] + (long)dh_nbits / 32 * (long)buc_vecs[tid]), d_upk[tid], d_vec16[tid] + buc_vecs[tid] * CSD16, task_vecs[tid], CSD, d_data[device_ptr]
    );
    CHECK_LAST_ERR;
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->v2h_stop[tid], streams[tid]));
    #endif
    CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));
    #if ENABLE_PROFILING
    {
        float v2h_tt;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&v2h_tt, logger->v2h_start[tid], logger->v2h_stop[tid]));
        logger->ev_v2h_us += 1000.f * v2h_tt;
    }
    #endif

    buc_vecs[tid] += task_vecs[tid];
    task_vecs[tid] = 0;

    return 0;
}

#if SPLIT_DHR
int dhr_buffer_t::run(int tid) {
    long N = (buc_vecs[tid] / 64) * 64;

    for (long bias = 0; bias < N; bias += 262144) {
        /// call dhr kernel
        #if ENABLE_PROFILING
        CHECK_CUDA_ERR(cudaEventRecord(logger->dhr_start[tid], streams[tid]));
        #endif
        CHECK_CUDA_ERR(cudaMemsetAsync(d_num_out[tid], 0, sizeof(int), streams[tid]));
        dhr_kernel<<<dhr_blocks, dhr_threads, dhr_shmem, streams[tid]>>>(
            d_out[tid], d_num_out[tid], out_max_size, d_dh[tid] + bias * (dh_nbits / 32), N - bias, th
        );
        #if ENABLE_PROFILING
        CHECK_CUDA_ERR(cudaEventRecord(logger->dhr_stop[tid], streams[tid]));
        #endif

        CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));
        
        /// check overflow
        int h_num_out;
        CHECK_CUDA_ERR(cudaMemcpyAsync(&h_num_out, d_num_out[tid], sizeof(int), cudaMemcpyDeviceToHost, streams[tid]));
        if (h_num_out > out_max_size) {
            lg_warn("thread %d, bias %ld, #out %d overflow(%d)", tid, bias, h_num_out, out_max_size);
        }
        h_num_out = h_num_out < out_max_size ? h_num_out : out_max_size;

        /// call mlf kernel
        #if ENABLE_PROFILING
        CHECK_CUDA_ERR(cudaEventRecord(logger->mlf_start[tid], streams[tid]));
        #endif
        for (int i = 0; i < h_num_out; i += traits::taskVecs) {
            int batch_num = h_num_out - i < traits::taskVecs ? h_num_out - i : traits::taskVecs;
            fpv_kernel<<<fpv_blocks, fpv_threads, fpv_shmem, streams[tid]>>>(d_upk[tid], d_vec16[tid] + bias * CSD16, d_out[tid] + 2 * i, batch_num);
            mlf_kernel<<<min_lift_traits::kernelBlocks, min_lift_traits::blockThreads, min_lift_traits::dynamic_shmem, streams[tid]>>>(
                d_upk[tid], batch_num, local_data[tid]
            );
        }
        #if ENABLE_PROFILING
        CHECK_CUDA_ERR(cudaEventRecord(logger->mlf_stop[tid], streams[tid]));
        CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));
        float dhr_tt, mlf_tt;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&dhr_tt, logger->dhr_start[tid], logger->dhr_stop[tid]));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&mlf_tt, logger->mlf_start[tid], logger->mlf_stop[tid]));
        logger->ev_dhr_us += 1000.f * dhr_tt;
        logger->ev_mlf_us += 1000.f * mlf_tt;
        logger->ev_mlf_256ops += ceil(h_num_out / 256.0);
        #endif
    }
    
    
    #if ENABLE_PROFILING
    logger->ev_dhb_num    += 1;
    logger->ev_dhb_ssum   += buc_vecs[tid];
    logger->ev_h2d_nbytes += buc_vecs[tid] * CSD;
    logger->ev_v2h_256ops += ceil(buc_vecs[tid] / 256.0);
    logger->ev_dhr_256ops += ceil(buc_vecs[tid] / 16.0) * ceil(buc_vecs[tid] / 16.0 + 1) * 0.5;
    #endif

    buc_vecs[tid] = 0;

    return 0;
}

int dhr_buffer_t::out(int tid) {   
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_res[tid], local_data[tid] + 1, ESD8 * (1024 + 4 + 4), cudaMemcpyDeviceToHost, streams[tid]));
    CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));

    MAT_QP b_QP = reducer->_pool->basis->get_b();

    for (int i = 0; i < ESD; i++) {
        int all_zero = 1;
        int same = 1;
        for (int j = 0; j < 256; j++) if (h_res[tid][ESD8 * 2 + i * 256 + j]) all_zero = 0;
        for (int j = 0; j < 256; j++) if (h_res[tid][ESD8 * 2 + i * 256 + j] != res[ESD8 * 2 + i * 256 + j]) same = 0;
        if (all_zero || same) continue;
        if (((float *)h_res[tid])[i] > 0.999997f * ((float *)res)[i]) continue;

        Lattice_QP *basis = reducer->_pool->basis;
        int pos = reducer->_pool->index_l - reducer->_pool->ESD + i;
        int *coeff = &h_res[tid][ESD8 * 2 + i * 256];
        int has_one = 0;
        for (int j = ESD8; j < ESD8 + CSD; j++) if (coeff[j] == 1 || coeff[j] == -1) has_one = 1;
        for (int j = 0; j < basis->NumCols(); j++) {
            v_QP[tid].hi[j] = 0.0;
            v_QP[tid].lo[j] = 0.0;
        }
        for (int j = 0; j < ESD; j++) {
            red(v_QP[tid].hi, v_QP[tid].lo, b_trans_QP.hi[j], b_trans_QP.lo[j], NTL::quad_float(-coeff[j]), basis->NumCols());
        }
        for (int j = 0; j < CSD; j++) {
            red(v_QP[tid].hi, v_QP[tid].lo, b_trans_QP.hi[ESD+j], b_trans_QP.lo[ESD+j], NTL::quad_float(-coeff[ESD8 + j]), basis->NumCols());
        }

        for (long j = reducer->_pool->index_l - ESD - 1; j >= 0; j--) {
            int32_t c = round(dot_avx2(v_QP[tid].hi, basis->get_b_star().hi[j], basis->NumCols()) / basis->get_B().hi[j]);
            red(v_QP[tid].hi, v_QP[tid].lo, b_QP.hi[j], b_QP.lo[j], NTL::quad_float(c), basis->NumCols());
        }

        NTL::quad_float proj2 = dot(v_QP[tid].hi, v_QP[tid].lo, v_QP[tid].hi, v_QP[tid].lo, basis->NumCols());

        for (int j = 0; j < reducer->_pool->index_l - ESD + i; j++) {
            NTL::quad_float _proj = dot(v_QP[tid].hi, v_QP[tid].lo, basis->get_b_star().hi[j], basis->get_b_star().lo[j], basis->NumCols());
            NTL::quad_float _B = NTL::quad_float(basis->get_B().hi[j], basis->get_B().lo[j]);
            proj2 -= _proj * _proj / _B;
        }

        double scaled_res = proj2.hi * reducer->_pool->_ratio * reducer->_pool->_ratio;

        int updated = 0;
        pthread_spin_lock(&min_lock);
        if (scaled_res < 0.999997f * ((float *)res)[i]) {
            ((float *)res)[i] = scaled_res;
            int same = 1;
            for (int j = 0; j < 256; j++) if (res[ESD8 * 2 + i * 256 + j] != coeff[j]) { same = 0; break; }
            if (!same) {
                updated = 1;
                memcpy(&res[ESD8 * 2 + i * 256], coeff, 256 * sizeof(int));
            }
        }
        pthread_spin_unlock(&min_lock);

        if (i >= report_range) continue;
        if (!updated) continue;
        if (reducer->_force_one) continue;
        
        double _ratio = reducer->_pool->_ratio;
        if (target_length != 0.0 && scaled_res * pow(reducer->_eta, i) > 
           (target_length * _ratio) * (target_length * _ratio)) continue;

        char tmp[4096];
        int pp = 0;
        pp += sprintf(tmp + pp, "%s[pos %d] length = %.2f(%.3f gh, %.3f old)\033[0m, vec = [", has_one ? "" : "\033[33m", 
                pos, sqrt(scaled_res) / _ratio, sqrt(scaled_res) / _ratio / basis->gh(pos, reducer->_pool->index_r), 
                sqrt(scaled_res) / reducer->_pool->_boost_data->evec[i * ESD + i]);

        for (int j = 0; j < basis->NumCols(); j++) {
            pp += sprintf(tmp + pp, "%g ", v_QP[tid].hi[j]);
        }

        pp += sprintf(tmp + pp, "\b]\n");
        lg_info("%s", tmp);
    }

    int to_report = 0;
    struct timeval curr;
    gettimeofday(&curr, NULL);
    pthread_spin_lock(&min_lock);
    if ((curr.tv_sec - last_report.tv_sec) + 1e-6 * (curr.tv_usec - last_report.tv_usec) > DH_REPORT_DURATION) {
        last_report = curr;
        to_report = 1;
    }
    pthread_spin_unlock(&min_lock);

    if (to_report) {
        char tmp[4096];
        int pp = 0;
        pp += sprintf(tmp + pp, "[");
        for (int i = 0; i < report_range && i < ESD; i++) {
            float old = reducer->_pool->_boost_data->evec[i * ESD + i];
            float gh = sqrt(reducer->_pool->gh2_scaled() / reducer->_pool->_boost_data->igh[i]);
            float curr = sqrt(((float *)res)[i]);
            pp += sprintf(tmp + pp, "%.4f(%.4f) ", curr / gh, curr / old);
        }
        pp += sprintf(tmp + pp, "\b]");
        lg_dbg("%s", tmp);
    }

    return 0;
}
#else
int dhr_buffer_t::run(int tid) {
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->dhr_start[tid], streams[tid]));
    #endif
    dhr_kernel<<<dhr_blocks, dhr_threads, dhr_shmem, streams[tid]>>>(
        d_out[tid], d_num_out[tid], out_max_size, d_dh[tid], (buc_vecs[tid] / 64) * 64, th
    );
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->dhr_stop[tid], streams[tid]));
    logger->ev_dhb_num    += 1;
    logger->ev_dhb_ssum   += buc_vecs[tid];
    logger->ev_h2d_nbytes += buc_vecs[tid] * CSD;
    logger->ev_v2h_256ops += ceil(buc_vecs[tid] / 256.0);
    logger->ev_dhr_256ops += ceil(buc_vecs[tid] / 16.0) * ceil(buc_vecs[tid] / 16.0 + 1) * 0.5;
    #endif

    buc_vecs[tid] = 0;

    return 0;
}

int dhr_buffer_t::out(int tid) {
    int h_num_out;
    CHECK_CUDA_ERR(cudaMemcpyAsync(&h_num_out, d_num_out[tid], sizeof(int), cudaMemcpyDeviceToHost, streams[tid]));
    CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));
    if (h_num_out > out_max_size) {
        lg_warn("thread %d, #out %d overflow(%d)", tid, h_num_out, out_max_size);
    }
    h_num_out = h_num_out < out_max_size ? h_num_out : out_max_size;
    CHECK_CUDA_ERR(cudaMemsetAsync(d_num_out[tid], 0, sizeof(int), streams[tid]));

    #if ENABLE_PROFILING
    {
        float dhr_tt;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&dhr_tt, logger->dhr_start[tid], logger->dhr_stop[tid]));
        logger->ev_dhr_us += 1000.f * dhr_tt;
        logger->ev_mlf_256ops += ceil(h_num_out / 256.0);
    }
    CHECK_CUDA_ERR(cudaEventRecord(logger->mlf_start[tid], streams[tid]));
    #endif
    

    for (int i = 0; i < h_num_out; i += traits::taskVecs) {
        int batch_num = h_num_out - i < traits::taskVecs ? h_num_out - i : traits::taskVecs;
        fpv_kernel<<<fpv_blocks, fpv_threads, fpv_shmem, streams[tid]>>>(d_upk[tid], d_vec16[tid], d_out[tid] + 2 * i, batch_num);
        mlf_kernel<<<min_lift_traits::kernelBlocks, min_lift_traits::blockThreads, min_lift_traits::dynamic_shmem, streams[tid]>>>(
            d_upk[tid], batch_num, local_data[tid]
        );
    }
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->mlf_stop[tid], streams[tid]));
    #endif
    
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_res[tid], local_data[tid] + 1, ESD8 * (1024 + 4 + 4), cudaMemcpyDeviceToHost, streams[tid]));
    CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));

    #if ENABLE_PROFILING
    {
        float mlf_tt;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&mlf_tt, logger->mlf_start[tid], logger->mlf_stop[tid]));
        logger->ev_mlf_us += 1000.f * mlf_tt;
    }
    #endif

    MAT_QP b_QP = reducer->_pool->basis->get_b();

    for (int i = 0; i < ESD; i++) {
        int all_zero = 1;
        int same = 1;
        for (int j = 0; j < 256; j++) if (h_res[tid][ESD8 * 2 + i * 256 + j]) all_zero = 0;
        for (int j = 0; j < 256; j++) if (h_res[tid][ESD8 * 2 + i * 256 + j] != res[ESD8 * 2 + i * 256 + j]) same = 0;
        if (all_zero || same) continue;
        if (((float *)h_res[tid])[i] > 0.999997f * ((float *)res)[i]) continue;

        Lattice_QP *basis = reducer->_pool->basis;
        int pos = reducer->_pool->index_l - reducer->_pool->ESD + i;
        int *coeff = &h_res[tid][ESD8 * 2 + i * 256];
        int has_one = 0;
        for (int j = ESD8; j < ESD8 + CSD; j++) if (coeff[j] == 1 || coeff[j] == -1) has_one = 1;
        for (int j = 0; j < basis->NumCols(); j++) {
            v_QP[tid].hi[j] = 0.0;
            v_QP[tid].lo[j] = 0.0;
        }
        for (int j = 0; j < ESD; j++) {
            red(v_QP[tid].hi, v_QP[tid].lo, b_trans_QP.hi[j], b_trans_QP.lo[j], NTL::quad_float(-coeff[j]), basis->NumCols());
        }
        for (int j = 0; j < CSD; j++) {
            red(v_QP[tid].hi, v_QP[tid].lo, b_trans_QP.hi[ESD+j], b_trans_QP.lo[ESD+j], NTL::quad_float(-coeff[ESD8 + j]), basis->NumCols());
        }

        for (long j = reducer->_pool->index_l - ESD - 1; j >= 0; j--) {
            int32_t c = round(dot_avx2(v_QP[tid].hi, basis->get_b_star().hi[j], basis->NumCols()) / basis->get_B().hi[j]);
            red(v_QP[tid].hi, v_QP[tid].lo, b_QP.hi[j], b_QP.lo[j], NTL::quad_float(c), basis->NumCols());
        }

        NTL::quad_float proj2 = dot(v_QP[tid].hi, v_QP[tid].lo, v_QP[tid].hi, v_QP[tid].lo, basis->NumCols());

        for (int j = 0; j < reducer->_pool->index_l - ESD + i; j++) {
            NTL::quad_float _proj = dot(v_QP[tid].hi, v_QP[tid].lo, basis->get_b_star().hi[j], basis->get_b_star().lo[j], basis->NumCols());
            NTL::quad_float _B = NTL::quad_float(basis->get_B().hi[j], basis->get_B().lo[j]);
            proj2 -= _proj * _proj / _B;
        }

        double scaled_res = proj2.hi * reducer->_pool->_ratio * reducer->_pool->_ratio;

        int updated = 0;
        pthread_spin_lock(&min_lock);
        if (scaled_res < 0.999997f * ((float *)res)[i]) {
            ((float *)res)[i] = scaled_res;
            int same = 1;
            for (int j = 0; j < 256; j++) if (res[ESD8 * 2 + i * 256 + j] != coeff[j]) { same = 0; break; }
            if (!same) {
                updated = 1;
                memcpy(&res[ESD8 * 2 + i * 256], coeff, 256 * sizeof(int));
            }
        }
        pthread_spin_unlock(&min_lock);

        if (i >= report_range) continue;
        if (!updated) continue;
        if (reducer->_force_one) continue;
        
        double _ratio = reducer->_pool->_ratio;
        if (target_length != 0.0 && scaled_res * pow(reducer->_eta, i) > 
           (target_length * _ratio) * (target_length * _ratio)) continue;

        char tmp[4096];
        int pp = 0;
        pp += sprintf(tmp + pp, "%s[pos %d] length = %.2f(%.3f gh, %.3f old)\033[0m, vec = [", has_one ? "" : "\033[33m", 
                pos, sqrt(scaled_res) / _ratio, sqrt(scaled_res) / _ratio / basis->gh(pos, reducer->_pool->index_r), 
                sqrt(scaled_res) / reducer->_pool->_boost_data->evec[i * ESD + i]);

        for (int j = 0; j < basis->NumCols(); j++) {
            pp += sprintf(tmp + pp, "%g ", v_QP[tid].hi[j]);
        }

        pp += sprintf(tmp + pp, "\b]\n");
        lg_info("%s", tmp);
    }

    int to_report = 0;
    struct timeval curr;
    gettimeofday(&curr, NULL);
    pthread_spin_lock(&min_lock);
    if ((curr.tv_sec - last_report.tv_sec) + 1e-6 * (curr.tv_usec - last_report.tv_usec) > DH_REPORT_DURATION) {
        last_report = curr;
        to_report = 1;
    }
    pthread_spin_unlock(&min_lock);

    if (to_report) {
        char tmp[4096];
        int pp = 0;
        pp += sprintf(tmp + pp, "[");
        for (int i = 0; i < report_range && i < ESD; i++) {
            float old = reducer->_pool->_boost_data->evec[i * ESD + i];
            float gh = sqrt(reducer->_pool->gh2_scaled() / reducer->_pool->_boost_data->igh[i]);
            float curr = sqrt(((float *)res)[i]);
            pp += sprintf(tmp + pp, "%.4f(%.4f) ", curr / gh, curr / old);
        }
        pp += sprintf(tmp + pp, "\b]");
        lg_dbg("%s", tmp);
    }

    return 0;
}
#endif



dh_bucketer_t::dh_bucketer_t(Pool_hd_t *pool, bwc_manager_t *bwc) {
    this->_pool = pool;
    this->_pwc = pool->pwc_manager;
    this->_bwc = bwc;

    #if ENABLE_PROFILING
    logger = new logger_t();
    logger->bucketer = this;
    logger->clear();
    #endif
}

dh_bucketer_t::~dh_bucketer_t() {
    if (_buc_pool) {
        for (int i = 0; i < _num_threads; i++) delete _buc_pool[i];
        free(_buc_pool);
        _buc_pool = NULL;
    }
    if (_buc_buf) {
        delete _buc_buf;
        _buc_buf = NULL;
    }
    #if ENABLE_PROFILING
    delete this->logger;
    #endif
}

int dh_bucketer_t::set_num_threads(int num_threads) {
    if (_buc_pool) {
        for (int i = 0; i < this->_num_threads; i++) delete _buc_pool[i];
        free(_buc_pool);
    }
    this->_num_threads = num_threads;
    
    _buc_pool = (thread_pool::thread_pool **)malloc(num_threads * sizeof(thread_pool::thread_pool *));
    for (int i = 0; i < num_threads; i++) _buc_pool[i] = new thread_pool::thread_pool(1);

    return 0;
}

int dh_bucketer_t::set_beta(double beta) {
    this->_beta = beta;
    return 0;
}

int dh_bucketer_t::set_min_batch(long min_batch) {
    this->_min_batch = min_batch;
    return 0;
}

int dh_bucketer_t::set_max_batch(long max_batch) {
    this->_max_batch = max_batch;
    return 0;
}

int dh_bucketer_t::set_num_buc_slimit(long num_buc_slimit) {
    this->_num_buc_slimit = num_buc_slimit;
    return 0;
}

int dh_bucketer_t::auto_bgj_params_set() {
    int ESD = _pool->ESD;
    double pool_size = _pool->pwc_manager->num_vec();
    double expect_buc_size = DH_BSIZE_RATIO * sqrt(pool_size);
    double expect_buc_ratio = expect_buc_size / pool_size;
    this->_beta = pow(expect_buc_ratio, 1.0 / _pool->ESD);
    if (this->_beta > 0.95) this->_beta = 0.95;
    
    if (!this->_num_buc_slimit) this->_num_buc_slimit = (double)BWC_SSD_SLIMIT / (expect_buc_size * 190.0);
    if (this->_num_buc_slimit > BWC_MAX_BUCKETS) this->_num_buc_slimit = BWC_MAX_BUCKETS;
    if (_num_buc_slimit > bwc_manager_t::bwc_max_buckets - DHR_DEFAULT_NUM_THREADS) 
        _num_buc_slimit = bwc_manager_t::bwc_max_buckets - DHR_DEFAULT_NUM_THREADS;

    const int cache_for_prefetch =  bwc_manager_t::bwc_auto_prefetch_for_read * 
                                    bwc_manager_t::bwc_auto_prefetch_for_read_depth + 
                                    bwc_manager_t::bwc_auto_prefetch_for_write;
    const int cache_for_bucketer = _bwc->max_cached_chunks() - cache_for_prefetch;
    int expect_max_batch = traits::max_batch_under(cache_for_bucketer < DH_MAX_BATCH ? cache_for_bucketer : DH_MAX_BATCH);
    if (_pool->ESD > 40 && expect_max_batch > 1024) expect_max_batch = 1024;
    if (!this->_max_batch) this->_max_batch = expect_max_batch < _num_buc_slimit ? 
                                              expect_max_batch : traits::max_batch_under(_num_buc_slimit);
    if (!this->_min_batch) {
        this->_min_batch = traits::max_batch_under(0.36 * _num_buc_slimit);
        if (this->_min_batch < DH_MIN_BATCH) this->_min_batch = DH_MIN_BATCH;
        if (this->_min_batch > this->_max_batch) this->_min_batch = this->_max_batch;
    }

    if (!this->_num_threads) this->set_num_threads(traits::buc_num_threads(_pool->CSD));

    lg_dbg("#thread %ld, beta %.3f, batch in [%ld, %ld], #buc <= %d(%.2f TB, %.2f GB)", 
            _num_threads, _beta, _min_batch, _max_batch, _num_buc_slimit, BWC_SSD_SLIMIT / 1e12, BWC_SSD_SLIMIT / 1e9 / _num_buc_slimit);

    return 0;
}

int dh_bucketer_t::run(double max_time) {
    constexpr long chunk_max_nvecs = Pool_hd_t::chunk_max_nvecs;
    constexpr int taskChunks = dh_traits_t::taskChunks;

    struct timeval start, curr, first_batch_done;
    gettimeofday(&start, NULL);

    uint64_t total_generated = 0;

    /// initialize device data
    if (_buc_buf) delete _buc_buf;
    _buc_buf = new dhb_buffer_t(this);

    for (int tid = 0; tid < _num_threads; tid++) {
        _buc_pool[tid]->push([this, tid] { _buc_buf->device_init(tid); });
    }
    for (int tid = 0; tid < _num_threads; tid++) {
        _buc_pool[tid]->wait_sleep();
    }

    #if ENABLE_PROFILING
    if (logger->_ll >= logger_t::ll_info) {
        char tmp[4096];
        int pp = 0;
        for (int i = 0; i < _buc_buf->num_devices; i++) {
            pp += sprintf(tmp + pp, "%.2f ", _buc_buf->used_gram[i] / 1e9);
        }
        pp += sprintf(tmp + pp, "\b");
        lg_dbg("device init done, used gram: %s, pageable RAM %.2f GB, pinned RAM %.2f GB", tmp, 
                _buc_buf->pageable_ram.load() / 1e9, _buc_buf->pinned_ram.load() / 1e9);
    }
    #endif

    /// prepare runtime data
    const int CSD = _pool->CSD;
    pthread_spinlock_t task_lock, buc_write_lock;
    pthread_spin_init(&task_lock, PTHREAD_PROCESS_SHARED);
    pthread_spin_init(&buc_write_lock, PTHREAD_PROCESS_SHARED);
    int32_t *buc_id = (int32_t *) malloc(_max_batch * sizeof(int32_t));

    for (;;) {
        int batch = -1;

        std::unique_lock<std::mutex> dhb_lock(_dhb_mtx);
        _dhb_cv.wait(dhb_lock, [&] {
            long num_ready = _bwc->num_ready();
            if (max_time != 0.0 && total_generated) {
                gettimeofday(&curr, NULL);
                double elapsed = (curr.tv_sec - start.tv_sec) + (curr.tv_usec - start.tv_usec) * 1e-6;
                double red_time = (curr.tv_sec - first_batch_done.tv_sec) + (curr.tv_usec - first_batch_done.tv_usec) * 1e-6;
                if (elapsed > 0.7 * max_time) {
                    double remain_time = max_time - elapsed;
                    double expect_bucs = remain_time / red_time * (total_generated - num_ready);
                    if (expect_bucs < 1.2 * num_ready) return true;
                }
            }
            
            if (_num_buc_slimit - num_ready < _min_batch) return false;
            batch = traits::max_batch_under(_num_buc_slimit - num_ready);
            if (batch > _max_batch) batch = _max_batch;
            return true;
        });
        dhb_lock.unlock();

        if (batch == -1) {
            std::unique_lock<std::mutex> lock(_reducer->_dhr_mtx);
            _reducer->flag = dh_reducer_t::flag_stop;
            _reducer->_dhr_cv.notify_all();
            break;
        }

        #if ENABLE_PROFILING
        struct timeval batch_start;
        gettimeofday(&batch_start, NULL);
        #endif

        for (int i = 0; i < batch; i++) buc_id[i] = _bwc->push_bucket();

        _buc_buf->center_prep(batch);

        lg_dbg("new batch (%d) started, current #buc: %ld / %ld", batch, _bwc->num_ready(), _num_buc_slimit);

        int rem_chunks = _pwc->num_chunks();

        for (int tid = 0; tid < _num_threads; tid++) {
            _buc_pool[tid]->push([&, tid] {
                for (;;) {
                    int task_chunks = 0, task_vecs = 0;
                    chunk_t *working_chunk[taskChunks] = {};
                    int32_t chunk_modified[taskChunks] = {};

                    int task = -1;
                    pthread_spin_lock(&task_lock);
                    task_chunks = rem_chunks > taskChunks ? taskChunks : rem_chunks;
                    rem_chunks -= task_chunks;
                    task = rem_chunks;
                    pthread_spin_unlock(&task_lock);

                    for (int i = 0; i < task_chunks; i++) {
                        #if ENABLE_PROFILING
                        struct timeval fetch_start, fetch_end;
                        gettimeofday(&fetch_start, NULL);
                        #endif
                        working_chunk[i] = _pwc->fetch(task + i);
                        #if ENABLE_PROFILING
                        gettimeofday(&fetch_end, NULL);
                        logger->ev_ld_stall_us += (fetch_end.tv_sec - fetch_start.tv_sec) * 1000000 + fetch_end.tv_usec - fetch_start.tv_usec;
                        #endif
                        if (!working_chunk[i]) {
                            lg_err("fetch chunk %d returned NULL", task + i);
                        } else {
                            chunk_modified[i] = _normalize_chunk(working_chunk[i], _pool->CSD);
                            _buc_buf->h2d(tid, working_chunk[i]);
                        }
                    }

                    for (int i = 0; i < task_chunks; i++) {
                        _pwc->prefetch(task + i - traits::pfch_ahead);
                    }

                    _buc_buf->run(tid);

                    int32_t working_chunk_size[taskChunks] = {};
                    for (int i = 0; i < task_chunks; i++) {
                        working_chunk_size[i] = working_chunk[i] ? working_chunk[i]->size : 0;
                        task_vecs += working_chunk_size[i];
                    }

                    random_interval_iter_t iiter(batch);

                    for (int _i = 0; _i < batch; _i++) {
                        int i = iiter.pop();
                        int to_add, *entry;
                        _buc_buf->out(tid, i, &to_add, &entry);
                        while (to_add > 0) {
                            chunk_t *_dst = _bwc->fetch_for_write(buc_id[i]);
                            if (!_dst) lg_err("tid = %d, bucket %d(%d-th this batch) fetch_for_write failed, "
                                            "%d entries ignored", tid, buc_id[i], i, to_add);
                            int to_move = to_add < chunk_max_nvecs - _dst->size ? to_add : chunk_max_nvecs - _dst->size;
                            for (int j = 0; j < to_move; j++) {
                                int pos = entry[to_add - 1 - j] >> 1;
                                int sign = entry[to_add - 1 - j] & 1;
                                if (pos >= task_vecs) lg_err("tid = %d, entry[%d] = %d of %d-th bucket out of range (%d)",
                                                            tid, to_add - 1 - j, entry[to_add - 1 - j], i, task_vecs);
                                int working_chunk_id = 0;
                                /// @todo may be copy every thing to a large array could be better?
                                while (pos >= working_chunk_size[working_chunk_id]) {
                                    pos -= working_chunk_size[working_chunk_id];
                                    working_chunk_id++;
                                }
                                chunk_t *_src = working_chunk[working_chunk_id];
                                _dst->u[_dst->size + j] = sign ? -_src->u[pos] : _src->u[pos];
                                _dst->score[_dst->size + j] = _src->score[pos];
                                _dst->norm[_dst->size + j] = _src->norm[pos];
                                traits::sign_copy_epi8(_dst->vec + CSD * (_dst->size + j), _src->vec + CSD * pos, CSD, sign);
                            }
                            to_add -= to_move;
                            _dst->size += to_move;
                            _bwc->write_done(_dst, buc_id[i]);
                        }
                    }

                    for (int i = 0; i < task_chunks; i++) {
                        if (!working_chunk[i]) continue;
                        if (chunk_modified[i]) _pwc->release_sync(working_chunk[i]->id);
                        else _pwc->release(working_chunk[i]->id);
                    }

                    if (rem_chunks == 0) break;
                }
            });
        }

        for (int tid = 0; tid < _num_threads; tid++) {
            _buc_pool[tid]->wait_sleep();
        }

        #if ENABLE_PROFILING
        struct timeval batch_end;
        gettimeofday(&batch_end, NULL);
        logger->ev_total_batch++;
        logger->ev_total_batch_us += (batch_end.tv_sec - batch_start.tv_sec) * 1000000UL + batch_end.tv_usec - batch_start.tv_usec;
        logger->ev_total_bucket += batch;
        #endif

        if (init_round) {
            int ESD8 = _buc_buf->ESD8;
            int *bres = (int *)malloc(ESD8 * (1024 + 4 + 4));
            int *tres = (int *)malloc(ESD8 * (1024 + 4 + 4));
            memset(bres, 0, ESD8 * (1024 + 4 + 4));
            for (int i = 0; i < ESD8; i++) ((float *)bres)[i] = 0x1p40f;

            for (int d = 0; d < hw::gpu_num; d++) {
                CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[d]));
                CHECK_CUDA_ERR(cudaMemcpy(tres, _buc_buf->bml_data[d] + 1, ESD8 * (1024 + 4 + 4), cudaMemcpyDeviceToHost));
                for (int i = 0; i < ESD8; i++) {
                    if (((float *)tres)[i] < ((float *)bres)[i]) {
                        ((float *)bres)[i] = ((float *)tres)[i];
                        memcpy(bres + ESD8 * 2 + i * 256, tres + ESD8 * 2 + i * 256, 1024);
                    }
                }
            }

            while (!_reducer->device_inited) {}

            {   
                int ESD = _reducer->_red_buf->ESD;
                char tmp[4096];
                int pp = 0;
                pp += sprintf(tmp + pp, "[");
                for (int i = 0; i < _reducer->_red_buf->report_range && i < ESD; i++) {
                    float old = _pool->_boost_data->evec[i * ESD + i];
                    float gh = sqrt(_pool->gh2_scaled() / _pool->_boost_data->igh[i]);
                    float curr = sqrt(((float *)bres)[i]);
                    pp += sprintf(tmp + pp, "%.4f(%.4f) ", curr / gh, curr / old);
                }
                pp += sprintf(tmp + pp, "\b]");
                lg_dbg("%s", tmp);
            }

            for (int t = 0; t < _reducer->_num_threads; t++) {
                int device_ptr = hw::gpu_ptr(t, _reducer->_num_threads);
                CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[device_ptr]));
                CHECK_CUDA_ERR(cudaMemcpy(_reducer->_red_buf->local_data[t] + 1, bres, ESD8 * (1024 + 4 + 4), cudaMemcpyHostToDevice));
            }

            free(bres);
            free(tres);
        }
        if (init_round) init_round = 0;
        else _reducer->_red_buf->last_report.tv_sec -= DH_REPORT_DURATION;

        lg_dbg("batch done, current #buc: %ld / %ld", _bwc->num_ready(), _num_buc_slimit);
        
        for (int i = 0; i < batch; i++) _bwc->bucket_finalize(buc_id[i]);

        {
            std::unique_lock<std::mutex> lock(_reducer->_dhr_mtx);
            _reducer->_dhr_cv.notify_all();
        }

        if (!total_generated) gettimeofday(&first_batch_done, NULL);
        total_generated += batch;
    }

    lg_dbg("no more bucket, waiting for reducer done");
    
    /// destroy runtime data
    pthread_spin_destroy(&task_lock);
    pthread_spin_destroy(&buc_write_lock);
    free(buc_id);

    /// destroy device data
    for (int tid = 0; tid < _num_threads; tid++) {
        _buc_pool[tid]->push([this, tid] { _buc_buf->device_done(tid); });
    }
    for (int tid = 0; tid < _num_threads; tid++) {
        _buc_pool[tid]->wait_sleep();
    }

    lg_report();

    return 0;
}


dh_reducer_t::dh_reducer_t(Pool_hd_t *pool, bwc_manager_t *bwc, double eta, int force_one) {
    #if ENABLE_PROFILING
    logger = new logger_t();
    logger->clear();
    #endif

    this->_pool = pool;
    this->_bwc = bwc;
    this->_eta = eta;
    this->_force_one = force_one;
}

dh_reducer_t::~dh_reducer_t() {
    if (_red_buf) {
        delete _red_buf;
        _red_buf = NULL;
    }
    if (_red_pool) {
        for (int i = 0; i < _num_threads; i++) delete _red_pool[i];
        free(_red_pool);
        _red_pool = NULL;
    }
    #if ENABLE_PROFILING
    delete this->logger;
    #endif
}

int dh_reducer_t::set_num_threads(int num_threads) {
    if (_red_pool) {
        for (int i = 0; i < this->_num_threads; i++) delete _red_pool[i];
        free(_red_pool);
    }
    this->_num_threads = num_threads;
    
    _red_pool = (thread_pool::thread_pool **)malloc(num_threads * sizeof(thread_pool::thread_pool *));
    for (int i = 0; i < num_threads; i++) _red_pool[i] = new thread_pool::thread_pool(1);

    return 0;
}

int dh_reducer_t::auto_bgj_params_set() {
    if (!this->_num_threads) this->set_num_threads(traits::red_num_threads(_pool->CSD, _pool->ESD));

    return 0;
}

int dh_reducer_t::run(double target_length) {
    if (_red_buf) delete _red_buf;
    _red_buf = new dhr_buffer_t(this, target_length);

    for (int tid = 0; tid < _num_threads; tid++) {
        _red_pool[tid]->push([this, tid] { _red_buf->device_init(tid); });
    }
    for (int tid = 0; tid < _num_threads; tid++) {
        _red_pool[tid]->wait_sleep();
    }
    device_inited = 1;

    #if ENABLE_PROFILING
    {
        char tmp[4096];
        int pp = 0;
        for (int i = 0; i < _red_buf->num_devices; i++) {
            pp += sprintf(tmp + pp, "%.2f ", _red_buf->used_gram[i] / 1e9);
        }
        pp += sprintf(tmp + pp, "\b");
        lg_dbg("device init done, used gram: %s, pageable RAM %.2f GB, pinned RAM %.2f GB", tmp, 
                _red_buf->pageable_ram.load() / 1e9, _red_buf->pinned_ram.load() / 1e9);
    }
    #endif

    for (int tid = 0; tid < _num_threads; tid++) {
        _red_pool[tid]->push([&, tid] {
            for (;;) {
                long bucket_id = -1;
                std::unique_lock<std::mutex> dhr_lock(_dhr_mtx);
                _dhr_cv.wait(dhr_lock, [&] {
                    if (flag & flag_stop_now) return true; 
                    bucket_id = _bwc->pop_bucket();
                    return bucket_id >= 0 || (flag & flag_stop);
                });
                dhr_lock.unlock();

                if ((flag & (flag_stop | flag_stop_now)) && bucket_id == -1) break;

                #if ENABLE_PROFILING
                struct timeval fetch_start, fetch_end;
                gettimeofday(&fetch_start, NULL);
                #endif
                chunk_t *curr_chunk = _bwc->fetch_for_read(bucket_id);
                #if ENABLE_PROFILING
                gettimeofday(&fetch_end, NULL);
                logger->ev_ld_stall_us += (fetch_end.tv_sec - fetch_start.tv_sec) * 1000000 + fetch_end.tv_usec - fetch_start.tv_usec;
                #endif
                int used = 0;
                for (;;) {
                    int task_vecs = 0;
                    int no_more_chunks_in_buc = 0;
                    while (task_vecs < traits::taskVecs) {
                        if (!curr_chunk) { no_more_chunks_in_buc = 1; break; }
                        if (!used) {
                            if (_normalize_chunk(curr_chunk, _pool->CSD)) {
                                lg_warn("chunk %d from bwc (bucket_id %d) not normalized", curr_chunk->id, bucket_id);
                            }
                        }
                        task_vecs += _red_buf->h2d(tid, curr_chunk, used);
                        if (used == -1) {
                            //lg_warn("bucket %d size exceeds buc_max_size(%d), truncated", bucket_id, _red_buf->buc_max_size);
                            _bwc->read_done(curr_chunk, bucket_id);
                            no_more_chunks_in_buc = 1;
                            break;
                        }
                        if (used == curr_chunk->size) {
                            _bwc->read_done(curr_chunk, bucket_id);
                            #if ENABLE_PROFILING
                            gettimeofday(&fetch_start, NULL);
                            #endif
                            curr_chunk = _bwc->fetch_for_read(bucket_id);
                            #if ENABLE_PROFILING
                            gettimeofday(&fetch_end, NULL);
                            logger->ev_ld_stall_us += (fetch_end.tv_sec - fetch_start.tv_sec) * 1000000 + fetch_end.tv_usec - fetch_start.tv_usec;
                            #endif
                            used = 0;
                        }
                    }

                    _red_buf->upk(tid);

                    if (no_more_chunks_in_buc) break;
                }

                _red_buf->run(tid);

                _red_buf->out(tid);
                            
                _bwc->bucket_finalize(bucket_id);
                {
                    std::unique_lock<std::mutex> lock(_bucketer->_dhb_mtx);
                    _bucketer->_dhb_cv.notify_all();
                }
            }
        });
    }
    for (int tid = 0; tid < _num_threads; tid++) {
        _red_pool[tid]->wait_sleep();
    }

    lg_dbg("reduce done");

    for (int tid = 0; tid < _num_threads; tid++) {
        _red_pool[tid]->push([this, tid] { _red_buf->device_done(tid); });
    }
    for (int tid = 0; tid < _num_threads; tid++) {
        _red_pool[tid]->wait_sleep();
    }

    lg_report();
    
    return 0;
}

int *dh_reducer_t::get_result() {
    long ESD8 = _red_buf->ESD8;
    int *ret = (int *)malloc(ESD8 * (1024 + 4 + 4));
    memcpy(ret, _red_buf->res, ESD8 * (1024 + 4 + 4));
    return ret;
}

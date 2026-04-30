/// @file bgj_hd_strategy.cu
/// @brief implementaion of bgj sieve & dual hash

#include "../include/bgj_hd.h"
#include "../include/pool_hd_device.h"
#include "../include/bgj_hd_device.h"

int Pool_hd_t::_bgj_Sieve_hd(int bgj) {
    if (pwc_manager->max_cached_chunks() > pwc_manager_t::pwc_default_max_cached_chunks) {
        pwc_manager->wait_work();
        pwc_manager->set_max_cached_chunks(pwc_manager_t::pwc_default_max_cached_chunks);
    }
    bwc_manager_t *bwc_manager = new bwc_manager_t(this);
    swc_manager_t *swc_manager = new swc_manager_t(this);
    ut_checker_t   *ut_checker = new ut_checker_t(0, this, uid_table, (pwc_manager_t *) swc_manager);
    Bucketer_t       *bucketer = new Bucketer_t(this, bwc_manager, swc_manager, ut_checker);
    Reducer_t         *reducer = new Reducer_t(this, bwc_manager, swc_manager, ut_checker);

    reducer->set_bucketer(bucketer);
    bucketer->set_reducer(reducer);
    
    #if ENABLE_PROFILING
    bwc_manager->logger->set_log_out(this->pwc_manager->logger->log_out());
    bwc_manager->logger->set_log_err(this->pwc_manager->logger->log_err());
    swc_manager->logger->set_log_out(this->pwc_manager->logger->log_out());
    swc_manager->logger->set_log_err(this->pwc_manager->logger->log_err());
    reducer->logger->set_log_out(this->logger->log_out());
    reducer->logger->set_log_err(this->logger->log_err());
    bucketer->logger->set_log_out(this->logger->log_out());
    bucketer->logger->set_log_err(this->logger->log_err());
    #endif

    bucketer->auto_bgj_params_set(bgj);
    reducer->auto_bgj_params_set(bgj);

    int ret;

    std::thread bucketer_thread([&]() { ret = bucketer->run(); });
    std::thread reducer_thread([&]() { reducer->run(); });

    reducer_thread.join();
    bucketer_thread.join();

    delete reducer;
    delete bucketer;
    delete ut_checker;
    delete bwc_manager;
    delete swc_manager;
    
    return ret;
}

int Pool_hd_t::bgj1_Sieve_hd() {
    lg_init();
    int ret = _bgj_Sieve_hd(1);

    lg_report();
    lg_exit();
    
    return ret;
}

int Pool_hd_t::bgj2_Sieve_hd() {
    lg_init();
    int ret = _bgj_Sieve_hd(2);

    lg_report();
    lg_exit();
    
    return ret;
}

int Pool_hd_t::bgj3_Sieve_hd() {
    lg_init();
    int ret = _bgj_Sieve_hd(3);

    lg_report();
    lg_exit();
    
    return ret;
}

int Pool_hd_t::bgj3l_Sieve_hd() {
    lg_init();
    int ret = _bgj_Sieve_hd(4);

    lg_report();
    lg_exit();
    
    return ret;
}

int Pool_hd_t::bgj4_Sieve_hd() {
    lg_init();
    int ret = _bgj_Sieve_hd(5);

    lg_report();
    lg_exit();

    return ret;
}


double buc_traits_t::l0_buc_ratio_estimate(double alpha0, long CSD) {
    double c3, c2, c1, c0;
    if (alpha0 < 0.23) {
        c3 = -0.000000069;
        c2 =  0.0000297;
        c1 =  0.75077010274852140626 * alpha0 * alpha0 * alpha0 + 
             -0.43076160924581596845 * alpha0 * alpha0 + 
              0.08762202607270587473 * alpha0 + 
             -0.01123110166226172491;
        c0 = -19.40319600128166399600 * alpha0 * alpha0 * alpha0 + 
              15.94002455646749716323 * alpha0 * alpha0 + 
             -4.91629671709746851604  * alpha0 + 
              1.22162537249167635345;
    } else if (alpha0 > 0.295) {
        c3 =  0.00000012083210079184 * alpha0 + -0.00000009309592337206;
        c2 = -0.00003326161960092788 * alpha0 +  0.00003628313255913869;
        c1 =  0.01459174834711793323 * alpha0 + -0.00781960851713770638;
        c0 = -0.69173799355295628732 * alpha0 +  0.86258349059657524194;
    } else {
        c3 = -0.000000067;
        c2 =  0.0000285;
        c1 = -0.60927614061399848477 * alpha0 * alpha0 + 
              0.32933429075587278279 * alpha0 + 
             -0.04871399533407529125;
        c0 = 20.11825673532691993728 * alpha0 * alpha0 + 
            -10.94571677923384456221 * alpha0 + 
              2.16332902361058110330;
    }

    double r = c3 * CSD * CSD * CSD + c2 * CSD * CSD + c1 * CSD + c0;
    return 0.66 * r * pow(1 - alpha0 * alpha0, CSD / 2);
}

double buc_traits_t::l0_buc_alpha0_estimate(double ratio, long CSD, double min, double max) {
    double alpha_max = max;
    double alpha_min = min;
    while (alpha_max - alpha_min > 1e-4) {
        double alpha_mid = (alpha_max + alpha_min) / 2;
        if (l0_buc_ratio_estimate(alpha_mid, CSD) > ratio) alpha_min = alpha_mid;
        else alpha_max = alpha_mid;
    }
    return .5 * (alpha_max + alpha_min);
}

double buc_traits_t::l0_out_max_size_ratio(double alpha0, long ESD, long CSD) {
    return 1.317 * l0_buc_ratio_estimate(alpha0, CSD) * (ESD ? 1.136 : 1.0);
}

int32_t buc_traits_t::l0_gbuc_freq(double alpha0, long CSD) {
    double ratio = l0_buc_ratio_estimate(alpha0, CSD);
    int ret = floor(0.25 / ratio);
    return ret ? ret : 1;
}

int32_t buc_traits_t::l0_max_batch0_under(long lim) {
    return 1 << (31 - __builtin_clz(lim));
}

int32_t buc_traits_t::l0_replace_threshold(Pool_hd_t *p, long sol_nvecs, double _improve_ratio) {
    uint64_t num_vec = 0;
    for (int i = 1; i < 65536; i++) num_vec += p->score_stat[i];

    double ratio = 1.0 - 1.43 * sol_nvecs / (double) num_vec;
    ratio = ratio < 0.0 ? 0.0 : ratio;
    uint64_t under_th = num_vec * ratio;

    int ret = 1;
    uint64_t curr = 0;
    while (curr + p->score_stat[ret] < under_th) { curr += p->score_stat[ret]; ret++; }

    return ret;
}

int32_t buc_traits_t::num_threads(int CSD, int ESD, double alpha0, int max_batch0) {
    long exp_gram_per_thread = taskVecs * 180UL + max_batch0 * 4 * l0_out_max_size_ratio(alpha0, ESD, CSD) * taskVecs;
    long threads_per_device = BUC_GRAM_SLIMIT / exp_gram_per_thread;
    int num_devices = hw::gpu_num;
    int ret = threads_per_device * num_devices;
    if (Bucketer_t::bucketer_default_num_threads < ret) ret = Bucketer_t::bucketer_default_num_threads;
    return ret; 
}

typename buc_traits_t::l0_buc_kernel_t buc_traits_t::kernel_chooser(int batch0, int CSD16) {
    typename buc_traits_t::l0_buc_kernel_t kernel = NULL;
    if (batch0 == 2048 && CSD16 == 176) kernel = _bucket_kernel<0, 2048, 176>;
    if (batch0 == 1024 && CSD16 == 176) kernel = _bucket_kernel<0, 1024, 176>;
    if (batch0 == 512  && CSD16 == 176) kernel = _bucket_kernel<0, 512,  176>;
    if (batch0 == 256  && CSD16 == 176) kernel = _bucket_kernel<0, 256,  176>;
    if (batch0 == 128  && CSD16 == 176) kernel = _bucket_kernel<0, 128,  176>;
    if (batch0 == 64   && CSD16 == 176) kernel = _bucket_kernel<0, 64,   176>;
    if (batch0 == 32   && CSD16 == 176) kernel = _bucket_kernel<0, 32,   176>;
    if (batch0 == 16   && CSD16 == 176) kernel = _bucket_kernel<0, 16,   176>;
    #if BGJ_MIN_CSD16 < 176
    if (batch0 == 2048 && CSD16 == 160) kernel = _bucket_kernel<0, 2048, 160>;
    if (batch0 == 1024 && CSD16 == 160) kernel = _bucket_kernel<0, 1024, 160>;
    if (batch0 == 512  && CSD16 == 160) kernel = _bucket_kernel<0, 512,  160>;
    if (batch0 == 256  && CSD16 == 160) kernel = _bucket_kernel<0, 256,  160>;
    if (batch0 == 128  && CSD16 == 160) kernel = _bucket_kernel<0, 128,  160>;
    if (batch0 == 64   && CSD16 == 160) kernel = _bucket_kernel<0, 64,   160>;
    if (batch0 == 32   && CSD16 == 160) kernel = _bucket_kernel<0, 32,   160>;
    if (batch0 == 16   && CSD16 == 160) kernel = _bucket_kernel<0, 16,   160>;
    #endif
    #if BGJ_MIN_CSD16 < 160
    if (batch0 == 2048 && CSD16 == 144) kernel = _bucket_kernel<0, 2048, 144>;
    if (batch0 == 1024 && CSD16 == 144) kernel = _bucket_kernel<0, 1024, 144>;
    if (batch0 == 512  && CSD16 == 144) kernel = _bucket_kernel<0, 512,  144>;
    if (batch0 == 256  && CSD16 == 144) kernel = _bucket_kernel<0, 256,  144>;
    if (batch0 == 128  && CSD16 == 144) kernel = _bucket_kernel<0, 128,  144>;
    if (batch0 == 64   && CSD16 == 144) kernel = _bucket_kernel<0, 64,   144>;
    if (batch0 == 32   && CSD16 == 144) kernel = _bucket_kernel<0, 32,   144>;
    if (batch0 == 16   && CSD16 == 144) kernel = _bucket_kernel<0, 16,   144>;
    #endif
    #if BGJ_MIN_CSD16 < 144
    if (batch0 == 2048 && CSD16 == 128) kernel = _bucket_kernel<0, 2048, 128>;
    if (batch0 == 1024 && CSD16 == 128) kernel = _bucket_kernel<0, 1024, 128>;
    if (batch0 == 512  && CSD16 == 128) kernel = _bucket_kernel<0, 512,  128>;
    if (batch0 == 256  && CSD16 == 128) kernel = _bucket_kernel<0, 256,  128>;
    if (batch0 == 128  && CSD16 == 128) kernel = _bucket_kernel<0, 128,  128>;
    if (batch0 == 64   && CSD16 == 128) kernel = _bucket_kernel<0, 64,   128>;
    if (batch0 == 32   && CSD16 == 128) kernel = _bucket_kernel<0, 32,   128>;
    if (batch0 == 16   && CSD16 == 128) kernel = _bucket_kernel<0, 16,   128>;
    #endif

    return kernel;
}

int red_traits_t::num_threads(int CSD, int ESD, int strategy) {
    int limit = 0;
    double exp_gram_per_thread = 0.0;
    double _buc_max_size = buc_max_size(CSD, ESD, strategy);
    double _out_max_size = red_out_max_size(CSD, ESD, strategy);
    double _flt_out_max_size = flt_out_max_size(CSD, ESD, strategy);
    double _l1_out_max_size = l1_out_max_size(CSD, ESD, strategy);
    double _bk1_max_size = bk1_max_size(CSD, ESD, strategy);
    double _bk2_max_size = bk2_max_size(CSD, ESD, strategy);
    double _bk3_max_size = bk3_max_size(CSD, ESD, strategy);
    double d_norm_vecs = (strategy == Reducer_t::strategy_bgj3l || strategy == Reducer_t::strategy_bgj4) ? _bk1_max_size : _buc_max_size;
    if (d_norm_vecs < taskVecs) d_norm_vecs = taskVecs;

    if (strategy == Reducer_t::strategy_bgj1) {
        exp_gram_per_thread = 190 * (filter_taskVecs + _flt_out_max_size) + 
                _out_max_size * 8 + (_buc_max_size + taskVecs) * 176 + d_norm_vecs * 4;
        limit = BGJ1_RED_DEFAULT_NUM_THREADS;
    }
    if (strategy == Reducer_t::strategy_bgj2) {
        double batch1 = BGJ2_DEFAULT_BATCH1;
        exp_gram_per_thread = 190 * (filter_taskVecs + _flt_out_max_size) + _out_max_size * 8;
        exp_gram_per_thread += taskVecs * 176 + _buc_max_size * 176 + d_norm_vecs * 4;
        exp_gram_per_thread += batch1 * 176 + batch1 * _bk1_max_size * 8;
        limit = BGJ2_RED_DEFAULT_NUM_THREADS;
    }
    if (strategy == Reducer_t::strategy_bgj3) {
        double tpb = BGJ3_DEFAULT_THREADS_PER_BUC;
        double batch1 = BGJ3_DEFAULT_BATCH1;
        double batch2 = BGJ3_DEFAULT_BATCH2;
        exp_gram_per_thread = (190 * (filter_taskVecs + _flt_out_max_size) + _out_max_size * 8) * tpb;
        exp_gram_per_thread += taskVecs * 176 + _buc_max_size * 176 + d_norm_vecs * 4;
        exp_gram_per_thread += batch1 * 176 + batch1 * _bk1_max_size * 8;
        exp_gram_per_thread += tpb * (batch2 * 176 + batch2 * _bk2_max_size * 8);
        limit = BGJ3_RED_DEFAULT_NUM_THREADS;
    }
    if (strategy == Reducer_t::strategy_bgj3l) {
        double batch1 = BGJ3L_DEFAULT_BATCH1;
        double batch2 = BGJ3L_DEFAULT_BATCH2;
        exp_gram_per_thread = (190 * (filter_taskVecs + _flt_out_max_size) + _out_max_size * 8);
        exp_gram_per_thread += (taskVecs * 176 + _bk1_max_size * 176 + d_norm_vecs * 4);
        exp_gram_per_thread += batch1 * 176 + batch1 * _l1_out_max_size * 8;
        exp_gram_per_thread += (batch2 * 176 + batch2 * _bk2_max_size * 8);
        limit = BGJ3L_RED_DEFAULT_NUM_THREADS;
    }
    if (strategy == Reducer_t::strategy_bgj4) {
        double batch1 = BGJ4_DEFAULT_BATCH1;
        double batch2 = BGJ4_DEFAULT_BATCH2;
        double batch3 = BGJ4_DEFAULT_BATCH3;
        exp_gram_per_thread = (190 * (filter_taskVecs + _flt_out_max_size) + _out_max_size * 8);
        exp_gram_per_thread += (taskVecs * 176 + _bk1_max_size * 176 + d_norm_vecs * 4);
        exp_gram_per_thread += batch1 * 176 + batch1 * _l1_out_max_size * 8;
        exp_gram_per_thread += (batch2 * 176 + batch2 * _bk2_max_size * 8);
        exp_gram_per_thread += (batch3 * 176 + batch3 * _bk3_max_size * 8);
        limit = BGJ4_RED_DEFAULT_NUM_THREADS;
    }
    
    int ret = 0;
    if (strategy == Reducer_t::strategy_bgj3l || strategy == Reducer_t::strategy_bgj4) {
        double threads_per_buc = strategy == Reducer_t::strategy_bgj3l ? BGJ3L_DEFAULT_THREADS_PER_BUC : BGJ4_DEFAULT_THREADS_PER_BUC;
        ret = floor(RED_GRAM_SLIMIT * (double)hw::gpu_num / exp_gram_per_thread / threads_per_buc);
    } else {
        long threads_per_device = floor(RED_GRAM_SLIMIT / exp_gram_per_thread);
        ret = threads_per_device * hw::gpu_num;
    }    

    if (limit < ret) ret = limit;
    return ret;
}

int red_traits_t::sieving_stuck(uint64_t total_check, uint64_t total_notin, long CSD) {
    if (total_notin * 36 < total_check && total_check > 100) return 1;
    return 0;
}

int red_traits_t::red_ESD8(int CSD, int ESD) {
    return ESD <= 24 ? 24 : 48;
}

int red_traits_t::red_sbuc_freq(int CSD, int strategy) {
    if (strategy == Reducer_t::strategy_bgj1) {
        return CSD > 100 ? 16 : 1;
    }
    if (strategy == Reducer_t::strategy_bgj2) {
        return CSD > 120 ? 16 : 1;
    }
    if (strategy == Reducer_t::strategy_bgj3) {
        return CSD > 120 ? 16 : 1;
    }
    if (strategy == Reducer_t::strategy_bgj3l) {
        if (CSD > 130) return 64;
        if (CSD > 120) return 16;
        return 1;
    }
    if (strategy == Reducer_t::strategy_bgj4) {
        return CSD > 120 ? 16 : 1;
    }
    return 0;
}

int red_traits_t::red_gbuc_freq(int CSD, int strategy) {
    if (strategy == Reducer_t::strategy_bgj1) {
        return CSD > 100 ? 128 : 2;
    }
    if (strategy == Reducer_t::strategy_bgj2) {
        return CSD > 120 ? 128 : 2;
    }
    if (strategy == Reducer_t::strategy_bgj3) {
        return CSD > 120 ? 128 : 2;
    }
    if (strategy == Reducer_t::strategy_bgj3l) {
        if (CSD > 130) return 512;
        if (CSD > 120) return 128;
        return 2;
    }
    if (strategy == Reducer_t::strategy_bgj4) {
        return CSD > 120 ? 128 : 2;
    }
    return 0;
}

int red_traits_t::bk1_gbuc_freq(int CSD, int strategy) {
    return 1;
    /// @todo better estimation
    int ret = 0;
    if (strategy == Reducer_t::strategy_bgj2) {
        ret = (int32_t) floor(0.15 / buc_traits_t::l0_buc_ratio_estimate(BGJ2_DEFAULT_ALPHA1, CSD));
    }
    if (strategy == Reducer_t::strategy_bgj3) {
        ret = (int32_t) floor(0.15 / buc_traits_t::l0_buc_ratio_estimate(BGJ3_DEFAULT_ALPHA1, CSD));
    }
    if (strategy == Reducer_t::strategy_bgj3l) {
        ret = (int32_t) floor(0.15 / buc_traits_t::l0_buc_ratio_estimate(BGJ3L_DEFAULT_ALPHA1, CSD));
    }
    if (strategy == Reducer_t::strategy_bgj4) {
        ret = (int32_t) floor(0.15 / buc_traits_t::l0_buc_ratio_estimate(BGJ4_DEFAULT_ALPHA1, CSD));
    }

    if (ret == 0) ret = 1;
    return ret;
}

int red_traits_t::bk2_gbuc_freq(int CSD, int strategy) {
    /// @todo better estimation
    int ret = 0;
    if (strategy == Reducer_t::strategy_bgj3) {
        ret = 8;
    }
    if (strategy == Reducer_t::strategy_bgj3l) {
        ret = 8;
    }
    if (strategy == Reducer_t::strategy_bgj4) {
        ret = 2;
    }

    if (ret == 0) ret = 1;
    return ret;
}

int red_traits_t::bk3_gbuc_freq(int CSD, int strategy) {
    return 1;
    /// @todo better estimation
    int ret = 0;
    if (strategy == Reducer_t::strategy_bgj4) {
        ret = (int32_t) floor(0.07 / buc_traits_t::l0_buc_ratio_estimate(BGJ4_DEFAULT_ALPHA3, CSD));
    }

    if (ret == 0) ret = 1;
    return ret;
}

long red_traits_t::buc_max_size(int CSD, int ESD, int strategy) {
    long ret = 0;
    double size_ratio = strategy == Reducer_t::strategy_bgj1 ? BGJ1_SIZE_RATIO :
                        strategy == Reducer_t::strategy_bgj2 ? BGJ2_SIZE_RATIO :
                        strategy == Reducer_t::strategy_bgj3 ? BGJ3_SIZE_RATIO :
                        strategy == Reducer_t::strategy_bgj3l ? BGJ3L_SIZE_RATIO :
                        strategy == Reducer_t::strategy_bgj4 ? BGJ4_SIZE_RATIO : 0.0;
    double alpha0 = strategy == Reducer_t::strategy_bgj1 ? BGJ1_L0_MIN_ALPHA0 :
                    strategy == Reducer_t::strategy_bgj2 ? BGJ2_L0_MIN_ALPHA0 :
                    strategy == Reducer_t::strategy_bgj3 ? BGJ3_L0_MIN_ALPHA0 :
                    strategy == Reducer_t::strategy_bgj3l ? BGJ3L_L0_MIN_ALPHA0 :
                    strategy == Reducer_t::strategy_bgj4 ? BGJ4_L0_MIN_ALPHA0 : 0.0;
    if (strategy == Reducer_t::strategy_bgj3 || strategy == Reducer_t::strategy_bgj3l) {
        ret = 1.05 * pow(2.0, 0.17286844935741618734 * CSD + 0.27598874612316492971);
    } else {
        ret = 1.05 * size_ratio * pow(4./3., CSD * .5) * buc_traits_t::l0_out_max_size_ratio(alpha0, ESD, CSD) * 0.9;
    }
    ret = (ret + 511L) / 512L * 512L;
    if (ret < 1024) ret = 1024;
    return ret;
}

long red_traits_t::red_out_max_size(int CSD, int ESD, int strategy) {
    long ret = 0;
    if (strategy == Reducer_t::strategy_bgj1) {
        ret = ESD ? pow(2.0, 0.135 * CSD + 1.0) + 100.0 : pow(2.0, 0.1 * CSD + 3.0) + 50.0;
    }
    if (strategy == Reducer_t::strategy_bgj2) {
        ret = (ESD ? pow(2.0, 0.125 * CSD + 7.5) : pow(2.0, 0.075 * CSD + 11.0)) + 50.0 * BGJ2_DEFAULT_BATCH1;
    }
    if (strategy == Reducer_t::strategy_bgj3) {
        ret = ESD ? 8388608 : 8388608;
    }
    if (strategy == Reducer_t::strategy_bgj3l) {
        ret = ESD ? 8388608 : 8388608;
    }
    if (strategy == Reducer_t::strategy_bgj4) {
        ret = ESD ? 8388608 : 8388608;
    }
    ret = (ret + 511L) / 512L * 512L;
    return ret;
}

long red_traits_t::flt_out_max_size(int CSD, int ESD, int strategy) {
    long ret = red_out_max_size(CSD, 0, strategy);
    ret = (ret + 511L) / 512L * 512L;
    if (strategy == Reducer_t::strategy_bgj3l || strategy == Reducer_t::strategy_bgj4) {
        if (ret < taskVecs) ret = taskVecs;
    }
    if (strategy == Reducer_t::strategy_bgj3l || strategy == Reducer_t::strategy_bgj4 || strategy == Reducer_t::strategy_bgj3) {
        ret = 3200000;
    }

    return ret;
}

long red_traits_t::bk1_max_size(int CSD, int ESD, int strategy) {
    long ret = 0;
    if (strategy == Reducer_t::strategy_bgj2) {
        ret = ESD ? pow(2.0, 0.135 * CSD - 2.1) : pow(2.0, 0.11 * CSD);
    }
    if (strategy == Reducer_t::strategy_bgj3) {
        ret = ESD ? pow(2.0, 0.14 * CSD + 0.3) : (pow(2.0, 0.15774307272784712786 * CSD + -2.31663107026080083983) * 1.15);
    }
    if (strategy == Reducer_t::strategy_bgj3l) {
        ret = ESD ? pow(2.0, 0.14 * CSD + 0.3) : (pow(2.0, 0.15774307272784712786 * CSD + -2.31663107026080083983) * 1.15);
    }
    if (strategy == Reducer_t::strategy_bgj4) {
        /// @todo better estimation
        ret = ESD ? pow(2.0, 0.145 * CSD + 2.0) : pow(2.0, 0.145 * CSD + 2.0);
    }
    ret = (ret + 511L) / 512L * 512L;
    if (ret < 1024) ret = 1024;
    return ret;
}

long red_traits_t::bk2_max_size(int CSD, int ESD, int strategy) {
    long ret = 0;
    if (strategy == Reducer_t::strategy_bgj3) {
        ret = ESD ? pow(2.0, 0.108 * CSD - 0.3) : (pow(2.0, 0.10741433650340584394 * CSD + -3.44189983264612742175) * 3.0);
    }
    if (strategy == Reducer_t::strategy_bgj3l) {
        ret = ESD ? pow(2.0, 0.108 * CSD - 0.3) : (pow(2.0, 0.10741433650340584394 * CSD + -3.44189983264612742175) * 3.0);
    }
    if (strategy == Reducer_t::strategy_bgj4) {
        /// @todo better estimation
        ret = ESD ? pow(2.0, 0.12 * CSD + 1.0) : pow(2.0, 0.12 * CSD + 1.0);
    }
    ret = (ret + 511L) / 512L * 512L;
    if (ret < 1024) ret = 1024;
    return ret;
}

long red_traits_t::bk3_max_size(int CSD, int ESD, int strategy) {
    long ret = 0;
    if (strategy == Reducer_t::strategy_bgj4) {
        /// @todo better estimation
    }
    ret = (ret + 511L) / 512L * 512L;
    if (ret < 1024) ret = 1024;
    return ret;
}

long red_traits_t::l1_out_max_size(int CSD, int ESD, int strategy) {
    /// @todo better estimation
    long ret = 0;
    float alpha1 = strategy == Reducer_t::strategy_bgj3l ? BGJ3L_DEFAULT_ALPHA1 : BGJ4_DEFAULT_ALPHA1;
    if (strategy == Reducer_t::strategy_bgj3l) {
        ret = taskVecs * pow(2.0, -0.01512537662956906 * CSD - 2.5926198163839658) * 1.15;
    } else if (strategy == Reducer_t::strategy_bgj4) {
        ret = taskVecs * buc_traits_t::l0_buc_ratio_estimate(alpha1, CSD) * 1.8;
    }
    ret = (ret + 511L) / 512L * 512L;
    return ret;
}

typename red_traits_t::pk_kernel_t red_traits_t::pk_kernel_chooser(int CSD16) {
    if (CSD16 == 176) return _device_pack<176>;
    #if RED_MIN_CSD16 < 176
    if (CSD16 == 160) return _device_pack<160>;
    #endif
    #if RED_MIN_CSD16 < 160
    if (CSD16 == 144) return _device_pack<144>;
    #endif
    #if RED_MIN_CSD16 < 144
    if (CSD16 == 128) return _device_pack<128>;
    #endif
    return NULL;
}

typename red_traits_t::upk_kernel_t red_traits_t::upk_kernel_chooser(int CSD16) {
    if (CSD16 == 176) return _device_unpack<176>;
    #if RED_MIN_CSD16 < 176
    if (CSD16 == 160) return _device_unpack<160>;
    #endif
    #if RED_MIN_CSD16 < 160
    if (CSD16 == 144) return _device_unpack<144>;
    #endif
    #if RED_MIN_CSD16 < 144
    if (CSD16 == 128) return _device_unpack<128>;
    #endif
    return NULL;
}

typename red_traits_t::rpp_kernel_t red_traits_t::rpp_kernel_chooser(int CSD16, int strategy) {
    if (strategy == Reducer_t::strategy_bgj1) return _reduce_prepare<0>;
    #if USE_GRAPH
    if (strategy == Reducer_t::strategy_bgj2) return _reduce_prepare<1>;
    if (strategy == Reducer_t::strategy_bgj3) return _reduce_prepare<1>;
    if (strategy == Reducer_t::strategy_bgj3l) return _reduce_prepare<1>;
    if (strategy == Reducer_t::strategy_bgj4) return _reduce_prepare<1>;
    #else
    if (strategy == Reducer_t::strategy_bgj2) return _multi_reduce_prepare;
    if (strategy == Reducer_t::strategy_bgj3) return _multi_reduce_prepare;
    if (strategy == Reducer_t::strategy_bgj3l) return _multi_reduce_prepare;
    if (strategy == Reducer_t::strategy_bgj4) return _multi_reduce_prepare;
    #endif
    return NULL;
}

#if USE_GRAPH
typename red_traits_t::red_kernel_t red_traits_t::red_kernel_chooser(int CSD16, int strategy, int sbuc_freq, int gbuc_freq) {
    if (strategy == Reducer_t::strategy_bgj1) {
        if (CSD16 == 176 && sbuc_freq == 16 && gbuc_freq == 128) return _reduce_kernel<0, 176, 16, 128>;
        if (CSD16 == 176 && sbuc_freq == 1  && gbuc_freq == 2  ) return _reduce_kernel<0, 176, 1,  2>;
        #if RED_MIN_CSD16 < 176
        if (CSD16 == 160 && sbuc_freq == 16 && gbuc_freq == 128) return _reduce_kernel<0, 160, 16, 128>;
        if (CSD16 == 160 && sbuc_freq == 1  && gbuc_freq == 2  ) return _reduce_kernel<0, 160, 1,  2>;
        #endif
        #if RED_MIN_CSD16 < 160
        if (CSD16 == 144 && sbuc_freq == 16 && gbuc_freq == 128) return _reduce_kernel<0, 144, 16, 128>;
        if (CSD16 == 144 && sbuc_freq == 1  && gbuc_freq == 2  ) return _reduce_kernel<0, 144, 1,  2>;
        #endif
        #if RED_MIN_CSD16 < 144
        if (CSD16 == 128 && sbuc_freq == 16 && gbuc_freq == 128) return _reduce_kernel<0, 128, 16, 128>;
        if (CSD16 == 128 && sbuc_freq == 1  && gbuc_freq == 2  ) return _reduce_kernel<0, 128, 1,  2>;
        #endif
        return NULL;
    } 

    if (strategy != Reducer_t::strategy_bgj2 && strategy != Reducer_t::strategy_bgj3 &&
        strategy != Reducer_t::strategy_bgj3l && strategy != Reducer_t::strategy_bgj4) return NULL;

    if (CSD16 == 176 && sbuc_freq == 16 && gbuc_freq == 128) return _reduce_kernel<1, 176, 16, 128>;
    if (CSD16 == 176 && sbuc_freq == 1  && gbuc_freq == 2  ) return _reduce_kernel<1, 176, 1,  2>;
    #if RED_MIN_CSD16 < 176
    if (CSD16 == 160 && sbuc_freq == 16 && gbuc_freq == 128) return _reduce_kernel<1, 160, 16, 128>;
    if (CSD16 == 160 && sbuc_freq == 1  && gbuc_freq == 2  ) return _reduce_kernel<1, 160, 1,  2>;
    #endif
    #if RED_MIN_CSD16 < 160
    if (CSD16 == 144 && sbuc_freq == 16 && gbuc_freq == 128) return _reduce_kernel<1, 144, 16, 128>;
    if (CSD16 == 144 && sbuc_freq == 1  && gbuc_freq == 2  ) return _reduce_kernel<1, 144, 1,  2>;
    #endif
    #if RED_MIN_CSD16 < 144
    if (CSD16 == 128 && sbuc_freq == 16 && gbuc_freq == 128) return _reduce_kernel<1, 128, 16, 128>;
    if (CSD16 == 128 && sbuc_freq == 1  && gbuc_freq == 2  ) return _reduce_kernel<1, 128, 1,  2>;
    #endif
    return NULL;
}
#else
typename red_traits_t::rd1_kernel_t red_traits_t::rd1_kernel_chooser(int CSD16, int strategy, int sbuc_freq, int gbuc_freq) {
    if (strategy == Reducer_t::strategy_bgj1) {
        if (CSD16 == 176 && sbuc_freq == 16 && gbuc_freq == 128) return _reduce_kernel<0, 176, 16, 128>;
        if (CSD16 == 176 && sbuc_freq == 1  && gbuc_freq == 2  ) return _reduce_kernel<0, 176, 1,  2>;
        #if RED_MIN_CSD16 < 176
        if (CSD16 == 160 && sbuc_freq == 16 && gbuc_freq == 128) return _reduce_kernel<0, 160, 16, 128>;
        if (CSD16 == 160 && sbuc_freq == 1  && gbuc_freq == 2  ) return _reduce_kernel<0, 160, 1,  2>;
        #endif
        #if RED_MIN_CSD16 < 160
        if (CSD16 == 144 && sbuc_freq == 16 && gbuc_freq == 128) return _reduce_kernel<0, 144, 16, 128>;
        if (CSD16 == 144 && sbuc_freq == 1  && gbuc_freq == 2  ) return _reduce_kernel<0, 144, 1,  2>;
        #endif
        #if RED_MIN_CSD16 < 144
        if (CSD16 == 128 && sbuc_freq == 16 && gbuc_freq == 128) return _reduce_kernel<0, 128, 16, 128>;
        if (CSD16 == 128 && sbuc_freq == 1  && gbuc_freq == 2  ) return _reduce_kernel<0, 128, 1,  2>;
        #endif
    }
    return NULL;
}
typename red_traits_t::red_kernel_t red_traits_t::red_kernel_chooser(int CSD16, int strategy, int sbuc_freq, int gbuc_freq) {
    if (strategy != Reducer_t::strategy_bgj2 && strategy != Reducer_t::strategy_bgj3 &&
        strategy != Reducer_t::strategy_bgj3l && strategy != Reducer_t::strategy_bgj4) return NULL;

    if (CSD16 == 176 && sbuc_freq == 64 && gbuc_freq == 512) return _multi_reduce_kernel<176, 64, 512>;
    if (CSD16 == 176 && sbuc_freq == 16 && gbuc_freq == 128) return _multi_reduce_kernel<176, 16, 128>;
    if (CSD16 == 176 && sbuc_freq == 1  && gbuc_freq == 2  ) return _multi_reduce_kernel<176, 1,  2>;
    #if RED_MIN_CSD16 < 176
    if (CSD16 == 160 && sbuc_freq == 64 && gbuc_freq == 512) return _multi_reduce_kernel<160, 64, 512>;
    if (CSD16 == 160 && sbuc_freq == 16 && gbuc_freq == 128) return _multi_reduce_kernel<160, 16, 128>;
    if (CSD16 == 160 && sbuc_freq == 1  && gbuc_freq == 2  ) return _multi_reduce_kernel<160, 1,  2>;
    #endif
    #if RED_MIN_CSD16 < 160
    if (CSD16 == 144 && sbuc_freq == 64 && gbuc_freq == 512) return _multi_reduce_kernel<144, 64, 512>;
    if (CSD16 == 144 && sbuc_freq == 16 && gbuc_freq == 128) return _multi_reduce_kernel<144, 16, 128>;
    if (CSD16 == 144 && sbuc_freq == 1  && gbuc_freq == 2  ) return _multi_reduce_kernel<144, 1,  2>;
    #endif
    #if RED_MIN_CSD16 < 144
    if (CSD16 == 128 && sbuc_freq == 64 && gbuc_freq == 512) return _multi_reduce_kernel<128, 64, 512>;
    if (CSD16 == 128 && sbuc_freq == 16 && gbuc_freq == 128) return _multi_reduce_kernel<128, 16, 128>;
    if (CSD16 == 128 && sbuc_freq == 1  && gbuc_freq == 2  ) return _multi_reduce_kernel<128, 1,  2>;
    #endif
    return NULL;
}
#endif

typename red_traits_t::fpv_kernel_t red_traits_t::fpv_kernel_chooser(int CSD16) {
    if (CSD16 == 176) return filter_prepare_vec<176>;
    #if RED_MIN_CSD16 < 176
    if (CSD16 == 160) return filter_prepare_vec<160>;
    #endif
    #if RED_MIN_CSD16 < 160
    if (CSD16 == 144) return filter_prepare_vec<144>;
    #endif
    #if RED_MIN_CSD16 < 144
    if (CSD16 == 128) return filter_prepare_vec<128>;
    #endif
    return NULL;
}

typename red_traits_t::flt_kernel_t red_traits_t::flt_kernel_chooser(int CSD16, int ESD8) {
    if (CSD16 == 176 && ESD8 == 24) return filter_kernel<176, 24>;
    if (CSD16 == 176 && ESD8 == 48) return filter_kernel<176, 48>;
    #if RED_MIN_CSD16 < 176
    if (CSD16 == 160 && ESD8 == 24) return filter_kernel<160, 24>;
    if (CSD16 == 160 && ESD8 == 48) return filter_kernel<160, 48>;
    #endif
    #if RED_MIN_CSD16 < 160
    if (CSD16 == 144 && ESD8 == 24) return filter_kernel<144, 24>;
    if (CSD16 == 144 && ESD8 == 48) return filter_kernel<144, 48>;
    #endif
    #if RED_MIN_CSD16 < 144
    if (CSD16 == 128 && ESD8 == 24) return filter_kernel<128, 24>;
    if (CSD16 == 128 && ESD8 == 48) return filter_kernel<128, 48>;
    #endif
    return NULL;
}

typename red_traits_t::fcs_kernel_t red_traits_t::fcs_kernel_chooser(int CSD16) {
    if (CSD16 == 176) return filter_collect_sol<176>;
    #if RED_MIN_CSD16 < 176
    if (CSD16 == 160) return filter_collect_sol<160>;
    #endif
    #if RED_MIN_CSD16 < 160
    if (CSD16 == 144) return filter_collect_sol<144>;
    #endif
    #if RED_MIN_CSD16 < 144
    if (CSD16 == 128) return filter_collect_sol<128>;
    #endif
    return NULL;
}

typename red_traits_t::bk_kernel_t red_traits_t::bk1_kernel_chooser(int CSD16, int batch1, int strategy) {
    typename red_traits_t::bk_kernel_t kernel = NULL;
    if (strategy == Reducer_t::strategy_bgj2) {
        if (CSD16 == 176 && batch1 == BGJ2_DEFAULT_BATCH1) kernel = _bucket_kernel<1, BGJ2_DEFAULT_BATCH1, 176>;
        #if BGJ_MIN_CSD16 < 176
        if (CSD16 == 160 && batch1 == BGJ2_DEFAULT_BATCH1) kernel = _bucket_kernel<1, BGJ2_DEFAULT_BATCH1, 160>;
        #endif
        #if BGJ_MIN_CSD16 < 160
        if (CSD16 == 144 && batch1 == BGJ2_DEFAULT_BATCH1) kernel = _bucket_kernel<1, BGJ2_DEFAULT_BATCH1, 144>;
        #endif
        #if BGJ_MIN_CSD16 < 144
        if (CSD16 == 128 && batch1 == BGJ2_DEFAULT_BATCH1) kernel = _bucket_kernel<1, BGJ2_DEFAULT_BATCH1, 128>;
        #endif
    }
    if (strategy == Reducer_t::strategy_bgj3) {
        if (CSD16 == 176 && batch1 == BGJ3_DEFAULT_BATCH1) kernel = _bucket_kernel<1, BGJ3_DEFAULT_BATCH1, 176>;
        #if BGJ_MIN_CSD16 < 176
        if (CSD16 == 160 && batch1 == BGJ3_DEFAULT_BATCH1) kernel = _bucket_kernel<1, BGJ3_DEFAULT_BATCH1, 160>;
        #endif
        #if BGJ_MIN_CSD16 < 160
        if (CSD16 == 144 && batch1 == BGJ3_DEFAULT_BATCH1) kernel = _bucket_kernel<1, BGJ3_DEFAULT_BATCH1, 144>;
        #endif
        #if BGJ_MIN_CSD16 < 144
        if (CSD16 == 128 && batch1 == BGJ3_DEFAULT_BATCH1) kernel = _bucket_kernel<1, BGJ3_DEFAULT_BATCH1, 128>;
        #endif
    }

    if (strategy == Reducer_t::strategy_bgj3l) {
        if (CSD16 == 176 && batch1 == BGJ3L_DEFAULT_BATCH1) kernel = _bucket_kernel<1, BGJ3L_DEFAULT_BATCH1, 176>;
        #if BGJ_MIN_CSD16 < 176
        if (CSD16 == 160 && batch1 == BGJ3L_DEFAULT_BATCH1) kernel = _bucket_kernel<1, BGJ3L_DEFAULT_BATCH1, 160>;
        #endif
        #if BGJ_MIN_CSD16 < 160
        if (CSD16 == 144 && batch1 == BGJ3L_DEFAULT_BATCH1) kernel = _bucket_kernel<1, BGJ3L_DEFAULT_BATCH1, 144>;
        #endif
        #if BGJ_MIN_CSD16 < 144
        if (CSD16 == 128 && batch1 == BGJ3L_DEFAULT_BATCH1) kernel = _bucket_kernel<1, BGJ3L_DEFAULT_BATCH1, 128>;
        #endif
    }
    if (strategy == Reducer_t::strategy_bgj4) {
        if (CSD16 == 176 && batch1 == BGJ4_DEFAULT_BATCH1) kernel = _bucket_kernel<1, BGJ4_DEFAULT_BATCH1, 176>;
        #if BGJ_MIN_CSD16 < 176
        if (CSD16 == 160 && batch1 == BGJ4_DEFAULT_BATCH1) kernel = _bucket_kernel<1, BGJ4_DEFAULT_BATCH1, 160>;
        #endif
        #if BGJ_MIN_CSD16 < 160
        if (CSD16 == 144 && batch1 == BGJ4_DEFAULT_BATCH1) kernel = _bucket_kernel<1, BGJ4_DEFAULT_BATCH1, 144>;
        #endif
        #if BGJ_MIN_CSD16 < 144
        if (CSD16 == 128 && batch1 == BGJ4_DEFAULT_BATCH1) kernel = _bucket_kernel<1, BGJ4_DEFAULT_BATCH1, 128>;
        #endif
    }

    return kernel;
}

typename red_traits_t::bk_kernel_t red_traits_t::bk2_kernel_chooser(int CSD16, int batch2, int strategy) {
    typename red_traits_t::bk_kernel_t kernel = NULL;
    if (strategy == Reducer_t::strategy_bgj3) {
        if (CSD16 == 176 && batch2 == BGJ3_DEFAULT_BATCH2) kernel = _bucket_kernel<2, BGJ3_DEFAULT_BATCH2, 176>;
        #if BGJ_MIN_CSD16 < 176
        if (CSD16 == 160 && batch2 == BGJ3_DEFAULT_BATCH2) kernel = _bucket_kernel<2, BGJ3_DEFAULT_BATCH2, 160>;
        #endif
        #if BGJ_MIN_CSD16 < 160
        if (CSD16 == 144 && batch2 == BGJ3_DEFAULT_BATCH2) kernel = _bucket_kernel<2, BGJ3_DEFAULT_BATCH2, 144>;
        #endif
        #if BGJ_MIN_CSD16 < 144
        if (CSD16 == 128 && batch2 == BGJ3_DEFAULT_BATCH2) kernel = _bucket_kernel<2, BGJ3_DEFAULT_BATCH2, 128>;
        #endif
    }
    if (strategy == Reducer_t::strategy_bgj3l) {
        if (CSD16 == 176 && batch2 == BGJ3L_DEFAULT_BATCH2) kernel = _bucket_kernel<1, BGJ3L_DEFAULT_BATCH2, 176>;
        #if BGJ_MIN_CSD16 < 176
        if (CSD16 == 160 && batch2 == BGJ3L_DEFAULT_BATCH2) kernel = _bucket_kernel<1, BGJ3L_DEFAULT_BATCH2, 160>;
        #endif
        #if BGJ_MIN_CSD16 < 160
        if (CSD16 == 144 && batch2 == BGJ3L_DEFAULT_BATCH2) kernel = _bucket_kernel<1, BGJ3L_DEFAULT_BATCH2, 144>;
        #endif
        #if BGJ_MIN_CSD16 < 144
        if (CSD16 == 128 && batch2 == BGJ3L_DEFAULT_BATCH2) kernel = _bucket_kernel<1, BGJ3L_DEFAULT_BATCH2, 128>;
        #endif
    }
    if (strategy == Reducer_t::strategy_bgj4) {
        if (CSD16 == 176 && batch2 == BGJ4_DEFAULT_BATCH2) kernel = _bucket_kernel<1, BGJ4_DEFAULT_BATCH2, 176>;
        #if BGJ_MIN_CSD16 < 176
        if (CSD16 == 160 && batch2 == BGJ4_DEFAULT_BATCH2) kernel = _bucket_kernel<1, BGJ4_DEFAULT_BATCH2, 160>;
        #endif
        #if BGJ_MIN_CSD16 < 160
        if (CSD16 == 144 && batch2 == BGJ4_DEFAULT_BATCH2) kernel = _bucket_kernel<1, BGJ4_DEFAULT_BATCH2, 144>;
        #endif
        #if BGJ_MIN_CSD16 < 144
        if (CSD16 == 128 && batch2 == BGJ4_DEFAULT_BATCH2) kernel = _bucket_kernel<1, BGJ4_DEFAULT_BATCH2, 128>;
        #endif
    }

    return kernel;
}

typename red_traits_t::bk_kernel_t red_traits_t::bk3_kernel_chooser(int CSD16, int batch3, int strategy) {
    typename red_traits_t::bk_kernel_t kernel = NULL;
    if (strategy == Reducer_t::strategy_bgj4) {
        if (CSD16 == 176 && batch3 == BGJ4_DEFAULT_BATCH3) kernel = _bucket_kernel<2, BGJ4_DEFAULT_BATCH3, 176>;
        #if BGJ_MIN_CSD16 < 176
        if (CSD16 == 160 && batch3 == BGJ4_DEFAULT_BATCH3) kernel = _bucket_kernel<2, BGJ4_DEFAULT_BATCH3, 160>;
        #endif
        #if BGJ_MIN_CSD16 < 160
        if (CSD16 == 144 && batch3 == BGJ4_DEFAULT_BATCH3) kernel = _bucket_kernel<2, BGJ4_DEFAULT_BATCH3, 144>;
        #endif
        #if BGJ_MIN_CSD16 < 144
        if (CSD16 == 128 && batch3 == BGJ4_DEFAULT_BATCH3) kernel = _bucket_kernel<2, BGJ4_DEFAULT_BATCH3, 128>;
        #endif
    }

    return kernel;
}

typename red_traits_t::bk_kernel_t red_traits_t::rep_kernel_chooser(int CSD16, int batch2, int strategy) {
    typename red_traits_t::bk_kernel_t kernel = NULL;

    if (strategy == Reducer_t::strategy_bgj3l) {
        if (CSD16 == 176 && batch2 == BGJ3L_DEFAULT_BATCH2) kernel = _bucket_kernel<2, BGJ3L_DEFAULT_BATCH2, 176>;
        #if BGJ_MIN_CSD16 < 176
        if (CSD16 == 160 && batch2 == BGJ3L_DEFAULT_BATCH2) kernel = _bucket_kernel<2, BGJ3L_DEFAULT_BATCH2, 160>;
        #endif
        #if BGJ_MIN_CSD16 < 160
        if (CSD16 == 144 && batch2 == BGJ3L_DEFAULT_BATCH2) kernel = _bucket_kernel<2, BGJ3L_DEFAULT_BATCH2, 144>;
        #endif
        #if BGJ_MIN_CSD16 < 144
        if (CSD16 == 128 && batch2 == BGJ3L_DEFAULT_BATCH2) kernel = _bucket_kernel<2, BGJ3L_DEFAULT_BATCH2, 128>;
        #endif
    }
    if (strategy == Reducer_t::strategy_bgj4) {
        if (CSD16 == 176 && batch2 == BGJ4_DEFAULT_BATCH2) kernel = _bucket_kernel<2, BGJ4_DEFAULT_BATCH2, 176>;
        #if BGJ_MIN_CSD16 < 176
        if (CSD16 == 160 && batch2 == BGJ4_DEFAULT_BATCH2) kernel = _bucket_kernel<2, BGJ4_DEFAULT_BATCH2, 160>;
        #endif
        #if BGJ_MIN_CSD16 < 160
        if (CSD16 == 144 && batch2 == BGJ4_DEFAULT_BATCH2) kernel = _bucket_kernel<2, BGJ4_DEFAULT_BATCH2, 144>;
        #endif
        #if BGJ_MIN_CSD16 < 144
        if (CSD16 == 128 && batch2 == BGJ4_DEFAULT_BATCH2) kernel = _bucket_kernel<2, BGJ4_DEFAULT_BATCH2, 128>;
        #endif
    }

    return kernel;
}

static inline long ceil256(long n) {
    return ((n + 255L) / 256L) * 256L;
}

buc_buffer_holder_t::buc_buffer_holder_t(Bucketer_t *bucketer) {
    #if ENABLE_PROFILING
    logger = bucketer->logger;
    #endif

    this->bucketer = bucketer;

    this->alpha0        = bucketer->_alpha0;
    this->CSD           = bucketer->_pool->CSD;
    this->CSD16         = (CSD + 15) / 16 * 16 < BUC_MIN_CSD16 ? BUC_MIN_CSD16 : (CSD + 15) / 16 * 16;
    this->max_batch0    = bucketer->_max_batch0;
    this->out_max_size  = traits::l0_out_max_size_ratio(alpha0, bucketer->_pool->ESD, CSD) * traits::taskVecs + 64;
    this->gbuc_freq     = traits::l0_gbuc_freq(alpha0, CSD);

    if (bucketer->_reducer->_strategy == Reducer_t::strategy_bgj3 || 
        bucketer->_reducer->_strategy == Reducer_t::strategy_bgj3l) {
        double size_ratio = bucketer->_reducer->_strategy == Reducer_t::strategy_bgj3 ? 
                            BGJ3_SIZE_RATIO : BGJ3L_SIZE_RATIO;
        double expect_pool_size = size_ratio * pow(4./3., CSD * .5);
        this->out_max_size = (1.1 * pow(2.0, 0.17286844935741618734 * CSD + 0.27598874612316492971) / expect_pool_size) * traits::taskVecs + 64;
    }
    
    /// thread & device info
    int _num_devices;
    _num_devices = hw::gpu_num;
    pthread_spin_init(&gram_lock, PTHREAD_PROCESS_SHARED);
    this->num_devices   = _num_devices;
    this->num_threads   = bucketer->_num_threads;
    this->streams       = (cudaStream_t *) malloc(num_threads * sizeof(cudaStream_t));
    this->used_gram     = (long *) calloc(num_devices, sizeof(long));
    
    #if ENABLE_PROFILING
    this->logger->num_devices = this->num_devices;
    this->logger->num_threads = this->num_threads;
    this->logger->chunk_nbytes = (14 + CSD) * Pool_hd_t::chunk_max_nvecs;
    this->logger->CSD16 = CSD16;
    #endif

    lg_dbg("#device %ld, CSD16 %ld, out_max_size %ld, gbuc_freq %d", num_devices, CSD16, out_max_size, gbuc_freq);
    
    /// runtime data
    long nbytes_task_vecs  = ceil256(num_threads * sizeof(int32_t));
    long nbytes_h_center16 = ceil256(max_batch0 * CSD16 * sizeof(int8_t));
    long nbytes_h_norm     = ceil256(traits::taskVecs * sizeof(int32_t));
    long nbytes_h_out      = ceil256(max_batch0 * (out_max_size + 1) * sizeof(uint32_t));
    long nbytes_pinned = nbytes_task_vecs + nbytes_h_center16 + (nbytes_h_norm + nbytes_h_out) * num_threads;
    nbytes_pinned = ((nbytes_pinned + 4095L) / 4096L) * 4096L;
    char *pinned_buf = NULL;
    if (posix_memalign((void **)&pinned_buf, 4096, nbytes_pinned)) {
        lg_err("posix_memalign failed");
    }
    CHECK_CUDA_ERR(cudaHostRegister(pinned_buf, nbytes_pinned, cudaHostAllocPortable));
    pinned_ram.fetch_add(nbytes_pinned, std::memory_order_relaxed);
    task_vecs  = (int32_t *)   pinned_buf;
    h_center16 = (int8_t *)    (pinned_buf + nbytes_task_vecs);
    d_center16 = (int8_t **)   malloc(num_threads * sizeof(int8_t *));
    d_vec      = (int8_t **)   malloc(num_threads * sizeof(int8_t *));
    h_norm     = (int32_t **)  malloc(num_threads * sizeof(int32_t *));
    d_norm     = (int32_t **)  malloc(num_threads * sizeof(int32_t *));
    d_n        = (int32_t **)  malloc(num_threads * sizeof(int32_t *));
    h_out      = (uint32_t **) malloc(num_threads * sizeof(uint32_t *));
    d_out      = (uint32_t **) malloc(num_threads * sizeof(uint32_t *));
    
    /// host init
    for (int i = 0; i < num_threads; i++) {
        h_norm[i] = (int32_t *) (pinned_buf + nbytes_task_vecs + nbytes_h_center16 + i * nbytes_h_norm);
        h_out[i]  = (uint32_t *)(pinned_buf + nbytes_task_vecs + nbytes_h_center16 + num_threads * nbytes_h_norm + i * nbytes_h_out);
    }
}

buc_buffer_holder_t::~buc_buffer_holder_t() {
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
    free(d_center16);
    free(d_vec);
    free(h_norm);
    free(d_norm);
    free(d_n);
    free(h_out);
    free(d_out);
}

int buc_buffer_holder_t::device_init(int tid) {
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
    CHECK_CUDA_ERR(cudaEventCreate(&logger->h2d_norm_start[tid]));
    #endif

    if (tid == 0 || hw::gpu_ptr(tid - 1, num_threads) != device_ptr)
        CHECK_CUDA_ERR(cudaMalloc(&d_center16[device_ptr], max_batch0 * CSD16 * sizeof(int8_t)));
    CHECK_CUDA_ERR(cudaMalloc(&d_norm[tid], traits::taskVecs * sizeof(int32_t)));
    CHECK_CUDA_ERR(cudaMalloc(&d_vec[tid], traits::taskVecs * CSD16 * sizeof(int8_t)));
    CHECK_CUDA_ERR(cudaMalloc(&d_n[tid], sizeof(int32_t)));
    CHECK_CUDA_ERR(cudaMalloc(&d_out[tid], max_batch0 * (out_max_size + 1L) * sizeof(uint32_t)));
    
    
    if (tid == 0) {
        CHECK_CUDA_ERR(cudaMalloc(&state, sizeof(curandState) * buccg_blocks * buccg_threads));
        init_curand<<<buccg_blocks, buccg_threads, 0, streams[tid]>>>(state, num_threads);
        CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));
    }

    
    pthread_spin_lock(&gram_lock);
    if (tid == 0 || hw::gpu_ptr(tid - 1, num_threads) != device_ptr)
        used_gram[device_ptr] += max_batch0 * CSD16 * sizeof(int8_t);
    used_gram[device_ptr] += traits::taskVecs * sizeof(int32_t);
    used_gram[device_ptr] += traits::taskVecs * CSD16 * sizeof(int8_t);
    used_gram[device_ptr] += sizeof(int32_t);
    used_gram[device_ptr] += max_batch0 * (out_max_size + 1) * sizeof(uint32_t);
    pthread_spin_unlock(&gram_lock);

    if (used_gram[device_ptr] > bucketer->_gram_slimit) {
        lg_warn("device %d using %ld byte GRAM for buc, exceeds limit %ld", 
                hw::gpu_id_list[device_ptr], used_gram[device_ptr], bucketer->_gram_slimit);
    }

    return 0;
}

int buc_buffer_holder_t::device_done(int tid) {
    int device_ptr = hw::gpu_ptr(tid, num_threads);

    #if ENABLE_PROFILING
    for (int i = 0; i < traits::taskChunks; i++) {
        CHECK_CUDA_ERR(cudaEventDestroy(logger->h2d_start[tid][i]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->h2d_stop[tid][i]));
    }
    CHECK_CUDA_ERR(cudaEventDestroy(logger->d2h_start[tid]));
    CHECK_CUDA_ERR(cudaEventDestroy(logger->d2h_stop[tid]));
    CHECK_CUDA_ERR(cudaEventDestroy(logger->kernel_start[tid]));
    CHECK_CUDA_ERR(cudaEventDestroy(logger->h2d_norm_start[tid]));
    #endif

    if (tid == 0 || hw::gpu_ptr(tid - 1, num_threads) != device_ptr)
        CHECK_CUDA_ERR(cudaFree(d_center16[device_ptr]));
    CHECK_CUDA_ERR(cudaFree(d_vec[tid]));
    CHECK_CUDA_ERR(cudaFree(d_norm[tid]));
    CHECK_CUDA_ERR(cudaFree(d_n[tid]));
    CHECK_CUDA_ERR(cudaFree(d_out[tid]));
    
    if (tid == 0) CHECK_CUDA_ERR(cudaFree(state));
    
    CHECK_CUDA_ERR(cudaStreamDestroy(streams[tid]));
    
    pthread_spin_lock(&gram_lock);
    if (tid == 0 || hw::gpu_ptr(tid - 1, num_threads) != device_ptr)
        used_gram[device_ptr] -= max_batch0 * CSD16 * sizeof(int8_t);
    used_gram[device_ptr] -= traits::taskVecs * sizeof(int32_t);
    used_gram[device_ptr] -= traits::taskVecs * CSD16 * sizeof(int8_t);
    used_gram[device_ptr] -= sizeof(int32_t);
    used_gram[device_ptr] -= max_batch0 * (out_max_size + 1) * sizeof(uint32_t);
    pthread_spin_unlock(&gram_lock);

    return 0;
}

int buc_buffer_holder_t::center_prep(int batch0, int &first_batch) {
    /// sample centers
    int center_norm = 0;

    if (first_batch) {
        first_batch = 0;
        int64_t total_num = 0;
        uint32_t *score_stat = bucketer->_pool->score_stat;
        for (int i = 1; i < 65536; i++) total_num += score_stat[i];
        total_num *= traits::l0_center_precentile;
        int goal_score = 1;
        while (total_num > 0) total_num -= score_stat[goal_score++];
        center_norm = goal_score * 2;
    } else {
        center_norm = bucketer->_reducer->center_norm;
    }

    CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[0]));
    cudaFuncSetAttribute(buc_ctr_gen, cudaFuncAttributeMaxDynamicSharedMemorySize, buccg_shmem);
    buc_ctr_gen<<<buccg_blocks, buccg_threads, buccg_shmem>>>(d_center16[0], center_norm, state, CSD, CSD16, batch0);
    CHECK_CUDA_ERR(cudaStreamSynchronize(0));
    CHECK_CUDA_ERR(cudaMemcpy(h_center16, d_center16[0], batch0 * CSD16, cudaMemcpyDeviceToHost));

    /// copy centers to devices
    for (int i = 1; i < num_devices; i++) {
        CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[i]));
        CHECK_CUDA_ERR(cudaMemcpy(d_center16[i], h_center16, batch0 * CSD16, cudaMemcpyHostToDevice));
    }

    /// prepare other runtime data
    curr_batch0 = batch0;
    for (int i = 0; i < num_threads; i++) task_vecs[i] = 0;

    kernel = traits::kernel_chooser(batch0, CSD16);

    if (kernel == NULL) lg_err("Unsupported batch0 %d, CSD16 %d", batch0, CSD16);

    return 0;
}


int buc_buffer_holder_t::h2d(int tid, chunk_t *chunk) {
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->h2d_start[tid][logger->h2d_count[tid]], streams[tid]));
    #endif
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_vec[tid] + CSD * task_vecs[tid], chunk->vec, 
                    chunk->size * CSD, cudaMemcpyHostToDevice, streams[tid]));
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->h2d_stop[tid][logger->h2d_count[tid]], streams[tid]));
    logger->h2d_count[tid]++;
    #endif
    memcpy(h_norm[tid] + task_vecs[tid], chunk->norm, chunk->size * sizeof(int32_t));
    task_vecs[tid] += chunk->size;
    return 0;
}

int buc_buffer_holder_t::run(int tid) {
    int device_ptr = hw::gpu_ptr(tid, num_threads);
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->h2d_norm_start[tid], streams[tid]));
    #endif
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_norm[tid], h_norm[tid], task_vecs[tid] * sizeof(int32_t), 
                        cudaMemcpyHostToDevice, streams[tid]));
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_n[tid], &task_vecs[tid], sizeof(int32_t), 
                        cudaMemcpyHostToDevice, streams[tid])); 
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->kernel_start[tid], streams[tid]));
    #endif
    CHECK_CUDA_ERR(cudaMemsetAsync(d_out[tid], 0, curr_batch0 * sizeof(uint32_t), streams[tid]));
    kernel<<<traits::kernelBlocks, traits::blockThreads, traits::l0_shmem, streams[tid]>>>(
        d_out[tid], out_max_size, NULL, d_norm[tid], d_center16[device_ptr], 0, d_vec[tid], d_n[tid], alpha0, CSD, gbuc_freq, 0
    );
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->d2h_start[tid], streams[tid]));
    #endif
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_out[tid], d_out[tid], curr_batch0 * (out_max_size + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[tid]));
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->d2h_stop[tid], streams[tid]));
    #endif
    CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));
    #if ENABLE_PROFILING
    {
        float h2d_tt, d2h_tt, kernel_tt;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&h2d_tt, logger->h2d_norm_start[tid], logger->kernel_start[tid]));
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
        logger->ev_kernel_vmmas += ceil(task_vecs[tid] / 16.0) * curr_batch0 / 16.0;
        logger->ev_d2h_nbytes += curr_batch0 * (out_max_size + 1) * sizeof(uint32_t);
        logger->ev_h2d_nbytes += task_vecs[tid] * CSD * sizeof(int8_t) + task_vecs[tid] * sizeof(int32_t);
    }
    #endif
    int full_count = 0;
    for (int i = 0; i < curr_batch0; i++) {
        if (h_out[tid][i] > out_max_size) { h_out[tid][i] = out_max_size; full_count++; }
    }
    /// if (full_count > 0.5 * curr_batch0) lg_warn("%d / %d buckets full", full_count, curr_batch0);
    task_vecs[tid] = 0;
    return 0;
}

int buc_buffer_holder_t::out(int tid, int bid, int *num, int **entry) {
    if (tid < 0 || tid >= num_threads || bid < 0 || bid >= curr_batch0) return -1;

    *num = h_out[tid][bid];
    *entry = (int *)(h_out[tid] + curr_batch0 + out_max_size * bid);
    return 0;
}


buc_iterator_t::buc_iterator_t(Bucketer_t *bucketer) {
    #if ENABLE_PROFILING
    this->logger = bucketer->logger;
    #endif

    this->CSD = bucketer->_pool->CSD;
    this->pwc = bucketer->_pwc;
    this->swc = bucketer->_swc;

    long nvecs_limit     = pow(4./3., bucketer->_pool->CSD * .5) * bucketer->_size_ratio;
    num_chunk_limit      = (nvecs_limit + Pool_hd_t::chunk_max_nvecs - 1) / Pool_hd_t::chunk_max_nvecs;
    last_chunk_limit     = nvecs_limit % Pool_hd_t::chunk_max_nvecs;
    last_insert_chunk_id = 0;

    int curr_num_chunks = pwc->num_chunks();
    if (curr_num_chunks >= num_chunk_limit) {
        chunk_t *ck = pwc->fetch(curr_num_chunks - 1);
        int modified = _normalize_chunk(ck, CSD);
        
        if (curr_num_chunks > num_chunk_limit) {
            num_chunk_limit = curr_num_chunks;
            last_chunk_limit = ck->size;
        } else {
            if (last_chunk_limit < ck->size) last_chunk_limit = ck->size;
        }

        pwc->release_sync(ck->id);
    } else {
        while (pwc->num_chunks() < num_chunk_limit) pwc->create_chunk();
    }

    num_working_sol = 0;
    num_reading_sol = 0;
    reading_sol = (chunk_t **)malloc(bucketer->_num_threads * sizeof(chunk_t *));
    pthread_spin_init(&reading_sol_lock, PTHREAD_PROCESS_SHARED);
    pthread_spin_init(&pop_id_lock, PTHREAD_PROCESS_SHARED);
}

buc_iterator_t::~buc_iterator_t() {
    for (int i = 0; i < num_reading_sol; i++) swc->read_done(reading_sol[i]);
    if (num_working_sol) lg_err("nonzero num_working_sol %d, ignored", num_working_sol);
    free(reading_sol);
    pthread_spin_destroy(&reading_sol_lock);
    pthread_spin_destroy(&pop_id_lock);
}

int buc_iterator_t::reset() {
    for (int i = pwc->num_chunks() - 1; i >= 0; i--) {
        if (pwc->chunk_size(i) != 0) {
            first_empty_chunk_id = i;
            break;
        }
    }

    total_poped_fulls = 0;
    first_empty_poped = 0;
    curr_empty_id = first_empty_chunk_id;
    curr_full_id  = last_insert_chunk_id;

    for (int i = 0; i < pfch_ahead; i++) {
        int pfch_id = (curr_full_id + i) % (first_empty_chunk_id + 1);
        pwc->prefetch(pfch_id);
    }

    for (int i = 0; i < pfch_ahead; i++) {
        int pfch_id = curr_empty_id + i;
        if (pfch_id < pwc->num_chunks()) pwc->prefetch(pfch_id);
    }

    return 0;
}

chunk_t *buc_iterator_t::pop(int exist_sol, int *dst_size_limit) {
    int32_t ret_id = -1;

    pthread_spin_lock(&pop_id_lock);
    if (exist_sol && curr_empty_id < num_chunk_limit) {
        ret_id = curr_empty_id++;
        first_empty_poped = 1;
    } else if (total_poped_fulls + first_empty_poped < first_empty_chunk_id + 1) {
        ret_id = curr_full_id;
        curr_full_id = (curr_full_id + 1) % (first_empty_chunk_id + 1);
        if (ret_id == first_empty_chunk_id) {
            if (first_empty_poped) {
                ret_id = 0;
                curr_full_id = (curr_full_id + 1) % (first_empty_chunk_id + 1);
                total_poped_fulls++;
            } else {
                first_empty_poped = 1;
            }
        } else total_poped_fulls++;
    }
    pthread_spin_unlock(&pop_id_lock);

    if (dst_size_limit) {
        *dst_size_limit = ret_id == num_chunk_limit - 1 ? last_chunk_limit : Pool_hd_t::chunk_max_nvecs;
    }

    if (ret_id > first_empty_chunk_id && ret_id + pfch_ahead < num_chunk_limit) pwc->prefetch(ret_id + pfch_ahead);
    else {
        int pfch_id = (curr_full_id + pfch_ahead) % (first_empty_chunk_id + 1);
        pwc->prefetch(pfch_id);
    }

    if (ret_id == -1) return NULL;
    #if ENABLE_PROFILING
    struct timeval fetch_start, fetch_end;
    gettimeofday(&fetch_start, NULL);
    #endif
    chunk_t *ret = pwc->fetch(ret_id);
    #if ENABLE_PROFILING
    gettimeofday(&fetch_end, NULL);
    logger->ev_ld_stall_us += (fetch_end.tv_sec - fetch_start.tv_sec) * 1000000 + fetch_end.tv_usec - fetch_start.tv_usec;
    if (ret) logger->ev_ld_chunks++;
    #endif
    return ret;
}

chunk_t *buc_iterator_t::pop_sol(int *no_more_sol) {
    chunk_t *ret = NULL;
    pthread_spin_lock(&reading_sol_lock);
    if (num_reading_sol) {
        ret = reading_sol[--num_reading_sol];
        num_working_sol++;
    } 
    pthread_spin_unlock(&reading_sol_lock);
    
    if (!ret) {
        #if ENABLE_PROFILING
        struct timeval fetch_start, fetch_end;
        gettimeofday(&fetch_start, NULL);
        #endif
        ret = swc->fetch_for_read();
        #if ENABLE_PROFILING
        gettimeofday(&fetch_end, NULL);
        logger->ev_ld_stall_us += (fetch_end.tv_sec - fetch_start.tv_sec) * 1000000 + fetch_end.tv_usec - fetch_start.tv_usec;
        if (ret) logger->ev_ld_chunks++;
        #endif
        if (ret) {
            pthread_spin_lock(&reading_sol_lock);
            num_working_sol++;
            pthread_spin_unlock(&reading_sol_lock);
        } else if (no_more_sol) *no_more_sol = 1;
    }

    if (ret) {
        if (_normalize_chunk(ret, CSD)) {
            lg_warn("chunk %d from swc not normalized", ret->id);
        }
    }

    return ret;
}

int buc_iterator_t::rel_sol(chunk_t *sol) {
    if (sol->size == 0) {
        swc->read_done(sol);
        pthread_spin_lock(&reading_sol_lock);
        num_working_sol--;
        pthread_spin_unlock(&reading_sol_lock);
    } else {
        pthread_spin_lock(&reading_sol_lock);
        reading_sol[num_reading_sol++] = sol;
        num_working_sol--;
        pthread_spin_unlock(&reading_sol_lock);
    }

    return 0;
}

void buc_iterator_t::inserted(int chunk_id) {
    if (chunk_id >= first_empty_chunk_id) return;
    last_insert_chunk_id = chunk_id;
    return;
}


Bucketer_t::Bucketer_t(Pool_hd_t *pool, bwc_manager_t *bwc, swc_manager_t *swc, ut_checker_t *ut_checker) {
    this->set_pool(pool);
    this->set_bwc_manager(bwc);
    this->set_swc_manager(swc);
    this->set_ut_checker(ut_checker);
    this->_improve_ratio     = bgj_default_improve_ratio;
    this->_saturation_ratio  = bgj_default_saturation_ratio;
    this->_saturation_radius = bgj_default_saturation_radius;


    #if ENABLE_PROFILING
    this->logger = new logger_t();
    this->logger->bucketer = this;
    logger->clear();
    #endif    
}

Bucketer_t::~Bucketer_t() {
    if (_buc_pool) {
        for (int i = 0; i < _num_threads; i++) delete _buc_pool[i];
        free(_buc_pool);
        _buc_pool = NULL;
    }
    #if ENABLE_PROFILING
    delete this->logger;
    #endif
}

int Bucketer_t::set_alpha0(float alpha0) {
    this->_alpha0 = alpha0;
    return 0;
}

int Bucketer_t::set_batch0(long batch0) {
    this->_min_batch0 = batch0;
    return 0;
}

int Bucketer_t::set_num_threads(int num_threads) {
    if (_buc_pool) {
        for (int i = 0; i < this->_num_threads; i++) delete _buc_pool[i];
        free(_buc_pool);
    }
    this->_num_threads = num_threads;
    
    _buc_pool = (thread_pool::thread_pool **)malloc(num_threads * sizeof(thread_pool::thread_pool *));
    for (int i = 0; i < num_threads; i++) _buc_pool[i] = new thread_pool::thread_pool(1);

    return 0;
}

int Bucketer_t::auto_bgj_params_set(int bgj) {
    int ret = 0;

    this->_bgj = bgj;

    int type = bgj;
    
    #define BGJ_SIZE_RATIO (type == 1 ? BGJ1_SIZE_RATIO : type == 2 ? BGJ2_SIZE_RATIO : type == 3 ? BGJ3_SIZE_RATIO : type == 4 ? BGJ3L_SIZE_RATIO : BGJ4_SIZE_RATIO)
    #define BGJ_L0_BATCH_RATIO (type == 1 ? BGJ1_L0_BATCH_RATIO : type == 2 ? BGJ2_L0_BATCH_RATIO : type == 3 ? BGJ3_L0_BATCH_RATIO : type == 4 ? BGJ3L_L0_BATCH_RATIO : BGJ4_L0_BATCH_RATIO)
    #define BGJ_L0_MIN_ALPHA0 (type == 1 ? BGJ1_L0_MIN_ALPHA0 : type == 2 ? BGJ2_L0_MIN_ALPHA0 : type == 3 ? BGJ3_L0_MIN_ALPHA0 : type == 4 ? BGJ3L_L0_MIN_ALPHA0 : BGJ4_L0_MIN_ALPHA0)
    #define BGJ_L0_MAX_ALPHA0 (type == 1 ? BGJ1_L0_MAX_ALPHA0 : type == 2 ? BGJ2_L0_MAX_ALPHA0 : type == 3 ? BGJ3_L0_MAX_ALPHA0 : type == 4 ? BGJ3L_L0_MAX_ALPHA0 : BGJ4_L0_MAX_ALPHA0)

    if (_ssd_slimit  == 0) {
        this->_ssd_slimit  = BWC_SSD_SLIMIT;
        uint64_t dim_max_ssd = 2.5 * BGJ_SIZE_RATIO * pow(4./3., _pool->CSD * .5) * (Pool_hd_t::vec_nbytes + 14);
        if (_ssd_slimit > dim_max_ssd) this->_ssd_slimit = dim_max_ssd;
    }
    if (_dram_slimit == 0) this->_dram_slimit = BWC_DRAM_SLIMIT;
    if (_gram_slimit == 0) this->_gram_slimit = BUC_GRAM_SLIMIT;
    if (_size_ratio == .0) this->_size_ratio  = BGJ_SIZE_RATIO;

    /// max_batch0
    const int cache_for_prefetch = bwc_manager_t::bwc_auto_prefetch_for_read * 
                                    bwc_manager_t::bwc_auto_prefetch_for_read_depth + 
                                    bwc_manager_t::bwc_auto_prefetch_for_write;
    const int cache_for_bucketer = _bwc->max_cached_chunks() - cache_for_prefetch;
    int _exp_max_batch0 = traits::l0_max_batch0_under(cache_for_bucketer < BGJ_L0_MAX_BATCH0 ?
                                                    cache_for_bucketer : BGJ_L0_MAX_BATCH0);
    if (this->_max_batch0) this->_max_batch0 = _exp_max_batch0 < this->_max_batch0 ? _exp_max_batch0 : this->_max_batch0;
    else this->_max_batch0 = _exp_max_batch0;

    /// min_batch0 && alpha0 && num_buc_slimit
    double exp_vec_nbytes = 14. + _pool->CSD;
    double exp_pool_nvecs = pow(4./3., _pool->CSD * .5) * _size_ratio;
    double ma0_buc_nvecs  = traits::l0_buc_ratio_estimate(BGJ_L0_MAX_ALPHA0, _pool->CSD) * exp_pool_nvecs;
    double ma0_batch0     = traits::l0_max_batch0_under(_ssd_slimit / ma0_buc_nvecs / exp_vec_nbytes * BGJ_L0_BATCH_RATIO);
    double exp_buc_nbytes = _ssd_slimit / ma0_batch0 * BGJ_L0_BATCH_RATIO;
    double exp_buc_nvecs  = exp_buc_nbytes / exp_vec_nbytes;
    double exp_alpha0     = traits::l0_buc_alpha0_estimate(exp_buc_nvecs / exp_pool_nvecs, _pool->CSD, BGJ_L0_MIN_ALPHA0, BGJ_L0_MAX_ALPHA0);
    if (this->_alpha0 != 0.0) exp_alpha0 = this->_alpha0;
    if (!_min_batch0) _min_batch0 = ma0_batch0;
    this->_alpha0         = exp_alpha0 > BGJ_L0_MIN_ALPHA0 ? exp_alpha0 : BGJ_L0_MIN_ALPHA0;
    this->_min_batch0     = ma0_batch0 < _max_batch0 ? ma0_batch0 : _max_batch0;
    if (!this->_num_buc_slimit)
        _num_buc_slimit = floor(_ssd_slimit / (exp_pool_nvecs * traits::l0_buc_ratio_estimate(_alpha0, _pool->CSD) * exp_vec_nbytes));
    if (_num_buc_slimit > bwc_manager_t::bwc_max_buckets - RED_MAX_NUM_THREADS) 
        _num_buc_slimit = bwc_manager_t::bwc_max_buckets - RED_MAX_NUM_THREADS;

    if (!this->_num_threads) this->set_num_threads(traits::num_threads(_pool->CSD, _pool->ESD, _alpha0, _max_batch0));

    if (bgj == 4) {
        if (_max_batch0 > 64) _max_batch0 = 64;
        if (_min_batch0 > 64) _min_batch0 = 64;
        if (_num_buc_slimit > 130) _num_buc_slimit = 130;
    }

    if (_min_batch0 < BGJ_L0_MIN_BATCH0) {
        lg_err("min_batch0 too small %ld, change to 16", _min_batch0);
        _min_batch0 = 16;
    }

    lg_dbg("#threads %ld, alpha0 %.3f, batch0 in [%ld, %ld], #buc <= %d(%.2f TB, %.2f GB)", 
            _num_threads, _alpha0, _min_batch0, _max_batch0, _num_buc_slimit, _ssd_slimit / 1e12, _ssd_slimit / 1e9 / _num_buc_slimit);

    return ret;
}

int Bucketer_t::run() {
    if (_buc_buf) delete _buc_buf;
    _buc_buf = new buc_buffer_holder_t(this);

    /// prepare runtime data
    flag = 0;
    buc_id = (int32_t *) malloc(_max_batch0 * sizeof(int32_t));
    ctr_record = (int8_t *) malloc(BWC_MAX_BUCKETS * Pool_hd_t::vec_nbytes);
    buc_iter = new buc_iterator_t(this);
    pthread_spin_init(&score_stat_lock, PTHREAD_PROCESS_SHARED);

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

    int first_batch = 1;

    for (;;) {
        std::unique_lock<std::mutex> buc_lock(_buc_mtx);
        #if 0
        _buc_cv.wait(buc_lock, [this] {
            int bwc_num_ready = _bwc->num_ready();
            return (_num_buc_slimit - bwc_num_ready > _min_batch0 && bwc_num_ready < 2 * _min_batch0) ||
                   (flag & flag_stuck);
        });
        #else
        for (;;) {
            int bwc_num_ready = _bwc->num_ready();
            if ((_num_buc_slimit - bwc_num_ready > _min_batch0 && bwc_num_ready < 2 * _min_batch0) ||
                (flag & flag_stuck)) break;
            _buc_cv.wait_for(buc_lock, std::chrono::milliseconds(5000));
        }
        #endif
        buc_lock.unlock();

        if (flag & flag_stuck) {
            _signal_buc_done();
            break;
        }

        long swc_rne = _swc->ready_nvecs_estimate();
        int replace_th = traits::l0_replace_threshold(_pool, swc_rne, _improve_ratio);
        #if ENABLE_PROFILING
        long old_oss = logger->ev_old_score_sum;
        long old_nss = logger->ev_new_score_sum;
        #endif

        int batch0 = traits::l0_max_batch0_under(_num_buc_slimit - _bwc->num_ready());
        if (batch0 > _max_batch0) batch0 = _max_batch0;

        #if ENABLE_PROFILING
        struct timeval batch_start;
        gettimeofday(&batch_start, NULL);
        #endif

        for (int i = 0; i < batch0; i++) buc_id[i] = _bwc->push_bucket();
        
        _buc_buf->center_prep(batch0, first_batch);

        lg_dbg("new batch (%d) started, current #buc: %ld / %ld, #sol %ld", batch0, _bwc->num_ready(), _num_buc_slimit, swc_rne);

        for (int i = 0; i < batch0; i++) {
            int bid = buc_id[i];
            memcpy(ctr_record + bid * Pool_hd_t::vec_nbytes, _buc_buf->h_center16 + i * _buc_buf->CSD16, _buc_buf->CSD16);
            memset(ctr_record + bid * Pool_hd_t::vec_nbytes + _buc_buf->CSD16, 0, Pool_hd_t::vec_nbytes - _buc_buf->CSD16);
        }

        buc_iter->reset();
        
        for (int tid = 0; tid < _num_threads; tid++) {
            _buc_pool[tid]->push([this, tid, replace_th, batch0] { _batch(tid, replace_th, batch0); });
        }
        for (int tid = 0; tid < _num_threads; tid++) {
            _buc_pool[tid]->wait_sleep();
        }

        _update_goal();
        if (_improve_ratio < 0.849999) _improve_ratio += 0.02;

        #if ENABLE_PROFILING
        struct timeval batch_end;
        gettimeofday(&batch_end, NULL);
        logger->ev_batch_num++;
        logger->ev_batch_us += (batch_end.tv_sec - batch_start.tv_sec) * 1000000 + batch_end.tv_usec - batch_start.tv_usec;
        logger->ev_bk0_num += batch0;
        for (int i = 0; i < batch0; i++) {
            int bnc = _bwc->bucket_num_chunks(buc_id[i]);
            logger->ev_ld_chunks += bnc;
            logger->ev_st_chunks += bnc;
        }
        old_oss = logger->ev_old_score_sum - old_oss;
        old_nss = logger->ev_new_score_sum - old_nss;
        #endif

        lg_dbg("batch done, current #buc: %ld / %ld, goal_norm %d, goal_score %d(%.4f)", 
                _bwc->num_ready(), _num_buc_slimit, _reducer->goal_norm, _reducer->goal_score, old_nss / (float)old_oss);

        for (int i = 0; i < batch0; i++) _bwc->bucket_finalize(buc_id[i]);    

        _signal_new_buc_ready();    

        if (_sieve_is_over() || (flag & flag_stuck)) {
            _signal_buc_done();
            break;
        }
    }

    if (flag & flag_stuck) {
        lg_warn("sieving stucked, waiting for reducer done");
    } else {
        lg_dbg("no more bucket, waiting for reducer done");
    }

    std::unique_lock<std::mutex> buc_lock(_buc_mtx);
    _buc_cv.wait(buc_lock, [this] { return (flag & flag_final); });

    lg_dbg("reducer done, inserting remaining solutions");

    int replace_th = traits::l0_replace_threshold(_pool, _swc->ready_nvecs_estimate(), _improve_ratio);

    buc_iter->reset();

    for (int tid = 0; tid < _num_threads; tid++) {
        _buc_pool[tid]->push([this, replace_th, tid] { _batch(tid, replace_th, 0); });
    }
    for (int tid = 0; tid < _num_threads; tid++) {
        _buc_pool[tid]->wait_sleep();
    }

    for (int tid = 0; tid < _num_threads; tid++) {
        _buc_pool[tid]->push([this, tid] { _buc_buf->device_done(tid); });
    }
    for (int tid = 0; tid < _num_threads; tid++) {
        _buc_pool[tid]->wait_sleep();
    }

    _ut_checker->input_done();

    _ut_checker->wait_work();

    /// destroy runtime data
    free(buc_id);
    free(ctr_record);
    delete buc_iter;
    pthread_spin_destroy(&score_stat_lock);

    delete _buc_buf;
    _buc_buf = NULL;

    lg_report();

    return (flag & flag_stuck) ? -1 : 0;
}

int Bucketer_t::_batch(int tid, int replace_th, int batch0) {
    constexpr long chunk_max_nvecs = Pool_hd_t::chunk_max_nvecs;
    constexpr unsigned int taskChunks = traits::taskChunks;

    if (batch0) cudaFuncSetAttribute(_buc_buf->kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, traits::l0_shmem);

    const int CSD = _pool->CSD;

    int32_t local_score_stat[65536] = {};
    int32_t num_to_remove_uid = 0;
    uint64_t to_remove_uid[65536];

    int32_t goal_score = _reducer->goal_score;

    #if ENABLE_PROFILING
    uint64_t wasted = 0;
    uint64_t inserted = 0;
    uint64_t replaced = 0;
    uint64_t old_score_sum = 0;
    uint64_t new_score_sum = 0;
    #endif
    
    for (int batch_done = 0;;) {
        int task_chunks = 0, task_vecs = 0, no_more_sol = 0;
        chunk_t *working_chunk[taskChunks] = {};
        int32_t chunk_modified[taskChunks] = {};
        chunk_t *src = NULL, *dst = NULL;
        int32_t dst_pos = 0, dst_size_limit = 0;
        
        while (task_chunks < taskChunks) {
            if (!src && !no_more_sol) src = buc_iter->pop_sol(&no_more_sol); 
            if (!dst) { 
                dst = buc_iter->pop(src ? 1 : 0, &dst_size_limit); 
                dst_pos = 0; 
                if (!dst) { batch_done = 1; break; }
                else chunk_modified[task_chunks] = _normalize_chunk(dst, _pool->CSD);
            }

            if (src) {
                int to_move = src->size < dst_size_limit - dst->size ?
                              src->size : dst_size_limit - dst->size;
                if (to_move >= 0) {
                    memcpy(dst->u + dst->size, src->u + src->size - to_move, sizeof(uint64_t) * to_move);
                    memcpy(dst->norm + dst->size, src->norm + src->size - to_move, sizeof(int32_t) * to_move);
                    memcpy(dst->score + dst->size, src->score + src->size - to_move, sizeof(uint16_t) * to_move);
                    memcpy(dst->vec + CSD * dst->size, src->vec + CSD * (src->size - to_move), CSD * to_move);
                    memset(src->norm + src->size - to_move, 0, sizeof(int32_t) * to_move);
                    memset(src->score + src->size - to_move, 0, sizeof(uint16_t) * to_move);
                    for (int i = 0; i < to_move; i++) local_score_stat[dst->score[dst->size + i]]++;
                    dst->size += to_move;
                    src->size -= to_move;
                    chunk_modified[task_chunks] |= 2;
                    #if ENABLE_PROFILING
                    inserted += to_move;
                    #endif
                }

                int src_old_size = src->size;

                while (src->size) {
                    while (dst_pos < dst_size_limit && dst->score[dst_pos] < replace_th) dst_pos++;
                    if (dst_pos >= dst_size_limit) break;
                    if (src->score[src->size - 1] >= dst->score[dst_pos]) { 
                        #if ENABLE_PROFILING
                        wasted++;
                        #endif
                        to_remove_uid[num_to_remove_uid++] = src->u[--src->size];
                        if (dst->score[dst_pos] < goal_score) dst_pos++;
                        continue;
                    }
                    #if ENABLE_PROFILING
                    replaced++;
                    old_score_sum += dst->score[dst_pos];
                    new_score_sum += src->score[src->size - 1];
                    #endif
                    to_remove_uid[num_to_remove_uid++] = dst->u[dst_pos];
                    local_score_stat[dst->score[dst_pos]]--;
                    local_score_stat[src->score[--src->size]]++;
                    dst->u[dst_pos] = src->u[src->size];
                    dst->norm[dst_pos] = src->norm[src->size];
                    dst->score[dst_pos] = src->score[src->size];
                    memcpy(dst->vec + CSD * dst_pos, src->vec + CSD * src->size, CSD);
                    dst_pos++;
                }

                if (src->size < src_old_size) {
                    memset(src->norm + src->size, 0, sizeof(int32_t) * (src_old_size - src->size));
                    memset(src->score + src->size, 0, sizeof(uint16_t) * (src_old_size - src->size));
                    chunk_modified[task_chunks] |= 2;
                }
            }
            
            if (dst_pos >= dst_size_limit || !src) {
                if (batch0) {
                    _buc_buf->h2d(tid, dst);
                    task_vecs += dst->size;
                    working_chunk[task_chunks++] = dst;
                } else {
                    task_vecs += dst->size;
                    if (chunk_modified[task_chunks] & 2) buc_iter->inserted(dst->id);
                    if (chunk_modified[task_chunks++]) {
                        _pwc->release_sync(dst->id);
                        #if ENABLE_PROFILING
                        logger->ev_st_chunks++;
                        #endif
                    } else _pwc->release(dst->id);
                }
                dst = NULL;
            }
            
            if (src) {
                if (src->size == 0) {
                    buc_iter->rel_sol(src);
                    src = NULL;
                }
            }

            if (num_to_remove_uid >= sizeof(to_remove_uid) / sizeof(uint64_t) - 2 * chunk_max_nvecs) {
                _ut_checker->task_commit(to_remove_uid, num_to_remove_uid);
                num_to_remove_uid = 0;
            }
        }
        if (src) buc_iter->rel_sol(src);

        if (batch0) {
            _buc_buf->run(tid);
            int32_t working_chunk_size[taskChunks] = {};
            for (int i = 0; i < task_chunks; i++) working_chunk_size[i] = working_chunk[i]->size;
            random_interval_iter_t iiter(batch0);

            for (int _i = 0; _i < batch0; _i++) {
                int i = iiter.pop();
                int to_add, *entry;
                _buc_buf->out(tid, i, &to_add, &entry);
                while (to_add > 0) {
                    #if ENABLE_PROFILING
                    struct timeval fetch_start, fetch_end;
                    gettimeofday(&fetch_start, NULL);
                    #endif
                    chunk_t *_dst = _bwc->fetch_for_write(buc_id[i]);
                    #if ENABLE_PROFILING
                    gettimeofday(&fetch_end, NULL);
                    logger->ev_ld_stall_us += (fetch_end.tv_sec - fetch_start.tv_sec) * 1000000 + fetch_end.tv_usec - fetch_start.tv_usec;
                    #endif
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
                        traits::l0_sign_copy_epi8(_dst->vec + CSD * (_dst->size + j), _src->vec + CSD * pos, CSD, sign);
                    }
                    to_add -= to_move;
                    _dst->size += to_move;
                    _bwc->write_done(_dst, buc_id[i]);
                }
            }

            for (int i = 0; i < task_chunks; i++) {
                if (chunk_modified[i] & 2) buc_iter->inserted(working_chunk[i]->id);
                if (chunk_modified[i]) {
                    _pwc->release_sync(working_chunk[i]->id);
                    #if ENABLE_PROFILING
                    logger->ev_st_chunks++;
                    #endif
                } else _pwc->release(working_chunk[i]->id);
            }
        }

        if (batch_done || (batch0 == 0 && no_more_sol)) break;
    }

    #if ENABLE_PROFILING
    logger->ev_wasted_num += wasted;
    logger->ev_inserted_num += inserted;
    logger->ev_replaced_num += replaced;
    logger->ev_old_score_sum += old_score_sum;
    logger->ev_new_score_sum += new_score_sum;
    #endif

    if (!batch0) {
        int no_more_sol = 0;
        chunk_t *src = NULL;
        for (;;) {
            src = buc_iter->pop_sol(&no_more_sol); 
            if (!src && no_more_sol) break;
            if (src) {
                for (int i = 0; i < src->size; i++) {
                    to_remove_uid[num_to_remove_uid++] = src->u[i];
                }
                if (num_to_remove_uid >= sizeof(to_remove_uid) / sizeof(uint64_t) - 2 * chunk_max_nvecs) {
                    _ut_checker->task_commit(to_remove_uid, num_to_remove_uid);
                    num_to_remove_uid = 0;
                }
                src->size = 0;
                buc_iter->rel_sol(src);
                src = NULL;
            }
        }
    }

    if (num_to_remove_uid) {
        _ut_checker->task_commit(to_remove_uid, num_to_remove_uid);
    }

    pthread_spin_lock(&score_stat_lock);
    for (int i = 0; i < 65536; i++) _pool->score_stat[i] += local_score_stat[i];
    pthread_spin_unlock(&score_stat_lock);
    return 0;
}

int Bucketer_t::_update_goal() {
    int64_t goal_num = _pwc->num_vec() * _improve_ratio;
    int64_t center_num = goal_num / _improve_ratio * traits::l0_center_precentile;

    int32_t goal_score = 1;
    for (int64_t curr_num = 0;;) {
        curr_num += _pool->score_stat[goal_score++];
        if (curr_num >= goal_num || goal_score == 65535 - 1) break;
    }
    int32_t goal_norm = goal_score * 2 * (_pool->ESD ? 1.12 : 1.003);

    int32_t center_score = 1;
    for (int64_t curr_num = 0;;) {
        curr_num += _pool->score_stat[center_score++];
        if (curr_num >= center_num || center_score == 65535 -1) break;
    }
    int32_t center_norm = center_score * 2 * (_pool->ESD ? 1.12 : 1.003);

    _reducer->goal_score = goal_score;
    _reducer->center_norm = center_norm;
    _reducer->goal_norm = goal_norm;

    return 0;
}

int Bucketer_t::_sieve_is_over() {
    int64_t goal_num = pow(_saturation_radius, _pool->CSD * .5) * .5 * _saturation_ratio;
    int32_t goal_score = round(_pool->gh2_scaled() * .25 * _saturation_radius);

    for (int i = goal_score; i > 0; i--) {
        goal_num -= _pool->score_stat[i];
        if (goal_num <= 0) return 1;
    }

    return 0;
}

int Bucketer_t::_signal_buc_done() {
    std::unique_lock<std::mutex> lock(_reducer->_red_mtx);
    _reducer->flag |= Reducer_t::flag_stop;
    _reducer->_red_cv.notify_all();
    return 0;
}

int Bucketer_t::_signal_new_buc_ready() {
    std::unique_lock<std::mutex> lock(_reducer->_red_mtx);
    _reducer->_red_cv.notify_all();
    return 0;
}

int Bucketer_t::_num_ready_buckets() {
    return _bwc->num_ready();
}

red_buffer_holder_t::red_buffer_holder_t(Reducer_t *reducer) {
    #if ENABLE_PROFILING
    logger = reducer->logger;
    #endif

    this->reducer = reducer;

    /// fixed during sieving
    this->CSD               = reducer->_pool->CSD;
    this->ESD               = reducer->_pool->ESD;
    this->CSD16             = (CSD + 15) / 16 * 16 < RED_MIN_CSD16 ? RED_MIN_CSD16 : (CSD + 15) / 16 * 16;
    this->ESD8              = traits::red_ESD8(CSD, ESD);
    this->strategy          = reducer->_strategy;
    this->sbuc_freq         = traits::red_sbuc_freq(CSD, strategy);
    this->gbuc_freq         = traits::red_gbuc_freq(CSD, strategy);
    this->bk1_gbuc_freq     = traits::bk1_gbuc_freq(CSD, strategy);
    this->bk2_gbuc_freq     = traits::bk2_gbuc_freq(CSD, strategy);
    this->bk3_gbuc_freq     = traits::bk3_gbuc_freq(CSD, strategy);
    this->buc_max_size      = traits::buc_max_size(CSD, ESD, strategy);
    this->out_max_size      = traits::red_out_max_size(CSD, ESD, strategy);
    this->flt_out_max_size  = traits::flt_out_max_size(CSD, ESD, strategy);
    this->bk1_max_size      = traits::bk1_max_size(CSD, ESD, strategy);
    this->bk2_max_size      = traits::bk2_max_size(CSD, ESD, strategy);
    this->bk3_max_size      = traits::bk3_max_size(CSD, ESD, strategy);
    this->l1_out_max_size   = traits::l1_out_max_size(CSD, ESD, strategy);
    this->batch1            = reducer->_batch1;
    this->batch2            = reducer->_batch2;
    this->batch3            = reducer->_batch3;
    this->alpha0            = reducer->_bucketer->_alpha0;
    this->alpha1            = reducer->_alpha1;
    this->alpha2            = reducer->_alpha2;
    this->alpha3            = reducer->_alpha3;
    this->bgj3l_repeat      = reducer->bgj3l_repeat;
    this->bgj4_repeat       = reducer->bgj4_repeat;
    local_data              = (local_data_t **) malloc(MAX_NUM_DEVICE * sizeof(local_data_t *));

    #if USE_GRAPH
    this->out_max_size_i    = this->out_max_size;
    this->bk1_max_size_i    = this->bk1_max_size;
    this->bk2_max_size_i    = this->bk2_max_size;
    this->bk3_max_size_i    = this->bk3_max_size;
    #endif

    /// kernel choice
    pk_kernel = traits::pk_kernel_chooser(CSD16);
    if (!pk_kernel) lg_err("pk_kernel_chooser: unsupported CSD16 %d", CSD16);
    upk_kernel = traits::upk_kernel_chooser(CSD16);
    if (!upk_kernel) lg_err("upk_kernel_chooser: unsupported CSD16 %d", CSD16);
    rpp_kernel = traits::rpp_kernel_chooser(CSD16, strategy);
    if (!rpp_kernel) lg_err("rpp_kernel_chooser: unsupported CSD16 %d, strategy %d", CSD16, strategy);
    #if USE_GRAPH
    red_kernel = traits::red_kernel_chooser(CSD16, strategy, sbuc_freq, gbuc_freq);
    if (!red_kernel) lg_err("red_kernel_chooser: unsupported CSD16 %d, strategy %d, sbuc_freq %d, gbuc_freq %d", CSD16, strategy, sbuc_freq, gbuc_freq);
    #else
    if (strategy == Reducer_t::strategy_bgj1) {
        rd1_kernel = traits::rd1_kernel_chooser(CSD16, strategy, sbuc_freq, gbuc_freq);
        if (!rd1_kernel) lg_err("rd1_kernel_chooser: unsupported CSD16 %d, strategy %d, sbuc_freq %d, gbuc_freq %d", CSD16, strategy, sbuc_freq, gbuc_freq);
    }
    if (strategy == Reducer_t::strategy_bgj2 || strategy == Reducer_t::strategy_bgj3 || 
        strategy == Reducer_t::strategy_bgj3l || strategy == Reducer_t::strategy_bgj4) {
        red_kernel = traits::red_kernel_chooser(CSD16, strategy, sbuc_freq, gbuc_freq);
        if (!red_kernel) lg_err("red_kernel_chooser: unsupported CSD16 %d, strategy %d, sbuc_freq %d, gbuc_freq %d", CSD16, strategy, sbuc_freq, gbuc_freq);
    }
    #endif
    fpv_kernel = traits::fpv_kernel_chooser(CSD16);
    if (!fpv_kernel) lg_err("fpv_kernel_chooser: unsupported CSD16 %d", CSD16);
    flt_kernel = traits::flt_kernel_chooser(CSD16, ESD8);
    if (!flt_kernel) lg_err("flt_kernel_chooser: unsupported CSD16 %d, ESD8 %d", CSD16, ESD8);
    fcs_kernel = traits::fcs_kernel_chooser(CSD16);
    if (!fcs_kernel) lg_err("fcs_kernel_chooser: unsupported CSD16 %d", CSD16);
    
    if (strategy == Reducer_t::strategy_bgj2 || strategy == Reducer_t::strategy_bgj3 ||
        strategy == Reducer_t::strategy_bgj3l || strategy == Reducer_t::strategy_bgj4) {
        bk1_kernel = traits::bk1_kernel_chooser(CSD16, batch1, strategy);
        if (!bk1_kernel) lg_err("bk1_kernel_chooser: unsupported CSD16 %d, batch1 %d, strategy %d", CSD16, batch1, strategy);
    }
    if (strategy == Reducer_t::strategy_bgj3 || strategy == Reducer_t::strategy_bgj3l || strategy == Reducer_t::strategy_bgj4) {
        bk2_kernel = traits::bk2_kernel_chooser(CSD16, batch2, strategy);
        if (!bk2_kernel) lg_err("bk2_kernel_chooser: unsupported CSD16 %d, batch2 %d, strategy %d", CSD16, batch2, strategy);
    }
    if (strategy == Reducer_t::strategy_bgj3l || strategy == Reducer_t::strategy_bgj4) {
        rep_kernel = traits::rep_kernel_chooser(CSD16, batch2, strategy);
        if (!rep_kernel) lg_err("rep_kernel_chooser: unsupported CSD16 %d, batch2 %d, strategy %d", CSD16, batch2, strategy);
    }
    if (strategy == Reducer_t::strategy_bgj4) {
        bk3_kernel = traits::bk3_kernel_chooser(CSD16, batch3, strategy);
        if (!bk3_kernel) lg_err("bk3_kernel_chooser: unsupported CSD16 %d, batch3 %d, strategy %d", CSD16, batch3, strategy);
    }

    /// thread & device info
    int _num_devices;
    _num_devices = hw::gpu_num;
    pthread_spin_init(&gram_lock, PTHREAD_PROCESS_SHARED);
    this->num_devices    = _num_devices;
    this->num_threads    = reducer->_num_threads;
    this->threads_per_buc = reducer->_threads_per_buc;
    long tpb = threads_per_buc ? threads_per_buc : 1;
    this->streams        = (cudaStream_t *) malloc(num_threads * sizeof(cudaStream_t));
    this->used_gram      = (long *) calloc(num_devices, sizeof(long));
    this->sstreams    = (cudaStream_t *) malloc(num_threads * tpb * sizeof(cudaStream_t));

    #if ENABLE_PROFILING
    this->logger->num_devices = this->num_devices;
    this->logger->num_threads = this->num_threads;
    this->logger->strategy = this->strategy;
    this->logger->CSD16 = this->CSD16;
    this->logger->chunk_nbytes = (14 + CSD) * Pool_hd_t::chunk_max_nvecs;
    #endif

    for (int i = 0; i < num_devices; i++) {
        CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[i]));
        check_traits::_prep_device_local_data(CSD16, ESD8, local_data[i], reducer->_pool);
    }

    if (strategy == Reducer_t::strategy_bgj1) {
        lg_dbg("#device %ld, CSD16 %ld, ESD8 %ld, #rout <= %ld, #fout <= %ld, sbuc_freq %ld, gbuc_freq %ld", 
                 num_devices, CSD16, ESD8, out_max_size, flt_out_max_size, sbuc_freq, gbuc_freq);
    }
    if (strategy == Reducer_t::strategy_bgj2) {
        lg_dbg("#device %ld, CSD16 %ld, ESD8 %ld, #rout <= %ld, #fout <= %ld, sbuc_freq %ld, gbuc_freq %ld, |bk1| = %ld", 
                num_devices, CSD16, ESD8, out_max_size, flt_out_max_size, sbuc_freq, gbuc_freq, bk1_max_size);
    }
    if (strategy == Reducer_t::strategy_bgj3 || strategy == Reducer_t::strategy_bgj3l) {
        lg_dbg("#device %ld, CSD16 %ld, ESD8 %ld, #rout <= %ld, #fout <= %ld, sfreq %ld, gfreq %ld, |bk1| = %ld(%ld), |bk2| = %ld", 
                num_devices, CSD16, ESD8, out_max_size, flt_out_max_size, sbuc_freq, gbuc_freq, bk1_max_size, l1_out_max_size, bk2_max_size);
    }
    if (strategy == Reducer_t::strategy_bgj4) {
        lg_dbg("#device %ld, CSD16 %ld, ESD8 %ld, #rout <= %ld, #fout <= %ld, sfreq %ld, gfreq %ld, |bk1| = %ld(%ld), |bk2| = %ld, |bk3| = %ld", 
                num_devices, CSD16, ESD8, out_max_size, flt_out_max_size, sbuc_freq, gbuc_freq, bk1_max_size, l1_out_max_size, bk2_max_size, bk3_max_size);
    }
        

    /// runtime data
    long nbytes_task_vecs       = (strategy == Reducer_t::strategy_bgj4 || strategy == Reducer_t::strategy_bgj3l) ? 
                                  ceil256(num_threads * tpb * sizeof(int)) : ceil256(num_threads * sizeof(int));
    long nbytes_buc_vecs        = (strategy == Reducer_t::strategy_bgj4 || strategy == Reducer_t::strategy_bgj3l) ? 
                                  ceil256(num_threads * tpb * sizeof(int)) : ceil256(num_threads * sizeof(int));
    long nbytes_h_num_flt_out   = ceil256(sizeof(int));
    long nbytes_h_vec_out       = ceil256(flt_out_max_size * CSD16 * sizeof(int8_t));
    long nbytes_h_score_out     = ceil256(flt_out_max_size * sizeof(uint16_t));
    long nbytes_h_norm_out      = ceil256(flt_out_max_size * sizeof(int32_t));
    long nbytes_h_u_out         = ceil256(flt_out_max_size * sizeof(uint64_t));
    long nbytes_h_norm          = ceil256(traits::taskVecs * sizeof(int32_t));
    long nbytes_h_repeat_buf    = (strategy == Reducer_t::strategy_bgj4 || strategy == Reducer_t::strategy_bgj3l) ? 
                                  ceil256(8 * traits::taskVecs) : 0;
    long nbytes_h_ct1           = (strategy == Reducer_t::strategy_bgj4 || strategy == Reducer_t::strategy_bgj3l) ? 
                                  ceil256(batch1 * Pool_hd_t::vec_nbytes) : 0;
    long nbytes_h_bko           = (strategy == Reducer_t::strategy_bgj4 || strategy == Reducer_t::strategy_bgj3l) ? 
                                  ceil256(batch1 * (2L * l1_out_max_size + 1L) * sizeof(int)) : 0;
    long nbytes_h_ct2           = (strategy == Reducer_t::strategy_bgj4 || strategy == Reducer_t::strategy_bgj3l) ? 
                                  ceil256(batch2 * Pool_hd_t::vec_nbytes) : 0;

    long nbytes_pinned = nbytes_task_vecs + nbytes_buc_vecs + num_threads * tpb * (nbytes_h_num_flt_out + 
                         nbytes_h_vec_out + nbytes_h_score_out + nbytes_h_norm_out + nbytes_h_u_out + 
                         nbytes_h_repeat_buf + nbytes_h_bko + nbytes_h_ct2) + num_threads * nbytes_h_ct1 + 
                         ((strategy == Reducer_t::strategy_bgj4 || strategy == Reducer_t::strategy_bgj3l) ? tpb : 1) * nbytes_h_norm * num_threads;
    nbytes_pinned = ((nbytes_pinned + 4095L) / 4096L) * 4096L;
    char *pinned_buf = NULL;
    if (posix_memalign((void **)&pinned_buf, 4096, nbytes_pinned)) {
        lg_err("posix_memalign failed");
    }
    CHECK_CUDA_ERR(cudaHostRegister(pinned_buf, nbytes_pinned, cudaHostAllocPortable));
    pinned_ram.fetch_add(nbytes_pinned, std::memory_order_relaxed);
    
    /// input buffers
    if (strategy == Reducer_t::strategy_bgj4 || strategy == Reducer_t::strategy_bgj3l) {
        task_vecs = (int *) pinned_buf;
        buc_vecs  = (int *) (pinned_buf + nbytes_task_vecs);
        memset(task_vecs, 0, num_threads * tpb * sizeof(int));
        memset(buc_vecs, 0, num_threads * tpb * sizeof(int));
        d_upk           = (int8_t **) malloc(num_threads * tpb * sizeof(int8_t *));
        d_vec16         = (int8_t **) malloc(num_threads * tpb * sizeof(int8_t *));
        h_norm          = (int32_t **) malloc(num_threads * tpb * sizeof(int32_t *));
        d_norm          = (int32_t **) malloc(num_threads * tpb * sizeof(int32_t *));
        d_n             = (int **) malloc(num_threads * tpb * sizeof(int *));
        repeat_buf      = (int **) malloc(num_threads * tpb * sizeof(int *));
        h_repeat_buf    = (int **) malloc(num_threads * tpb * sizeof(int *));
        repeat_buf_size = (int **) malloc(num_threads * tpb * sizeof(int *));
    } else {
        task_vecs = (int *) pinned_buf;
        buc_vecs  = (int *) (pinned_buf + nbytes_task_vecs);
        memset(task_vecs, 0, num_threads * sizeof(int));
        memset(buc_vecs, 0, num_threads * sizeof(int));
        d_upk           = (int8_t **) malloc(num_threads * sizeof(int8_t *));
        d_vec16         = (int8_t **) malloc(num_threads * sizeof(int8_t *));
        h_norm          = (int32_t **) malloc(num_threads * sizeof(int32_t *));
        d_norm          = (int32_t **) malloc(num_threads * sizeof(int32_t *));
        d_n             = (int **) malloc(num_threads * sizeof(int *));
    }
    /// reducing & output buffers, each thread holds one
    d_num_red_out   = (int **) malloc(num_threads * tpb * sizeof(int *));
    d_red_out       = (int **) malloc(num_threads * tpb * sizeof(int *));
    d_num_flt_out   = (int **) malloc(num_threads * tpb * sizeof(int *));
    h_num_flt_out   = (int **) malloc(num_threads * tpb * sizeof(int *));
    d_vec_out       = (int8_t **) malloc(num_threads * tpb * sizeof(int8_t *));
    h_vec_out       = (int8_t **) malloc(num_threads * tpb * sizeof(int8_t *));
    data            = (int8_t **) malloc(num_threads * tpb * sizeof(int8_t *));
    d_score_out     = (uint16_t **) malloc(num_threads * tpb * sizeof(uint16_t *));
    h_score_out     = (uint16_t **) malloc(num_threads * tpb * sizeof(uint16_t *));
    d_norm_out      = (int32_t **) malloc(num_threads * tpb * sizeof(int32_t *));
    h_norm_out      = (int32_t **) malloc(num_threads * tpb * sizeof(int32_t *));
    d_u_out         = (uint64_t **) malloc(num_threads * tpb * sizeof(uint64_t *));
    h_u_out         = (uint64_t **) malloc(num_threads * tpb * sizeof(uint64_t *));
    
    /// random and graph
    state               = (curandState **) malloc(num_threads * tpb * sizeof(curandState*));
    statte              = (curandState **) malloc(num_threads * sizeof(curandState*));
    #if USE_GRAPH
    graphs              = (cudaGraph_t *) calloc(num_threads * tpb, sizeof(cudaGraph_t));
    graphExecs          = (cudaGraphExec_t *) calloc(num_threads * tpb, sizeof(cudaGraphExec_t));
    redKernelNodes      = (cudaGraphNode_t **) calloc(num_threads * tpb, sizeof(cudaGraphNode_t *));
    redKernelParams     = (cudaKernelNodeParams **) calloc(num_threads * tpb, sizeof(cudaKernelNodeParams *));
    redKernelArgsList   = (arg_t **) calloc(num_threads * tpb, sizeof(arg_t *));
    #endif

    if (strategy == Reducer_t::strategy_bgj2) {
        d_ct1       = (int8_t **) malloc(num_threads * sizeof(int8_t *));
        d_bk1       = (int **) malloc(num_threads * sizeof(int *));
    }
    if (strategy == Reducer_t::strategy_bgj3) {
        d_ct1       = (int8_t **) malloc(num_threads * sizeof(int8_t *));
        d_bk1       = (int **) malloc(num_threads * sizeof(int *));
        d_ct2       = (int8_t **) malloc(num_threads * tpb * sizeof(int8_t *));
        d_bk2       = (int **) malloc(num_threads * tpb * sizeof(int *));
    }
    if (strategy == Reducer_t::strategy_bgj3l) {
        d_ct1       = (int8_t **) malloc(num_threads * tpb * sizeof(int8_t *));
        d_ctt1      = (int8_t **) malloc(num_threads * sizeof(int8_t *));
        h_ct1       = (int8_t **) malloc(num_threads * sizeof(int8_t *));
        d_bk1       = (int **) malloc(num_threads * tpb * sizeof(int *));
        h_bk1       = (int **) malloc(num_threads * sizeof(int *));
        h_bko       = (int **) malloc(num_threads * tpb * sizeof(int *));
        d_ct2       = (int8_t **) malloc(num_threads * tpb * sizeof(int8_t *));
        h_ct2       = (int8_t **) malloc(num_threads * tpb * sizeof(int8_t *));
        d_bk2       = (int **) malloc(num_threads * tpb * sizeof(int *));
    }
    if (strategy == Reducer_t::strategy_bgj4) {
        d_ct1       = (int8_t **) malloc(num_threads * tpb * sizeof(int8_t *));
        d_ctt1      = (int8_t **) malloc(num_threads * sizeof(int8_t *));
        h_ct1       = (int8_t **) malloc(num_threads * sizeof(int8_t *));
        d_bk1       = (int **) malloc(num_threads * tpb * sizeof(int *));
        h_bk1       = (int **) malloc(num_threads * sizeof(int *));
        h_bko       = (int **) malloc(num_threads * tpb * sizeof(int *));
        d_ct2       = (int8_t **) malloc(num_threads * tpb * sizeof(int8_t *));
        h_ct2       = (int8_t **) malloc(num_threads * tpb * sizeof(int8_t *));
        d_bk2       = (int **) malloc(num_threads * tpb * sizeof(int *));
        d_ct3       = (int8_t **) malloc(num_threads * tpb * sizeof(int8_t *));
        d_bk3       = (int **) malloc(num_threads * tpb * sizeof(int *));
    }

    /// host init
    pinned_buf += nbytes_task_vecs + nbytes_buc_vecs;
    for (int i = 0; i < num_threads * tpb; i++) {
        h_num_flt_out[i] = (int *) pinned_buf;
        pinned_buf += nbytes_h_num_flt_out;
        h_vec_out[i] = (int8_t *) pinned_buf;
        pinned_buf += nbytes_h_vec_out;
        h_score_out[i] = (uint16_t *) pinned_buf;
        pinned_buf += nbytes_h_score_out;
        h_norm_out[i] = (int32_t *) pinned_buf;
        pinned_buf += nbytes_h_norm_out;
        h_u_out[i] = (uint64_t *) pinned_buf;
        pinned_buf += nbytes_h_u_out;
        if (strategy == Reducer_t::strategy_bgj4 || strategy == Reducer_t::strategy_bgj3l || i < num_threads) {
            h_norm[i] = (int32_t *) pinned_buf;
            pinned_buf += nbytes_h_norm;
        }
    }

    if (strategy == Reducer_t::strategy_bgj4 || strategy == Reducer_t::strategy_bgj3l) {
        for (int i = 0; i < tpb * num_threads; i++) {
            h_repeat_buf[i] = (int *) pinned_buf;
            pinned_buf += nbytes_h_repeat_buf;
        }
        for (int i = 0; i < num_threads; i++) {
            h_ct1[i] = (int8_t *) pinned_buf;
            pinned_buf += nbytes_h_ct1;
        }
        for (int i = 0; i < num_threads; i++) h_bk1[i] = (int *) malloc(batch1 * (2L * bk1_max_size + 1L) * sizeof(int));
        pageable_ram.fetch_add(batch1 * (2L * bk1_max_size + 1L) * sizeof(int) * num_threads, std::memory_order_relaxed);
        for (int i = 0; i < tpb * num_threads; i++) {
            h_bko[i] = (int *) pinned_buf;
            pinned_buf += nbytes_h_bko;
        }
        for (int i = 0; i < tpb * num_threads; i++) {
            h_ct2[i] = (int8_t *) pinned_buf;
            pinned_buf += nbytes_h_ct2;
        }
    }
}

red_buffer_holder_t::~red_buffer_holder_t() {
    for (int i = 0; i < num_devices; i++) {
        CHECK_CUDA_ERR(cudaFree(local_data[i]));
    }

    /// thread & device info free
    for (int i = 0; i < num_devices; i++) {
        if (used_gram[i] != 0) lg_err("%ld bytes of GPU memory leak detected", used_gram[i]);
    }
    pthread_spin_destroy(&gram_lock);
    free(streams);
    free(sstreams);
    free(used_gram);

    if (strategy == Reducer_t::strategy_bgj4 || strategy == Reducer_t::strategy_bgj3l) {
        free(repeat_buf);
        free(h_repeat_buf);
        free(repeat_buf_size);
        for (int i = 0; i < num_threads; i++) free(h_bk1[i]);
    }

    /// runtime data free
    CHECK_CUDA_ERR(cudaHostUnregister(task_vecs));
    free(task_vecs);
    free(d_upk);
    free(d_vec16);
    free(h_norm);
    free(d_norm);
    free(d_num_red_out);
    free(d_red_out);
    free(d_n);
    free(local_data);
    free(d_num_flt_out);
    free(h_num_flt_out);
    free(d_vec_out);
    free(h_vec_out);
    free(data);
    free(d_score_out);
    free(h_score_out);
    free(d_norm_out);
    free(h_norm_out);
    free(d_u_out);
    free(h_u_out);
    free(state);
    free(statte);
    #if USE_GRAPH
    free(graphs);
    free(graphExecs);
    free(redKernelNodes);
    free(redKernelParams);
    free(redKernelArgsList);
    #endif
    if (h_bk1) { free(h_bk1); h_bk1 = NULL; }
    if (h_bko) { free(h_bko); h_bko = NULL; }
    if (d_ct1) { free(d_ct1); d_ct1 = NULL; }
    if (d_ctt1){free(d_ctt1); d_ctt1= NULL; }
    if (h_ct1) { free(h_ct1); h_ct1 = NULL; }
    if (d_bk1) { free(d_bk1); d_bk1 = NULL; }
    if (d_ct2) { free(d_ct2); d_ct2 = NULL; }
    if (h_ct2) { free(h_ct2); h_ct2 = NULL; }
    if (d_bk2) { free(d_bk2); d_bk2 = NULL; }
    if (d_ct3) { free(d_ct3); d_ct3 = NULL; }
    if (d_bk3) { free(d_bk3); d_bk3 = NULL; }
}

int red_buffer_holder_t::device_init(int tid, int sid) {
    int device_ptr;
    if (strategy == Reducer_t::strategy_bgj1 || strategy == Reducer_t::strategy_bgj2 || strategy == Reducer_t::strategy_bgj3) {
        device_ptr = hw::gpu_ptr(tid, num_threads);
    }
    if (strategy == Reducer_t::strategy_bgj3l || strategy == Reducer_t::strategy_bgj4) {
        device_ptr = hw::gpu_ptrl(tid, sid, num_threads, threads_per_buc);
    }
    CHECK_CUDA_ERR(cudaSetDevice(hw::gpu_id_list[device_ptr]));

    #if ENABLE_PROFILING
    if (strategy == Reducer_t::strategy_bgj1 || strategy == Reducer_t::strategy_bgj2 || 
       (strategy == Reducer_t::strategy_bgj3 && sid == -1)) {
        CHECK_CUDA_ERR(cudaEventCreate(&logger->h2d_sstart[tid]));
        CHECK_CUDA_ERR(cudaEventCreate(&logger->h2d_sstop[tid]));
        CHECK_CUDA_ERR(cudaEventCreate(&logger->upk_stop[tid]));
    }
    if ((strategy == Reducer_t::strategy_bgj3l || strategy == Reducer_t::strategy_bgj4) && sid >= 0) {
        int id = tid * threads_per_buc + sid;
        for (int i = 0; i < traits::taskChunks; i++) {
            CHECK_CUDA_ERR(cudaEventCreate(&logger->h2d_start[id][i]));
            CHECK_CUDA_ERR(cudaEventCreate(&logger->h2d_stop[id][i]));
        }
        CHECK_CUDA_ERR(cudaEventCreate(&logger->h2d_norm_start[id]));
        CHECK_CUDA_ERR(cudaEventCreate(&logger->bk1_start[id]));
        CHECK_CUDA_ERR(cudaEventCreate(&logger->bk2_start[id]));
    }
    if (strategy == Reducer_t::strategy_bgj3 && sid >= 0) {
        int id = tid * threads_per_buc + sid;
        CHECK_CUDA_ERR(cudaEventCreate(&logger->bk2_start[id]));
        CHECK_CUDA_ERR(cudaEventCreate(&logger->bk2_stop[id]));
    }
    if (strategy == Reducer_t::strategy_bgj1 || strategy == Reducer_t::strategy_bgj2) {
        CHECK_CUDA_ERR(cudaEventCreate(&logger->d2h_start[tid]));
        CHECK_CUDA_ERR(cudaEventCreate(&logger->d2h_stop[tid]));
        CHECK_CUDA_ERR(cudaEventCreate(&logger->fff_start[tid]));
        CHECK_CUDA_ERR(cudaEventCreate(&logger->fff_stop[tid]));
        CHECK_CUDA_ERR(cudaEventCreate(&logger->red_start[tid]));
        CHECK_CUDA_ERR(cudaEventCreate(&logger->red_stop[tid]));
    } else if (sid >= 0) {
        int id = tid * threads_per_buc + sid;
        CHECK_CUDA_ERR(cudaEventCreate(&logger->d2h_start[id]));
        CHECK_CUDA_ERR(cudaEventCreate(&logger->d2h_stop[id]));
        CHECK_CUDA_ERR(cudaEventCreate(&logger->fff_start[id]));
        CHECK_CUDA_ERR(cudaEventCreate(&logger->fff_stop[id]));
        CHECK_CUDA_ERR(cudaEventCreate(&logger->red_start[id]));
        CHECK_CUDA_ERR(cudaEventCreate(&logger->red_stop[id]));
    }
    #endif
    #if USE_GRAPH
    cudaFuncSetAttribute(red_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 
        strategy == Reducer_t::strategy_bgj1 ? traits::l0_shmem : traits::l1_shmem);
    #else
    if (strategy == Reducer_t::strategy_bgj1) {
        cudaFuncSetAttribute(rd1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, traits::l0_shmem);
    }
    if (strategy == Reducer_t::strategy_bgj2 || strategy == Reducer_t::strategy_bgj3 ||
        strategy == Reducer_t::strategy_bgj3l || strategy == Reducer_t::strategy_bgj4) {
        cudaFuncSetAttribute(red_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, traits::l1_shmem);
    }
    #endif
    cudaFuncSetAttribute(fpv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, fpv_shmem);
    cudaFuncSetAttribute(flt_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, check_traits::dynamic_shmem);
    cudaFuncSetAttribute(fcs_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, fcs_shmem);
    cudaFuncSetAttribute(upk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, utils_t::packshmem);
    cudaFuncSetAttribute(pk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, utils_t::packshmem);
    if (strategy == Reducer_t::strategy_bgj2) {
        cudaFuncSetAttribute(bgj2_ctr_gen, cudaFuncAttributeMaxDynamicSharedMemorySize, buccg_shmem);
        cudaFuncSetAttribute(bk1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, buc_traits_t::l1_shmem);
    }
    if (strategy == Reducer_t::strategy_bgj3) {
        cudaFuncSetAttribute(bgj2_ctr_gen, cudaFuncAttributeMaxDynamicSharedMemorySize, buccg_shmem);
        cudaFuncSetAttribute(bgj34_ctr_gen, cudaFuncAttributeMaxDynamicSharedMemorySize, buccg_shmem);
        cudaFuncSetAttribute(bk1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, buc_traits_t::l1_shmem);
        cudaFuncSetAttribute(bk2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, buc_traits_t::l2_shmem);
    }
    if (strategy == Reducer_t::strategy_bgj3l) {
        cudaFuncSetAttribute(bgj2_ctr_gen, cudaFuncAttributeMaxDynamicSharedMemorySize, buccg_shmem);
        cudaFuncSetAttribute(bgj34_ctr_gen, cudaFuncAttributeMaxDynamicSharedMemorySize, buccg_shmem);
        cudaFuncSetAttribute(bk1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, buc_traits_t::l1_shmem);
        cudaFuncSetAttribute(bk2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, buc_traits_t::l1_shmem);
        cudaFuncSetAttribute(rep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, buc_traits_t::l2_shmem);
    }
    if (strategy == Reducer_t::strategy_bgj4) {
        cudaFuncSetAttribute(bgj2_ctr_gen, cudaFuncAttributeMaxDynamicSharedMemorySize, buccg_shmem);
        cudaFuncSetAttribute(bgj34_ctr_gen, cudaFuncAttributeMaxDynamicSharedMemorySize, buccg_shmem);
        cudaFuncSetAttribute(bk1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, buc_traits_t::l1_shmem);
        cudaFuncSetAttribute(bk2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, buc_traits_t::l1_shmem);
        cudaFuncSetAttribute(bk3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, buc_traits_t::l2_shmem);
        cudaFuncSetAttribute(rep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, buc_traits_t::l2_shmem);
    }
    
    if (strategy == Reducer_t::strategy_bgj3l || strategy == Reducer_t::strategy_bgj4) {
        if (sid == -1) {
            CHECK_CUDA_ERR(cudaStreamCreate(&streams[tid]));
            CHECK_CUDA_ERR(cudaMalloc(&d_ctt1[tid], batch1 * Pool_hd_t::vec_nbytes));
            CHECK_CUDA_ERR(cudaMalloc(&statte[tid], buccg_threads * buccg_blocks * sizeof(curandState)));
            init_curand<<<buccg_blocks, buccg_threads, 0, streams[tid]>>>(statte[tid], tid);
            CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));
            return 0;
        }

        long _used_gram = 0;

        int id = tid * threads_per_buc + sid;
        CHECK_CUDA_ERR(cudaStreamCreate(&sstreams[id]));

        CHECK_CUDA_ERR(cudaMalloc(&repeat_buf[id], bk1_max_size * 8));
        CHECK_CUDA_ERR(cudaMalloc(&repeat_buf_size[id], sizeof(int)));
        _used_gram += bk1_max_size * 8 + sizeof(int);

        long d_norm_vecs = bk1_max_size > traits::taskVecs ? bk1_max_size : traits::taskVecs;
        CHECK_CUDA_ERR(cudaMalloc(&d_num_red_out[id], sizeof(int)));
        CHECK_CUDA_ERR(cudaMalloc(&d_red_out[id], out_max_size * 2L * sizeof(int)));
        CHECK_CUDA_ERR(cudaMalloc(&d_num_flt_out[id], sizeof(int)));
        CHECK_CUDA_ERR(cudaMalloc(&d_vec_out[id], flt_out_max_size * CSD16 * sizeof(int8_t)));
        CHECK_CUDA_ERR(cudaMalloc(&d_score_out[id], flt_out_max_size * sizeof(uint16_t)));
        CHECK_CUDA_ERR(cudaMalloc(&d_norm_out[id], flt_out_max_size * sizeof(int32_t)));
        CHECK_CUDA_ERR(cudaMalloc(&d_u_out[id], flt_out_max_size * sizeof(uint64_t)));
        CHECK_CUDA_ERR(cudaMalloc(&data[id], (2 + 4 + 8 + Pool_hd_t::vec_nbytes) * (long) filter_taskVecs));        
        CHECK_CUDA_ERR(cudaMalloc(&d_n[id], sizeof(int)));
        CHECK_CUDA_ERR(cudaMalloc(&d_upk[id], traits::taskVecs * CSD16 * sizeof(int8_t)));
        CHECK_CUDA_ERR(cudaMalloc(&d_vec16[id], bk1_max_size * CSD16 * sizeof(int8_t)));
        CHECK_CUDA_ERR(cudaMalloc(&d_norm[id], d_norm_vecs * sizeof(int32_t)));
        CHECK_CUDA_ERR(cudaMalloc(&state[id], buccg_threads * buccg_blocks * sizeof(curandState)));
        init_curand<<<buccg_blocks, buccg_threads, 0, sstreams[id]>>>(state[id], id);
        _used_gram += sizeof(int) + out_max_size * 2 * sizeof(int) + sizeof(int) + 
                      flt_out_max_size * CSD16 * sizeof(int8_t) + flt_out_max_size * sizeof(uint16_t) + 
                      flt_out_max_size * sizeof(int32_t) + flt_out_max_size * sizeof(uint64_t) + 
                      (2 + 4 + 8 + Pool_hd_t::vec_nbytes) * filter_taskVecs + 
                      traits::taskVecs * CSD16 + bk1_max_size * CSD16 + d_norm_vecs * sizeof(int32_t) + sizeof(int);

        CHECK_CUDA_ERR(cudaMalloc(&d_ct1[id], batch1 * Pool_hd_t::vec_nbytes));
        CHECK_CUDA_ERR(cudaMalloc(&d_bk1[id], batch1 * (2 * l1_out_max_size + 1) * sizeof(int)));
        CHECK_CUDA_ERR(cudaMalloc(&d_ct2[id], batch2 * Pool_hd_t::vec_nbytes));
        CHECK_CUDA_ERR(cudaMalloc(&d_bk2[id], batch2 * (2 * bk2_max_size + 1) * sizeof(int)));
        _used_gram += batch1 * Pool_hd_t::vec_nbytes + batch1 * (2 * l1_out_max_size + 1) * sizeof(int) + 
                      (batch2 * Pool_hd_t::vec_nbytes + batch2 * (2 * bk2_max_size + 1) * sizeof(int));
    
        if (strategy == Reducer_t::strategy_bgj4) {
            CHECK_CUDA_ERR(cudaMalloc(&d_ct3[id], batch3 * Pool_hd_t::vec_nbytes));
            CHECK_CUDA_ERR(cudaMalloc(&d_bk3[id], batch3 * (2 * bk3_max_size + 1) * sizeof(int)));
            _used_gram += (batch3 * Pool_hd_t::vec_nbytes + batch3 * (2 * bk3_max_size + 1) * sizeof(int));
        }

        #if USE_GRAPH
        this->graph_init(tid, sid);
        #endif

        pthread_spin_lock(&gram_lock);
        used_gram[device_ptr] += _used_gram;
        pthread_spin_unlock(&gram_lock);

        if (used_gram[device_ptr] > reducer->_gram_slimit) {
            lg_warn("device %d using %ld byte GRAM for red, exceeds limit %ld", 
                    hw::gpu_id_list[device_ptr], used_gram[device_ptr], reducer->_gram_slimit);
        }

        return 0;
    }

    if (sid >= 0) {
        CHECK_CUDA_ERR(cudaStreamCreate(&sstreams[tid * threads_per_buc + sid]));
        #if USE_GRAPH
        this->graph_init(tid, sid);
        #endif
        return 0;
    }
    CHECK_CUDA_ERR(cudaStreamCreate(&streams[tid]));

    int tpb = threads_per_buc ? threads_per_buc : 1;

    long _used_gram = 0;

    for (int t = 0; t < tpb; t++) {
        CHECK_CUDA_ERR(cudaMalloc(&d_num_red_out[tid * tpb + t], sizeof(int)));
        CHECK_CUDA_ERR(cudaMalloc(&d_red_out[tid * tpb + t], out_max_size * 2L * sizeof(int)));
        CHECK_CUDA_ERR(cudaMalloc(&d_num_flt_out[tid * tpb + t], sizeof(int)));
        CHECK_CUDA_ERR(cudaMalloc(&d_vec_out[tid * tpb + t], flt_out_max_size * CSD16 * sizeof(int8_t)));
        CHECK_CUDA_ERR(cudaMalloc(&d_score_out[tid * tpb + t], flt_out_max_size * sizeof(uint16_t)));
        CHECK_CUDA_ERR(cudaMalloc(&d_norm_out[tid * tpb + t], flt_out_max_size * sizeof(int32_t)));
        CHECK_CUDA_ERR(cudaMalloc(&d_u_out[tid * tpb + t], flt_out_max_size * sizeof(uint64_t)));
        CHECK_CUDA_ERR(cudaMalloc(&data[tid * tpb + t], (2 + 4 + 8 + Pool_hd_t::vec_nbytes) * (long) filter_taskVecs));
        _used_gram += sizeof(int) + out_max_size * 2 * sizeof(int) + sizeof(int) + 
                      flt_out_max_size * CSD16 * sizeof(int8_t) + flt_out_max_size * sizeof(uint16_t) + 
                      flt_out_max_size * sizeof(int32_t) + flt_out_max_size * sizeof(uint64_t) + 
                      (2 + 4 + 8 + Pool_hd_t::vec_nbytes) * filter_taskVecs;

        if (t == 0) {
            long d_norm_vecs = buc_max_size > traits::taskVecs ? buc_max_size : traits::taskVecs;
            CHECK_CUDA_ERR(cudaMalloc(&d_n[tid], sizeof(int)));
            CHECK_CUDA_ERR(cudaMalloc(&d_upk[tid], traits::taskVecs * CSD16 * sizeof(int8_t)));
            CHECK_CUDA_ERR(cudaMalloc(&d_vec16[tid], buc_max_size * CSD16 * sizeof(int8_t)));
            CHECK_CUDA_ERR(cudaMalloc(&d_norm[tid], d_norm_vecs * sizeof(int32_t)));
            _used_gram += traits::taskVecs * CSD16 + buc_max_size * CSD16 + d_norm_vecs * sizeof(int32_t) + sizeof(int);
        }
        CHECK_CUDA_ERR(cudaMalloc(&state[tid * tpb + t], buccg_threads * buccg_blocks * sizeof(curandState)));
        init_curand<<<buccg_blocks, buccg_threads, 0, streams[tid]>>>(state[tid * tpb + t], tid * tpb + t);
    }
    CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));

    if (strategy == Reducer_t::strategy_bgj2) {
        CHECK_CUDA_ERR(cudaMalloc(&d_ct1[tid], batch1 * Pool_hd_t::vec_nbytes));
        CHECK_CUDA_ERR(cudaMalloc(&d_bk1[tid], batch1 * (2 * bk1_max_size + 1) * sizeof(int)));
        _used_gram += batch1 * Pool_hd_t::vec_nbytes + batch1 * (2 * bk1_max_size + 1) * sizeof(int);
    }

    if (strategy == Reducer_t::strategy_bgj3) {
        CHECK_CUDA_ERR(cudaMalloc(&d_ct1[tid], batch1 * Pool_hd_t::vec_nbytes));
        CHECK_CUDA_ERR(cudaMalloc(&d_bk1[tid], batch1 * (2 * bk1_max_size + 1) * sizeof(int)));
        for (int t = 0; t < tpb; t++) {
            CHECK_CUDA_ERR(cudaMalloc(&d_ct2[tid * tpb + t], batch2 * Pool_hd_t::vec_nbytes));
            CHECK_CUDA_ERR(cudaMalloc(&d_bk2[tid * tpb + t], batch2 * (2 * bk2_max_size + 1) * sizeof(int)));
        }
        _used_gram += batch1 * Pool_hd_t::vec_nbytes + batch1 * (2 * bk1_max_size + 1) * sizeof(int) + 
                      tpb * (batch2 * Pool_hd_t::vec_nbytes + batch2 * (2 * bk2_max_size + 1) * sizeof(int));
    }
    
    pthread_spin_lock(&gram_lock);
    used_gram[device_ptr] += _used_gram;
    pthread_spin_unlock(&gram_lock);

    if (used_gram[device_ptr] > reducer->_gram_slimit) {
        lg_warn("device %d using %ld byte GRAM for red, exceeds limit %ld", 
                hw::gpu_id_list[device_ptr], used_gram[device_ptr], reducer->_gram_slimit);
    }

    #if USE_GRAPH
    if (strategy == Reducer_t::strategy_bgj2) this->graph_init(tid, sid);
    #endif

    return 0;
}

int red_buffer_holder_t::device_done(int tid, int sid) {
    int device_ptr;
    if (strategy == Reducer_t::strategy_bgj1 || strategy == Reducer_t::strategy_bgj2 || strategy == Reducer_t::strategy_bgj3) {
        device_ptr = hw::gpu_ptr(tid, num_threads);
    }
    if (strategy == Reducer_t::strategy_bgj3l || strategy == Reducer_t::strategy_bgj4) {
        device_ptr = hw::gpu_ptrl(tid, sid, num_threads, threads_per_buc);
    }
    #if ENABLE_PROFILING
    if (strategy == Reducer_t::strategy_bgj1 || strategy == Reducer_t::strategy_bgj2 || 
       (strategy == Reducer_t::strategy_bgj3 && sid == -1)) {
        CHECK_CUDA_ERR(cudaEventDestroy(logger->h2d_sstart[tid]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->h2d_sstop[tid]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->upk_stop[tid]));
    }
    if ((strategy == Reducer_t::strategy_bgj3l || strategy == Reducer_t::strategy_bgj4) && sid >= 0) {
        int id = tid * threads_per_buc + sid;
        for (int i = 0; i < traits::taskChunks; i++) {
            CHECK_CUDA_ERR(cudaEventDestroy(logger->h2d_start[id][i]));
            CHECK_CUDA_ERR(cudaEventDestroy(logger->h2d_stop[id][i]));
        }
        CHECK_CUDA_ERR(cudaEventDestroy(logger->h2d_norm_start[id]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->bk1_start[id]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->bk2_start[id]));
    }
    if (strategy == Reducer_t::strategy_bgj3 && sid >= 0) {
        int id = tid * threads_per_buc + sid;
        CHECK_CUDA_ERR(cudaEventDestroy(logger->bk2_start[id]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->bk2_stop[id]));
    }
    if (strategy == Reducer_t::strategy_bgj1 || strategy == Reducer_t::strategy_bgj2) {
        CHECK_CUDA_ERR(cudaEventDestroy(logger->d2h_start[tid]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->d2h_stop[tid]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->fff_start[tid]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->fff_stop[tid]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->red_start[tid]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->red_stop[tid]));
    } else if (sid >= 0) {
        int id = tid * threads_per_buc + sid;
        CHECK_CUDA_ERR(cudaEventDestroy(logger->d2h_start[id]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->d2h_stop[id]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->fff_start[id]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->fff_stop[id]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->red_start[id]));
        CHECK_CUDA_ERR(cudaEventDestroy(logger->red_stop[id]));
    }
    #endif

    if (strategy == Reducer_t::strategy_bgj3l || strategy == Reducer_t::strategy_bgj4) {
        if (sid == -1) {
            CHECK_CUDA_ERR(cudaFree(d_ctt1[tid]));
            CHECK_CUDA_ERR(cudaFree(statte[tid]));
            CHECK_CUDA_ERR(cudaStreamDestroy(streams[tid]));
            return 0;
        }

        long _used_gram = 0;

        int id = tid * threads_per_buc + sid;

        CHECK_CUDA_ERR(cudaFree(repeat_buf[id]));
        CHECK_CUDA_ERR(cudaFree(repeat_buf_size[id]));
        _used_gram += bk1_max_size * 8 + sizeof(int);

        long d_norm_vecs = bk1_max_size > traits::taskVecs ? bk1_max_size : traits::taskVecs;
        CHECK_CUDA_ERR(cudaFree(d_num_red_out[id]));
        CHECK_CUDA_ERR(cudaFree(d_red_out[id]));
        CHECK_CUDA_ERR(cudaFree(d_num_flt_out[id]));
        CHECK_CUDA_ERR(cudaFree(d_vec_out[id]));
        CHECK_CUDA_ERR(cudaFree(d_score_out[id]));
        CHECK_CUDA_ERR(cudaFree(d_norm_out[id]));
        CHECK_CUDA_ERR(cudaFree(d_u_out[id]));
        CHECK_CUDA_ERR(cudaFree(data[id]));
        CHECK_CUDA_ERR(cudaFree(d_n[id]));
        CHECK_CUDA_ERR(cudaFree(d_upk[id]));
        CHECK_CUDA_ERR(cudaFree(d_vec16[id]));
        CHECK_CUDA_ERR(cudaFree(d_norm[id]));
        CHECK_CUDA_ERR(cudaFree(state[id]));
        _used_gram += sizeof(int) + out_max_size * 2 * sizeof(int) + sizeof(int) + 
                      flt_out_max_size * CSD16 * sizeof(int8_t) + flt_out_max_size * sizeof(uint16_t) + 
                      flt_out_max_size * sizeof(int32_t) + flt_out_max_size * sizeof(uint64_t) + 
                      (2 + 4 + 8 + Pool_hd_t::vec_nbytes) * filter_taskVecs + 
                      traits::taskVecs * CSD16 + bk1_max_size * CSD16 + d_norm_vecs * sizeof(int32_t) + sizeof(int);
        
        CHECK_CUDA_ERR(cudaFree(d_ct1[id]));
        CHECK_CUDA_ERR(cudaFree(d_bk1[id]));
        CHECK_CUDA_ERR(cudaFree(d_ct2[id]));
        CHECK_CUDA_ERR(cudaFree(d_bk2[id]));
        _used_gram += batch1 * Pool_hd_t::vec_nbytes + batch1 * (2 * l1_out_max_size + 1) * sizeof(int) + 
                      (batch2 * Pool_hd_t::vec_nbytes + batch2 * (2 * bk2_max_size + 1) * sizeof(int));
    
        if (strategy == Reducer_t::strategy_bgj4) {
            CHECK_CUDA_ERR(cudaFree(d_ct3[id]));
            CHECK_CUDA_ERR(cudaFree(d_bk3[id]));
            _used_gram += (batch3 * Pool_hd_t::vec_nbytes + batch3 * (2 * bk3_max_size + 1) * sizeof(int));
        }

        #if USE_GRAPH
        this->graph_done(tid, sid);
        #endif
        CHECK_CUDA_ERR(cudaStreamDestroy(sstreams[id]));


        pthread_spin_lock(&gram_lock);
        used_gram[device_ptr] -= _used_gram;
        pthread_spin_unlock(&gram_lock);

        return 0;
    }

    if (sid >= 0) {
        #if USE_GRAPH
        this->graph_done(tid, sid);
        #endif
        CHECK_CUDA_ERR(cudaStreamDestroy(sstreams[tid * threads_per_buc + sid]));
        return 0;
    }
    CHECK_CUDA_ERR(cudaStreamDestroy(streams[tid]));

    long tpb = threads_per_buc ? threads_per_buc : 1;

    long _used_gram = 0;

    for (int t = 0; t < tpb; t++) {
        CHECK_CUDA_ERR(cudaFree(d_num_red_out[tid * tpb + t]));
        CHECK_CUDA_ERR(cudaFree(d_red_out[tid * tpb + t]));
        CHECK_CUDA_ERR(cudaFree(d_num_flt_out[tid * tpb + t]));
        CHECK_CUDA_ERR(cudaFree(d_vec_out[tid * tpb + t]));
        CHECK_CUDA_ERR(cudaFree(d_score_out[tid * tpb + t]));
        CHECK_CUDA_ERR(cudaFree(d_norm_out[tid * tpb + t]));
        CHECK_CUDA_ERR(cudaFree(d_u_out[tid * tpb + t]));
        CHECK_CUDA_ERR(cudaFree(data[tid * tpb + t]));
        _used_gram += sizeof(int) + out_max_size * 2 * sizeof(int) + sizeof(int) +
                      flt_out_max_size * CSD16 * sizeof(int8_t) + flt_out_max_size * sizeof(uint16_t) + 
                      flt_out_max_size * sizeof(int32_t) + flt_out_max_size * sizeof(uint64_t) + 
                      (2 + 4 + 8 + Pool_hd_t::vec_nbytes) * filter_taskVecs;

        if (t == 0) {
            int d_norm_vecs = buc_max_size > traits::taskVecs ? buc_max_size : traits::taskVecs;
            CHECK_CUDA_ERR(cudaFree(d_n[tid]));
            CHECK_CUDA_ERR(cudaFree(d_upk[tid]));
            CHECK_CUDA_ERR(cudaFree(d_vec16[tid]));
            CHECK_CUDA_ERR(cudaFree(d_norm[tid]));
            _used_gram += traits::taskVecs * CSD16 + buc_max_size * CSD16 + d_norm_vecs * sizeof(int32_t) + sizeof(int);
        }
        CHECK_CUDA_ERR(cudaFree(state[tid * tpb + t]));
    }

    if (strategy == Reducer_t::strategy_bgj2) {
        CHECK_CUDA_ERR(cudaFree(d_ct1[tid]));
        CHECK_CUDA_ERR(cudaFree(d_bk1[tid]));
        _used_gram += batch1 * Pool_hd_t::vec_nbytes + batch1 * (2 * bk1_max_size + 1) * sizeof(int);
    }

    if (strategy == Reducer_t::strategy_bgj3) {
        CHECK_CUDA_ERR(cudaFree(d_ct1[tid]));
        CHECK_CUDA_ERR(cudaFree(d_bk1[tid]));
        for (int t = 0; t < tpb; t++) {
            CHECK_CUDA_ERR(cudaFree(d_ct2[tid * tpb + t]));
            CHECK_CUDA_ERR(cudaFree(d_bk2[tid * tpb + t]));
        }
        _used_gram += batch1 * Pool_hd_t::vec_nbytes + batch1 * (2 * bk1_max_size + 1) * sizeof(int) + 
                      tpb * (batch2 * Pool_hd_t::vec_nbytes + batch2 * (2 * bk2_max_size + 1) * sizeof(int));
    }
    
    pthread_spin_lock(&gram_lock);
    used_gram[device_ptr] -= _used_gram;
    pthread_spin_unlock(&gram_lock);

    #if USE_GRAPH
    if (strategy == Reducer_t::strategy_bgj2) this->graph_done(tid, sid);
    #endif

    return 0;
}

#if USE_GRAPH
int red_buffer_holder_t::graph_init(int tid, int sid) {
    if (strategy != Reducer_t::strategy_bgj2 && strategy != Reducer_t::strategy_bgj3 && 
        strategy != Reducer_t::strategy_bgj3l && strategy != Reducer_t::strategy_bgj4) return -1;

    int id = strategy == Reducer_t::strategy_bgj2 ? tid : tid * threads_per_buc + sid;
    int vid = strategy == Reducer_t::strategy_bgj3 ? tid : id;
    int batch = strategy == Reducer_t::strategy_bgj2 ? batch1 : 
            (strategy == Reducer_t::strategy_bgj3 || strategy == Reducer_t::strategy_bgj3l) ? batch2 : batch3;
    int **d_bk = strategy == Reducer_t::strategy_bgj2 ? d_bk1 : 
            (strategy == Reducer_t::strategy_bgj3 || strategy == Reducer_t::strategy_bgj3l) ? d_bk2 : d_bk3;
    int *bk_max_size_ptr = strategy == Reducer_t::strategy_bgj2 ? &bk1_max_size_i : 
            (strategy == Reducer_t::strategy_bgj3 || strategy == Reducer_t::strategy_bgj3l) ? &bk2_max_size_i : &bk3_max_size_i;
    
    CHECK_CUDA_ERR(cudaGraphCreate(&graphs[id], 0));
    redKernelNodes[id] = (cudaGraphNode_t *) malloc(batch * sizeof(cudaGraphNode_t));
    redKernelParams[id] = (cudaKernelNodeParams *) malloc(batch * sizeof(cudaKernelNodeParams));
    redKernelArgsList[id] = (arg_t *) malloc(9 * batch * sizeof(arg_t));

    /// reducing tasks
    for (int i = 0; i < batch; i++) {
        redKernelArgsList[id][i * 9 + 7] = d_bk[id] + batch + i * 2 * bk_max_size_ptr[0];
        redKernelArgsList[id][i * 9 + 8] = d_bk[id] + i;

        /// rpp kernel
        cudaKernelNodeParams rppNodeParams = {};
        void *rppArgs[] = {
            (void*)&redKernelArgsList[id][i * 9 + 7],
            (void*)&redKernelArgsList[id][i * 9 + 8],
            (void*)bk_max_size_ptr
        };
        rppNodeParams.func = (void*)rpp_kernel;
        rppNodeParams.gridDim = dim3(1);
        rppNodeParams.blockDim = dim3(512);
        rppNodeParams.sharedMemBytes = 0;
        rppNodeParams.kernelParams = rppArgs;
        rppNodeParams.extra = nullptr;

        cudaGraphNode_t rppNode;
        CHECK_CUDA_ERR(cudaGraphAddKernelNode(&rppNode, graphs[id], nullptr, 0, &rppNodeParams));

        /// red kernel
        cudaKernelNodeParams *redNodeParams = &redKernelParams[id][i];
        void **redArgs = redKernelArgsList[id] + i * 9;
        redArgs[0] = (void*)&d_red_out[id];
        redArgs[1] = (void*)&d_num_red_out[id];
        redArgs[2] = (void*)&out_max_size_i;
        redArgs[3] = (void*)&d_vec16[vid];
        redArgs[4] = (void*)&redKernelArgsList[id][i * 9 + 7];
        redArgs[5] = (void*)&redKernelArgsList[id][i * 9 + 8];
        redArgs[6] = (void*)&reducer->goal_norm;

        redNodeParams->func = (void*)red_kernel;
        redNodeParams->gridDim = dim3(1);
        redNodeParams->blockDim = dim3(traits::blockThreads);
        redNodeParams->sharedMemBytes = traits::l1_shmem;
        redNodeParams->kernelParams = redArgs;
        redNodeParams->extra = nullptr;

        cudaGraphNode_t redNode;
        cudaGraphNode_t redDependencies[] = { rppNode };
        CHECK_CUDA_ERR(cudaGraphAddKernelNode(&redNode, graphs[id], redDependencies, 1, redNodeParams));
    
        redKernelNodes[id][i] = redNode;
    }

    CHECK_CUDA_ERR(cudaGraphInstantiate(&graphExecs[id], graphs[id], nullptr, nullptr, 0));
    
    return 0;
}

int red_buffer_holder_t::graph_done(int tid, int sid) {
    int id = strategy == Reducer_t::strategy_bgj2 ? tid : tid * threads_per_buc + sid;
    
    if (strategy == Reducer_t::strategy_bgj2 || strategy == Reducer_t::strategy_bgj3 || 
        strategy == Reducer_t::strategy_bgj3l || strategy == Reducer_t::strategy_bgj4) {
        CHECK_CUDA_ERR(cudaGraphExecDestroy(graphExecs[id]));
        CHECK_CUDA_ERR(cudaGraphDestroy(graphs[id]));
        graphExecs[id] = nullptr;
        graphs[id] = nullptr;
        free(redKernelNodes[id]);
        free(redKernelParams[id]);
        free(redKernelArgsList[id]);
    }

    return 0;
}
#endif

int red_buffer_holder_t::bgjs_h2d(int tid, chunk_t *chunk, int &used) {
    int buc_full = 0;
    int to_copy = chunk->size - used < traits::taskVecs - task_vecs[tid] ?
                  chunk->size - used : traits::taskVecs - task_vecs[tid];
    if (to_copy + task_vecs[tid] + buc_vecs[tid] >= buc_max_size) {
        to_copy = buc_max_size - buc_vecs[tid] - task_vecs[tid];
        if (to_copy < 0) to_copy = 0;
        buc_full = 1;
    }
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->h2d_sstart[tid], streams[tid]));
    #endif
    if (to_copy)
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_upk[tid] + task_vecs[tid] * CSD, chunk->vec + used * CSD, 
                                   to_copy * CSD, cudaMemcpyHostToDevice, streams[tid]));
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->h2d_sstop[tid], streams[tid]));
    #endif
    if (to_copy) memcpy(h_norm[tid] + task_vecs[tid], chunk->norm + used, to_copy * sizeof(int32_t));
    task_vecs[tid] += to_copy;
    used = buc_full ? -1 : used + to_copy;

    CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));
    #if ENABLE_PROFILING
    {
        float tt;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&tt, logger->h2d_sstart[tid], logger->h2d_sstop[tid]));
        logger->ev_h2d_us += 1000.f * tt;
    }
    #endif

    return to_copy;
}

int red_buffer_holder_t::bgjs_upk(int tid) {
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->h2d_sstart[tid], streams[tid]));
    #endif
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_norm[tid] + buc_vecs[tid], h_norm[tid], task_vecs[tid] * 4, 
                                   cudaMemcpyHostToDevice, streams[tid]));
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->h2d_sstop[tid], streams[tid]));
    #endif
    upk_kernel<<<utils_t::packBlocks, utils_t::packThreads, utils_t::packshmem, streams[tid]>>>(
        d_vec16[tid] + buc_vecs[tid] * CSD16, d_upk[tid], CSD, task_vecs[tid]
    );
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->upk_stop[tid], streams[tid]));
    #endif
    CHECK_LAST_ERR;
    CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));
    #if ENABLE_PROFILING
    {
        float h2d_tt, upk_tt;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&h2d_tt, logger->h2d_sstart[tid], logger->h2d_sstop[tid]));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&upk_tt, logger->h2d_sstop[tid], logger->upk_stop[tid]));
        logger->ev_h2d_us += 1000.f * h2d_tt;
        logger->ev_upk_us += 1000.f * upk_tt;
        logger->ev_h2d_nbytes += task_vecs[tid] * (CSD + 4);
    }
    #endif

    buc_vecs[tid] += task_vecs[tid];
    task_vecs[tid] = 0;

    return 0;
}

int red_buffer_holder_t::bgj1_run(int tid) {
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_n[tid], &buc_vecs[tid], sizeof(int), cudaMemcpyHostToDevice, streams[tid]));
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->red_start[tid], streams[tid]));
    #endif

    CHECK_CUDA_ERR(cudaMemsetAsync(d_num_red_out[tid], 0, sizeof(int), streams[tid]));
    rpp_kernel<<<1, 512, 0, streams[tid]>>>(d_norm[tid], d_n[tid], buc_max_size);
    #if USE_GRAPH
    red_kernel<<<traits::kernelBlocks, traits::blockThreads, traits::l0_shmem, streams[tid]>>>(
        d_red_out[tid], d_num_red_out[tid], out_max_size, d_vec16[tid], d_norm[tid], d_n[tid], reducer->goal_norm
    );
    #else
    rd1_kernel<<<traits::kernelBlocks, traits::blockThreads, traits::l0_shmem, streams[tid]>>>(
        d_red_out[tid], d_num_red_out[tid], out_max_size, d_vec16[tid], d_norm[tid], d_n[tid], reducer->goal_norm
    );
    #endif

    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->red_stop[tid], streams[tid]));
    logger->ev_red_vmmas += ceil(buc_vecs[tid] / 16.0) * ceil(buc_vecs[tid] / 16.0 + 1.0) / 2;
    #endif

    buc_vecs[tid] = 0;

    return 0;
}

int red_buffer_holder_t::bgjs_out(int tid, int *size, int8_t **h_vec, int32_t **h_norm, uint16_t **h_score, uint64_t **h_u) {
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_num_flt_out[tid], d_num_red_out[tid], sizeof(int), cudaMemcpyDeviceToHost, streams[tid]));
    CHECK_CUDA_ERR(cudaMemsetAsync(d_num_flt_out[tid], 0, sizeof(int), streams[tid]));
    CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));
    #if ENABLE_PROFILING
    float ret_tt;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&ret_tt, logger->red_start[tid], logger->red_stop[tid]));
    logger->ev_red_us += 1000.f * ret_tt;
    #endif
    if ((int)h_num_flt_out[tid][0] < 0) lg_err("thread %d, num_red_out overflow(%d), ignored", tid, h_num_flt_out[tid][0]);    
    int to_flt = h_num_flt_out[tid][0] < out_max_size ? h_num_flt_out[tid][0] : out_max_size;
    #if ENABLE_PROFILING
    logger->ev_flt_num += 1;
    logger->ev_red_msum += h_num_flt_out[tid][0];
    logger->ev_red_max = std::max((int)logger->ev_red_max.load(), h_num_flt_out[tid][0]);
    logger->ev_red_ssum += to_flt;
    #endif

    int device_ptr = hw::gpu_ptr(tid, num_threads);
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->fff_start[tid], streams[tid]));
    #endif
    for (int i = 0; i < to_flt; i += filter_taskVecs) {
        int batch_num = to_flt - i < filter_taskVecs ? to_flt - i : filter_taskVecs;
        fpv_kernel<<<fpv_blocks, fpv_threads, fpv_shmem, streams[tid]>>>(data[tid], d_vec16[tid], d_red_out[tid] + 2 * i, batch_num);
        flt_kernel<<<check_traits::kernelBlocks, check_traits::blockThreads, check_traits::dynamic_shmem, streams[tid]>>>(
            data[tid], batch_num, local_data[device_ptr]
        );
        fcs_kernel<<<fcs_blocks, fcs_threads, fcs_shmem, streams[tid]>>>(d_vec_out[tid], d_score_out[tid], 
            d_norm_out[tid], d_u_out[tid], d_num_flt_out[tid], flt_out_max_size, data[tid], batch_num, reducer->goal_score);
    }
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->fff_stop[tid], streams[tid]));
    #endif

    CHECK_CUDA_ERR(cudaMemcpyAsync(h_num_flt_out[tid], d_num_flt_out[tid], sizeof(int), cudaMemcpyDeviceToHost, streams[tid]));
    CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));
    #if ENABLE_PROFILING
    logger->ev_flt_max = std::max((int)logger->ev_flt_max.load(), h_num_flt_out[tid][0]);
    logger->ev_flt_ssum += std::min((long)h_num_flt_out[tid][0], flt_out_max_size);
    CHECK_CUDA_ERR(cudaEventRecord(logger->d2h_start[tid], streams[tid]));
    #endif

    h_num_flt_out[tid][0] = h_num_flt_out[tid][0] < flt_out_max_size ? h_num_flt_out[tid][0] : flt_out_max_size;
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_norm_out[tid], d_norm_out[tid], h_num_flt_out[tid][0] * sizeof(int32_t), cudaMemcpyDeviceToHost, streams[tid]));
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_score_out[tid], d_score_out[tid], h_num_flt_out[tid][0] * sizeof(uint16_t), cudaMemcpyDeviceToHost, streams[tid]));
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_u_out[tid], d_u_out[tid], h_num_flt_out[tid][0] * sizeof(uint64_t), cudaMemcpyDeviceToHost, streams[tid]));

    for (int i = 0; i < h_num_flt_out[tid][0]; i += filter_taskVecs) {
        int batch_num = h_num_flt_out[tid][0] - i < filter_taskVecs ? h_num_flt_out[tid][0] - i : filter_taskVecs;
        pk_kernel<<<utils_t::packBlocks, utils_t::packThreads, utils_t::packshmem, streams[tid]>>>(
            data[tid], d_vec_out[tid] + i * CSD16, CSD, batch_num
        );
        CHECK_CUDA_ERR(cudaMemcpyAsync(h_vec_out[tid] + i * CSD, data[tid], batch_num * CSD, cudaMemcpyDeviceToHost, streams[tid]));
    }
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->d2h_stop[tid], streams[tid]));
    #endif
    CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));

    #if ENABLE_PROFILING
    {
        float d2h_tt, fff_tt;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&d2h_tt, logger->d2h_start[tid], logger->d2h_stop[tid]));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&fff_tt, logger->fff_start[tid], logger->fff_stop[tid]));
        logger->ev_d2h_us += 1000.f * d2h_tt;
        logger->ev_fff_us += 1000.f * fff_tt;
        logger->ev_d2h_nbytes += h_num_flt_out[tid][0] * (CSD + 14);
    }
    #endif

    *size = h_num_flt_out[tid][0];
    *h_vec = h_vec_out[tid];
    *h_norm = h_norm_out[tid];
    *h_score = h_score_out[tid];
    *h_u = h_u_out[tid];

    return 0;
}

int red_buffer_holder_t::bgj2_ctr(int tid, int8_t *ctr0) {
    constexpr int vec_nbytes = Pool_hd_t::vec_nbytes;

    vec_t c0;
    memcpy(c0.v, ctr0, vec_nbytes);
    bgj2_ctr_gen<<<buccg_blocks, buccg_threads, buccg_shmem, streams[tid]>>>(
        d_ct1[tid], c0, alpha0, reducer->center_norm, state[tid], CSD, CSD16, batch1
    );

    buc_vecs[tid] = 0;
    CHECK_CUDA_ERR(cudaMemsetAsync(d_bk1[tid], 0, batch1 * sizeof(int), streams[tid]));
    CHECK_CUDA_ERR(cudaMemsetAsync(d_num_red_out[tid], 0, sizeof(int), streams[tid]));

    return 0;
}

int red_buffer_holder_t::bgjm_upk(int tid) {
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->h2d_sstart[tid], streams[tid]));
    #endif
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_norm[tid] + buc_vecs[tid], h_norm[tid], task_vecs[tid] * 4, 
                                   cudaMemcpyHostToDevice, streams[tid]));
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->h2d_sstop[tid], streams[tid]));
    #endif
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_n[tid], &task_vecs[tid], sizeof(int), cudaMemcpyHostToDevice, streams[tid]));
    bk1_kernel<<<buc_traits_t::kernelBlocks, buc_traits_t::blockThreads, buc_traits_t::l1_shmem, streams[tid]>>>(
        (uint32_t *)d_bk1[tid], bk1_max_size, d_vec16[tid] + CSD16 * buc_vecs[tid], d_norm[tid] + buc_vecs[tid], 
                    d_ct1[tid], 0, d_upk[tid], d_n[tid], alpha1, CSD, bk1_gbuc_freq, buc_vecs[tid]
    );
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->upk_stop[tid], streams[tid]));
    #endif
    CHECK_LAST_ERR;
    CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));
    #if ENABLE_PROFILING
    {
        float h2d_tt, bk1_tt;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&h2d_tt, logger->h2d_sstart[tid], logger->h2d_sstop[tid]));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&bk1_tt, logger->h2d_sstop[tid], logger->upk_stop[tid]));
        logger->ev_h2d_us += 1000.f * h2d_tt;
        logger->ev_bk1_us += 1000.f * bk1_tt;
        logger->ev_h2d_nbytes += task_vecs[tid] * (CSD + 4);
    }
    #endif

    buc_vecs[tid] += task_vecs[tid];
    task_vecs[tid] = 0;

    return 0;
}

int red_buffer_holder_t::bgj2_run(int tid) {
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->red_start[tid], streams[tid]));
    #endif

    #if USE_GRAPH
    for (int i = 0; i < batch1; i++) {
        CHECK_CUDA_ERR(cudaGraphExecKernelNodeSetParams(graphExecs[tid], redKernelNodes[tid][i], &redKernelParams[tid][i]));
    }

    CHECK_CUDA_ERR(cudaGraphLaunch(graphExecs[tid], streams[tid]));
    #else
    rpp_kernel<<<batch1, 512, 0, streams[tid]>>>(
        d_bk1[tid] + batch1, d_bk1[tid], bk1_max_size
    );
    red_kernel<<<batch1, traits::blockThreads, traits::l1_shmem, streams[tid]>>>(
        d_red_out[tid], d_num_red_out[tid], out_max_size, d_vec16[tid], d_bk1[tid] + batch1, d_bk1[tid], bk1_max_size, reducer->goal_norm
    );
    #endif

    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->red_stop[tid], streams[tid]));
    #endif

    /// CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));

    return 0;
}

int red_buffer_holder_t::bgj3_ctr(int tid, int8_t *ctr0) {
    constexpr int vec_nbytes = Pool_hd_t::vec_nbytes;

    vec_t c0;
    memcpy(c0.v, ctr0, vec_nbytes);
    bgj2_ctr_gen<<<buccg_blocks, buccg_threads, buccg_shmem, streams[tid]>>>(
        d_ct1[tid], c0, alpha0, reducer->center_norm, state[tid * threads_per_buc], CSD, CSD16, batch1
    );

    buc_vecs[tid] = 0;
    CHECK_CUDA_ERR(cudaMemsetAsync(d_bk1[tid], 0, batch1 * sizeof(int), streams[tid]));

    for (int i = 0; i < threads_per_buc; i++) {
        CHECK_CUDA_ERR(cudaMemsetAsync(d_num_red_out[tid * threads_per_buc + i], 0, sizeof(int), streams[tid]));
    }
    CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));

    return 0;
}

int red_buffer_holder_t::bgj3_run(int tid, int sid, int b) {
    int id = tid * threads_per_buc + sid;
    CHECK_CUDA_ERR(cudaMemsetAsync(d_bk2[id], 0, batch2 * sizeof(int), sstreams[id]));
    bgj34_ctr_gen<<<buccg_blocks, buccg_threads, buccg_shmem, sstreams[id]>>>(
        d_ct2[id], d_vec16[tid], d_bk1[tid] + batch1 + bk1_max_size * 2 * b, d_bk1[tid] + b, 
        bk1_max_size, reducer->center_norm, state[id], CSD, CSD16, batch2
    );
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->bk2_start[id], sstreams[id]));
    #endif
    bk2_kernel<<<buc_traits_t::kernelBlocks, buc_traits_t::blockThreads, buc_traits_t::l2_shmem, sstreams[id]>>>(
        (uint32_t *)d_bk2[id], bk2_max_size, d_vec16[tid], NULL, d_ct2[id], bk1_max_size, 
        (const int8_t *)(d_bk1[tid] + batch1 + bk1_max_size * 2 * b), d_bk1[tid] + b, alpha2, CSD, bk2_gbuc_freq, 0
    );
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->bk2_stop[id], sstreams[id]));
    #endif

    #if USE_GRAPH
    CHECK_CUDA_ERR(cudaGraphLaunch(graphExecs[id], sstreams[id]));
    #else
    rpp_kernel<<<batch2, 512, 0, sstreams[id]>>>(
        d_bk2[id] + batch2, d_bk2[id], bk2_max_size
    );
    red_kernel<<<batch2, traits::blockThreads, traits::l1_shmem, sstreams[id]>>>(
        d_red_out[id], d_num_red_out[id], out_max_size, d_vec16[tid], d_bk2[id] + batch2, d_bk2[id], bk2_max_size, reducer->goal_norm
    );
    #endif
    
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->red_stop[id], sstreams[id]));
    #endif

    /// CHECK_CUDA_ERR(cudaStreamSynchronize(sstreams[id]));

    return 0;
}

int red_buffer_holder_t::bgjm_out(int tid, int sid, int *size, int8_t **h_vec, int32_t **h_norm, uint16_t **h_score, uint64_t **h_u) {
    int id = tid * threads_per_buc + sid;
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_num_flt_out[id], d_num_red_out[id], sizeof(int), cudaMemcpyDeviceToHost, sstreams[id]));
    CHECK_CUDA_ERR(cudaMemsetAsync(d_num_red_out[id], 0, sizeof(int), sstreams[id]));
    CHECK_CUDA_ERR(cudaMemsetAsync(d_num_flt_out[id], 0, sizeof(int), sstreams[id]));
    CHECK_CUDA_ERR(cudaStreamSynchronize(sstreams[id]));
    if ((int)h_num_flt_out[id][0] < 0) lg_err("thread %d | %d, num_red_out overflow(%d), ignored", tid, sid, h_num_flt_out[id][0]);
    int to_flt = h_num_flt_out[id][0] < out_max_size ? h_num_flt_out[id][0] : out_max_size;
    #if ENABLE_PROFILING
    logger->ev_flt_num += 1;
    logger->ev_red_msum += h_num_flt_out[id][0];
    logger->ev_red_max = std::max((int)logger->ev_red_max.load(), h_num_flt_out[id][0]);
    logger->ev_red_ssum += to_flt;
    #endif
    int device_ptr = hw::gpu_ptr(tid, num_threads);
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->fff_start[id], sstreams[id]));
    #endif
    for (int i = 0; i < to_flt; i += filter_taskVecs) {
        int batch_num = to_flt - i < filter_taskVecs ? to_flt - i : filter_taskVecs;
        fpv_kernel<<<fpv_blocks, fpv_threads, fpv_shmem, sstreams[id]>>>(data[id], d_vec16[tid], d_red_out[id] + 2 * i, batch_num);
        flt_kernel<<<check_traits::kernelBlocks, check_traits::blockThreads, check_traits::dynamic_shmem, sstreams[id]>>>(
            data[id], batch_num, local_data[device_ptr]
        );
        fcs_kernel<<<fcs_blocks, fcs_threads, fcs_shmem, sstreams[id]>>>(d_vec_out[id], d_score_out[id], 
            d_norm_out[id], d_u_out[id], d_num_flt_out[id], flt_out_max_size, data[id], batch_num, reducer->goal_score);
    }
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->fff_stop[id], sstreams[id]));
    #endif

    CHECK_CUDA_ERR(cudaMemcpyAsync(h_num_flt_out[id], d_num_flt_out[id], sizeof(int), cudaMemcpyDeviceToHost, sstreams[id]));
    CHECK_CUDA_ERR(cudaStreamSynchronize(sstreams[id]));
    #if ENABLE_PROFILING
    logger->ev_flt_max = std::max((int)logger->ev_flt_max.load(), h_num_flt_out[id][0]);
    logger->ev_flt_ssum += std::min((long)h_num_flt_out[id][0], flt_out_max_size);
    CHECK_CUDA_ERR(cudaEventRecord(logger->d2h_start[id], sstreams[id]));
    #endif
    h_num_flt_out[id][0] = h_num_flt_out[id][0] < flt_out_max_size ? h_num_flt_out[id][0] : flt_out_max_size;

    CHECK_CUDA_ERR(cudaMemcpyAsync(h_norm_out[id], d_norm_out[id], h_num_flt_out[id][0] * sizeof(int32_t), cudaMemcpyDeviceToHost, sstreams[id]));
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_score_out[id], d_score_out[id], h_num_flt_out[id][0] * sizeof(uint16_t), cudaMemcpyDeviceToHost, sstreams[id]));
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_u_out[id], d_u_out[id], h_num_flt_out[id][0] * sizeof(uint64_t), cudaMemcpyDeviceToHost, sstreams[id]));

    for (int i = 0; i < h_num_flt_out[id][0]; i += filter_taskVecs) {
        int batch_num = h_num_flt_out[id][0] - i < filter_taskVecs ? h_num_flt_out[id][0] - i : filter_taskVecs;
        pk_kernel<<<utils_t::packBlocks, utils_t::packThreads, utils_t::packshmem, sstreams[id]>>>(
            data[id], d_vec_out[id] + i * CSD16, CSD, batch_num
        );
        CHECK_CUDA_ERR(cudaMemcpyAsync(h_vec_out[id] + i * CSD, data[id], batch_num * CSD, cudaMemcpyDeviceToHost, sstreams[id]));
    }
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->d2h_stop[id], sstreams[id]));
    #endif
    CHECK_CUDA_ERR(cudaStreamSynchronize(sstreams[id]));
    #if ENABLE_PROFILING
    {
        float d2h_tt, fff_tt;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&d2h_tt, logger->d2h_start[id], logger->d2h_stop[id]));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&fff_tt, logger->fff_start[id], logger->fff_stop[id]));
        logger->ev_d2h_us += 1000.f * d2h_tt;
        logger->ev_fff_us += 1000.f * fff_tt;
        logger->ev_d2h_nbytes += h_num_flt_out[tid][0] * (CSD + 14);
    }
    #endif

    *size = h_num_flt_out[id][0];
    *h_vec = h_vec_out[id];
    *h_norm = h_norm_out[id];
    *h_score = h_score_out[id];
    *h_u = h_u_out[id];

    return 0;
}

int red_buffer_holder_t::bgjl_out(int tid, int sid, int *size, int8_t **h_vec, int32_t **h_norm, uint16_t **h_score, uint64_t **h_u) {
    int id = tid * threads_per_buc + sid;
    #if ENABLE_PROFILING
    logger->ev_flt_max = std::max((int)logger->ev_flt_max.load(), h_num_flt_out[id][0]);
    logger->ev_flt_ssum += std::min((long)h_num_flt_out[id][0], flt_out_max_size);
    #endif
    h_num_flt_out[id][0] = h_num_flt_out[id][0] < flt_out_max_size ? h_num_flt_out[id][0] : flt_out_max_size;
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->d2h_start[id], sstreams[id]));
    #endif
    CHECK_CUDA_ERR(cudaMemsetAsync(d_num_flt_out[id], 0, sizeof(int), sstreams[id]));
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_norm_out[id], d_norm_out[id], h_num_flt_out[id][0] * sizeof(int32_t), cudaMemcpyDeviceToHost, sstreams[id]));
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_score_out[id], d_score_out[id], h_num_flt_out[id][0] * sizeof(uint16_t), cudaMemcpyDeviceToHost, sstreams[id]));
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_u_out[id], d_u_out[id], h_num_flt_out[id][0] * sizeof(uint64_t), cudaMemcpyDeviceToHost, sstreams[id]));

    for (int i = 0; i < h_num_flt_out[id][0]; i += filter_taskVecs) {
        int batch_num = h_num_flt_out[id][0] - i < filter_taskVecs ? h_num_flt_out[id][0] - i : filter_taskVecs;
        pk_kernel<<<utils_t::packBlocks, utils_t::packThreads, utils_t::packshmem, sstreams[id]>>>(
            data[id], d_vec_out[id] + i * CSD16, CSD, batch_num
        );
        CHECK_CUDA_ERR(cudaMemcpyAsync(h_vec_out[id] + i * CSD, data[id], batch_num * CSD, cudaMemcpyDeviceToHost, sstreams[id]));
    }
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->d2h_stop[id], sstreams[id]));
    #endif
    CHECK_CUDA_ERR(cudaStreamSynchronize(sstreams[id]));
    #if ENABLE_PROFILING
    {
        float d2h_tt;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&d2h_tt, logger->d2h_start[id], logger->d2h_stop[id]));
        logger->ev_d2h_us += 1000.f * d2h_tt;
        logger->ev_d2h_nbytes += h_num_flt_out[id][0] * (CSD + 14);
    }
    #endif

    *size = h_num_flt_out[id][0];
    *h_vec = h_vec_out[id];
    *h_norm = h_norm_out[id];
    *h_score = h_score_out[id];
    *h_u = h_u_out[id];

    return 0;
}

int red_buffer_holder_t::bgjl_buc_ctr(int tid, int8_t *ctr0) {
    constexpr int vec_nbytes = Pool_hd_t::vec_nbytes;

    vec_t c0;
    memcpy(c0.v, ctr0, vec_nbytes);
    bgj2_ctr_gen<<<buccg_blocks, buccg_threads, buccg_shmem, streams[tid]>>>(
        d_ctt1[tid], c0, alpha0, reducer->center_norm, statte[tid], CSD, CSD16, batch1
    );
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_ct1[tid], d_ctt1[tid], batch1 * CSD16, cudaMemcpyDeviceToHost, streams[tid]));

    memset(h_bk1[tid], 0, batch1 * sizeof(int));
    
    for (int i = 0; i < threads_per_buc; i++) {
        task_vecs[tid * threads_per_buc + i] = 0;
    }
    CHECK_CUDA_ERR(cudaStreamSynchronize(streams[tid]));

    return 0;
}

int red_buffer_holder_t::bgjl_buc_h2d(int tid, int sid, chunk_t *chunk) {
    int id = tid * threads_per_buc + sid;
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->h2d_start[id][logger->h2d_count[id]], sstreams[id]));
    #endif
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_upk[id] + task_vecs[id] * CSD, chunk->vec, 
                                   chunk->size * CSD, cudaMemcpyHostToDevice, sstreams[id]));
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->h2d_stop[id][logger->h2d_count[id]], sstreams[id]));
    logger->h2d_count[id]++;
    #endif
    memcpy(h_norm[id] + task_vecs[id], chunk->norm, chunk->size * sizeof(int32_t));
    task_vecs[id] += chunk->size;

    return chunk->size;
}

int red_buffer_holder_t::bgjl_buc_run(int tid, int sid) {
    int id = tid * threads_per_buc + sid;
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->h2d_norm_start[id], sstreams[id]));
    #endif
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_norm[id], h_norm[id], task_vecs[id] * sizeof(int32_t), 
                                   cudaMemcpyHostToDevice, sstreams[id]));
    CHECK_CUDA_ERR(cudaMemsetAsync(d_bk1[id], 0, batch1 * sizeof(int), sstreams[id]));
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->bk1_start[id], sstreams[id]));
    #endif
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_n[id], &task_vecs[id], sizeof(int), cudaMemcpyHostToDevice, sstreams[id]));
    bk1_kernel<<<buc_traits_t::kernelBlocks, buc_traits_t::blockThreads, buc_traits_t::l1_shmem, sstreams[id]>>>(
        (uint32_t *)d_bk1[id], l1_out_max_size, data[id], d_norm[id], d_ct1[id], 0, d_upk[id], d_n[id], alpha1, CSD, bk1_gbuc_freq, 0
    );
    CHECK_LAST_ERR;
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->d2h_start[id], sstreams[id]));
    #endif
    CHECK_CUDA_ERR(cudaMemcpyAsync(h_bko[id], d_bk1[id], batch1 * (2 * l1_out_max_size + 1) * 4, cudaMemcpyDeviceToHost, sstreams[id]));
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->d2h_stop[id], sstreams[id]));
    #endif
    CHECK_CUDA_ERR(cudaStreamSynchronize(sstreams[id]));
    for (int i = 0; i < batch1; i++) {
        if (h_bko[id][i] > l1_out_max_size) h_bko[id][i] = l1_out_max_size;
    }
    #if ENABLE_PROFILING
    {
        float h2d_tt, d2h_tt, bk1_tt;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&h2d_tt, logger->h2d_norm_start[id], logger->bk1_start[id]));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&bk1_tt, logger->bk1_start[id], logger->d2h_start[id]));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&d2h_tt, logger->d2h_start[id], logger->d2h_stop[id]));
        for (int i = 0; i < logger->h2d_count[id]; i++) {
            float tt;
            CHECK_CUDA_ERR(cudaEventElapsedTime(&tt, logger->h2d_start[id][i], logger->h2d_stop[id][i]));
            h2d_tt += tt;
        }
        logger->h2d_count[id] = 0;
        logger->ev_bk1_us += 1000.f * bk1_tt;
        logger->ev_d2h_us += 1000.f * d2h_tt;
        logger->ev_h2d_us += 1000.f * h2d_tt;
        logger->ev_d2h_nbytes += batch1 * (2 * l1_out_max_size + 1) * 4;
        logger->ev_h2d_nbytes += task_vecs[id] * (CSD + 4);
    }
    #endif
    task_vecs[id] = 0;

    return 0;
}

int red_buffer_holder_t::bgjl_buc_out(int tid, int sid, int **buc_out) {
    int id = tid * threads_per_buc + sid;
    *buc_out = h_bko[id];

    return 0;
}

int red_buffer_holder_t::bgjl_ctr(int tid, int sid, int bk1_size, int *bk1_ptr, chunk_t **working_chunks) {
    #if !BGJL_HOST_UPK
    int id = tid * threads_per_buc + sid;

    int center_norm = reducer->center_norm;
    constexpr int chunk_max_nvecs = Pool_hd_t::chunk_max_nvecs;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, bk1_size - 1);

    for (int i = 0; i < batch2; i++) {
        int ind = dis(gen);
        uint32_t ptr = bk1_ptr[ind * 2];
        uint32_t ptr_hi = (((uint32_t)bk1_ptr[ind * 2 + 1]) >> 24) << 19;
        memcpy(h_ct2[id] + i * CSD16, working_chunks[ptr / chunk_max_nvecs + ptr_hi]->vec + (ptr % chunk_max_nvecs) * CSD, CSD);
        memset(h_ct2[id] + i * CSD16 + CSD, 0, CSD16 - CSD);
        h_ct2[id][i * CSD16] = 0; 
    }

    for (int i = 0; i < batch2; i++) {
        int norm = 0;
        for (int j = 0; j < CSD; j++) {
            norm += (int) h_ct2[id][i * CSD16 + j] * (int) h_ct2[id][i * CSD16 + j];
        }
        float ratio = sqrtf(2.0f * center_norm / (float) norm);
        for (int j = 0; j < CSD; j++) {
            h_ct2[id][i * CSD16 + j] = (int8_t) (h_ct2[id][i * CSD16 + j] * ratio);
        }

        norm = 0;
        for (int j = 0; j < CSD; j++) {
            norm += (int) h_ct2[id][i * CSD16 + j] * (int) h_ct2[id][i * CSD16 + j];
        }

        int rn = norm - center_norm * 2;
        int8_t *v = h_ct2[id] + i * CSD16;

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
    }

    CHECK_CUDA_ERR(cudaMemcpyAsync(d_ct2[id], h_ct2[id], batch2 * CSD16, cudaMemcpyHostToDevice, sstreams[id]));

    CHECK_CUDA_ERR(cudaMemsetAsync(d_bk2[id], 0, batch2 * sizeof(int), sstreams[id]));
    #endif

    return 0;
}

int red_buffer_holder_t::bgjl_h2d(int tid, int sid, int num) {
    int id = tid * threads_per_buc + sid;

    #if BGJL_HOST_UPK
    #else
    int32_t *hn_ptr = h_norm[id];
    int8_t *hv_ptr = h_vec_out[id];
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_upk[id], hv_ptr, num * CSD, cudaMemcpyHostToDevice, sstreams[id]));
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_norm[id], hn_ptr, num * sizeof(int32_t), cudaMemcpyHostToDevice, sstreams[id]));
    #endif

    task_vecs[id] = num;

    return 0;
}

int red_buffer_holder_t::bgjl_upk(int tid, int sid, int ind_bias) {
    int id = tid * threads_per_buc + sid;

    #if BGJL_HOST_UPK
    int8_t *hv_ptr = h_vec_out[id];
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_vec16[id] + ind_bias * CSD16, hv_ptr, task_vecs[id] * CSD16, cudaMemcpyHostToDevice, sstreams[id]));
    #else
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_n[id], &task_vecs[id], sizeof(int), cudaMemcpyHostToDevice, sstreams[id]));
    bk2_kernel<<<buc_traits_t::kernelBlocks, buc_traits_t::blockThreads, buc_traits_t::l1_shmem, sstreams[id]>>>(
        (uint32_t *)d_bk2[id], bk2_max_size, d_vec16[id] + ind_bias * CSD16, d_norm[id], d_ct2[id], 0, d_upk[id], d_n[id], alpha2, CSD, bk2_gbuc_freq, ind_bias
    );
    #endif
    
    return 0;
}

int red_buffer_holder_t::bgj3l_run(int tid, int sid) {
    int id = tid * threads_per_buc + sid;

    #if !BGJL_HOST_UPK
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->red_start[id], sstreams[id]));
    #endif

    #if USE_GRAPH
    CHECK_CUDA_ERR(cudaGraphLaunch(graphExecs[id], sstreams[id]));
    #else
    rpp_kernel<<<batch2, 512, 0, sstreams[id]>>>(
        d_bk2[id] + batch2, d_bk2[id], bk2_max_size
    );
    red_kernel<<<batch2, traits::blockThreads, traits::l1_shmem, sstreams[id]>>>(
        d_red_out[id], d_num_red_out[id], out_max_size, d_vec16[id], d_bk2[id] + batch2, d_bk2[id], bk2_max_size, reducer->goal_norm
    );
    #endif

    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->red_stop[id], sstreams[id]));
    #endif
    #endif

    for (int l = (BGJL_HOST_UPK ? 0 : 1); l < bgj3l_repeat; l++) {
        #if USE_GRAPH
        CHECK_CUDA_ERR(cudaStreamSynchronize(sstreams[id]));
        if (l > 1 && h_num_flt_out[id][0] > out_max_size) break;
        #endif
        CHECK_CUDA_ERR(cudaMemsetAsync(d_bk2[id], 0, batch2 * sizeof(int), sstreams[id]));
        #if USE_GRAPH
        CHECK_CUDA_ERR(cudaMemcpyAsync(h_num_flt_out[id], d_num_red_out[id], sizeof(int), cudaMemcpyDeviceToHost, sstreams[id]));
        #endif
        bgj34_ctr_gen<<<buccg_blocks, buccg_threads, buccg_shmem, sstreams[id]>>>(
            d_ct2[id], d_vec16[id], repeat_buf[id], repeat_buf_size[id], 
            bk1_max_size, reducer->center_norm, state[id], CSD, CSD16, batch2
        );
        #if ENABLE_PROFILING && BGJL_HOST_UPK
        if (l == 0) CHECK_CUDA_ERR(cudaEventRecord(logger->bk2_start[id], sstreams[id]));
        #endif
        rep_kernel<<<128, buc_traits_t::blockThreads, buc_traits_t::l2_shmem, sstreams[id]>>>(
            (uint32_t *)d_bk2[id], bk2_max_size, d_vec16[id], NULL, d_ct2[id], bk1_max_size, 
            (const int8_t *)(repeat_buf[id]), repeat_buf_size[id], alpha2, CSD, bk2_gbuc_freq, 0
        );
        #if ENABLE_PROFILING && BGJL_HOST_UPK
        if (l == 0) CHECK_CUDA_ERR(cudaEventRecord(logger->red_start[id], sstreams[id]));
        #endif
        #if USE_GRAPH
        CHECK_CUDA_ERR(cudaGraphLaunch(graphExecs[id], sstreams[id]));
        #else
        rpp_kernel<<<batch2, 512, 0, sstreams[id]>>>(
            d_bk2[id] + batch2, d_bk2[id], bk2_max_size
        );
        red_kernel<<<batch2, traits::blockThreads, traits::l1_shmem, sstreams[id]>>>(
            d_red_out[id], d_num_red_out[id], out_max_size, d_vec16[id], d_bk2[id] + batch2, d_bk2[id], bk2_max_size, reducer->goal_norm
        );
        #if ENABLE_PROFILING && BGJL_HOST_UPK
        if (l == 0) CHECK_CUDA_ERR(cudaEventRecord(logger->red_stop[id], sstreams[id]));
        #endif
        #endif
    }

    #if BGJL_HOST_UPK && ENABLE_PROFILING
    int bk2_size[BGJ3L_DEFAULT_BATCH2];
    CHECK_CUDA_ERR(cudaMemcpyAsync(bk2_size, d_bk2[id], batch2 * sizeof(int), cudaMemcpyDeviceToHost, sstreams[id]));
    #endif

    CHECK_CUDA_ERR(cudaMemcpyAsync(h_num_flt_out[id], d_num_red_out[id], sizeof(int), cudaMemcpyDeviceToHost, sstreams[id]));
    CHECK_CUDA_ERR(cudaMemsetAsync(d_num_red_out[id], 0, sizeof(int), sstreams[id]));

    CHECK_CUDA_ERR(cudaStreamSynchronize(sstreams[id]));
    if ((int)h_num_flt_out[id][0] < 0) lg_err("thread %d | %d, num_red_out overflow(%d), ignored", tid, sid, h_num_flt_out[id][0]);
    int to_flt = h_num_flt_out[id][0] < out_max_size ? h_num_flt_out[id][0] : out_max_size;
    #if ENABLE_PROFILING
    #if BGJL_HOST_UPK
    float bk2_tt;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&bk2_tt, logger->bk2_start[id], logger->red_start[id]));
    logger->ev_bk2_us += bgj3l_repeat * 1000.f * bk2_tt;
    for (int i = 0; i < batch2; i++) {
        logger->ev_bk2_max   = std::max((int)logger->ev_bk2_max.load(), bk2_size[i]);
        logger->ev_bk2_ssum += bgj3l_repeat * bk2_size[i];
        logger->ev_red_vmmas += bgj3l_repeat * ceil(bk2_size[i] / 16.0) * ceil(bk2_size[i] / 16.0 + 1.0) * 0.5;
    }
    #endif
    float red_tt;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&red_tt, logger->red_start[id], logger->red_stop[id]));
    logger->ev_red_us += bgj3l_repeat * 1000.f * red_tt;
    logger->ev_flt_num += 1;
    logger->ev_red_msum += h_num_flt_out[id][0];
    logger->ev_red_max = std::max((int)logger->ev_red_max.load(), h_num_flt_out[id][0]);
    logger->ev_red_ssum += to_flt;
    #endif

    #if 0
    to_flt = device_remove_dup(d_red_out[id], d_num_red_out[id], to_flt, sstreams[id]);

    #if ENABLE_PROFILING
    logger->ev_red_usum += to_flt;
    #endif
    #endif

    int device_ptr = hw::gpu_ptrl(tid, sid, num_threads, threads_per_buc);
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->fff_start[id], sstreams[id]));
    #endif
    for (int i = 0; i < to_flt; i += filter_taskVecs) {
        int batch_num = to_flt - i < filter_taskVecs ? to_flt - i : filter_taskVecs;
        fpv_kernel<<<fpv_blocks, fpv_threads, fpv_shmem, sstreams[id]>>>(data[id], d_vec16[id], d_red_out[id] + 2 * i, batch_num);
        flt_kernel<<<check_traits::kernelBlocks, check_traits::blockThreads, check_traits::dynamic_shmem, sstreams[id]>>>(
            data[id], batch_num, local_data[device_ptr]
        );
        fcs_kernel<<<fcs_blocks, fcs_threads, fcs_shmem, sstreams[id]>>>(d_vec_out[id], d_score_out[id], 
            d_norm_out[id], d_u_out[id], d_num_flt_out[id], flt_out_max_size, data[id], batch_num, reducer->goal_score);
    }
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->fff_stop[id], sstreams[id]));
    #endif

    CHECK_CUDA_ERR(cudaMemsetAsync(d_bk2[id], 0, batch2 * sizeof(int), sstreams[id]));


    return 0;
}

int red_buffer_holder_t::bgj4_run(int tid, int sid) {
    int id = tid * threads_per_buc + sid;

    #if !BGJL_HOST_UPK
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->red_start[id], sstreams[id]));
    #endif

    for (int i = 0; i < batch2; i++) {
        CHECK_CUDA_ERR(cudaMemsetAsync(d_bk3[id], 0, batch3 * sizeof(int), sstreams[id]));
        bgj34_ctr_gen<<<buccg_blocks, buccg_threads, buccg_shmem, sstreams[id]>>>(
            d_ct3[id], d_vec16[id], d_bk2[id] + batch2 + bk2_max_size * 2 * i, d_bk2[id] + i, 
            bk2_max_size, reducer->center_norm, state[id], CSD, CSD16, batch3
        );
        bk3_kernel<<<buc_traits_t::kernelBlocks, buc_traits_t::blockThreads, buc_traits_t::l2_shmem, sstreams[id]>>>(
            (uint32_t *)d_bk3[id], bk3_max_size, d_vec16[id], NULL, d_ct3[id], bk2_max_size, 
            (const int8_t *)(d_bk2[id] + batch2 + bk2_max_size * 2 * i), d_bk2[id] + i, alpha3, CSD, bk3_gbuc_freq, 0
        );

        #if USE_GRAPH
        CHECK_CUDA_ERR(cudaGraphLaunch(graphExecs[id], sstreams[id]));
        CHECK_CUDA_ERR(cudaStreamSynchronize(sstreams[id]));
        #else
        rpp_kernel<<<batch3, 512, 0, sstreams[id]>>>(
            d_bk3[id] + batch3, d_bk3[id], bk3_max_size
        );
        red_kernel<<<batch3, traits::blockThreads, traits::l1_shmem, sstreams[id]>>>(
            d_red_out[id], d_num_red_out[id], out_max_size, d_vec16[id], d_bk3[id] + batch3, d_bk3[id], bk3_max_size, reducer->goal_norm
        );
        #endif
    }
    #endif

    for (int l = (BGJL_HOST_UPK ? 0 : 1); l < bgj3l_repeat; l++) {
        #if USE_GRAPH
        CHECK_CUDA_ERR(cudaStreamSynchronize(sstreams[id]));
        if (l > 1 && h_num_flt_out[id][0] > out_max_size) break;
        #endif
        CHECK_CUDA_ERR(cudaMemsetAsync(d_bk2[id], 0, batch2 * sizeof(int), sstreams[id]));
        #if USE_GRAPH
        CHECK_CUDA_ERR(cudaMemcpyAsync(h_num_flt_out[id], d_num_red_out[id], sizeof(int), cudaMemcpyDeviceToHost, sstreams[id]));
        #endif
        bgj34_ctr_gen<<<buccg_blocks, buccg_threads, buccg_shmem, sstreams[id]>>>(
            d_ct2[id], d_vec16[id], repeat_buf[id], repeat_buf_size[id], 
            bk1_max_size, reducer->center_norm, state[id], CSD, CSD16, batch2
        );
        #if ENABLE_PROFILING && BGJL_HOST_UPK
        if (l == 0) CHECK_CUDA_ERR(cudaEventRecord(logger->bk2_start[id], sstreams[id]));
        #endif
        rep_kernel<<<buc_traits_t::kernelBlocks, buc_traits_t::blockThreads, buc_traits_t::l2_shmem, sstreams[id]>>>(
            (uint32_t *)d_bk2[id], bk2_max_size, d_vec16[id], NULL, d_ct2[id], bk1_max_size, 
            (const int8_t *)(repeat_buf[id]), repeat_buf_size[id], alpha2, CSD, bk2_gbuc_freq, 0
        );
        #if ENABLE_PROFILING && BGJL_HOST_UPK
        if (l == 0) CHECK_CUDA_ERR(cudaEventRecord(logger->red_start[id], sstreams[id]));
        #endif
        for (int i = 0; i < batch2; i++) {
            CHECK_CUDA_ERR(cudaMemsetAsync(d_bk3[id], 0, batch3 * sizeof(int), sstreams[id]));
            bgj34_ctr_gen<<<buccg_blocks, buccg_threads, buccg_shmem, sstreams[id]>>>(
                d_ct3[id], d_vec16[id], d_bk2[id] + batch2 + bk2_max_size * 2 * i, d_bk2[id] + i, 
                bk2_max_size, reducer->center_norm, state[id], CSD, CSD16, batch3
            );
            bk3_kernel<<<buc_traits_t::kernelBlocks, buc_traits_t::blockThreads, buc_traits_t::l2_shmem, sstreams[id]>>>(
                (uint32_t *)d_bk3[id], bk3_max_size, d_vec16[id], NULL, d_ct3[id], bk2_max_size, 
                (const int8_t *)(d_bk2[id] + batch2 + bk2_max_size * 2 * i), d_bk2[id] + i, alpha3, CSD, bk3_gbuc_freq, 0
            );
            #if USE_GRAPH
            CHECK_CUDA_ERR(cudaGraphLaunch(graphExecs[id], sstreams[id]));
            CHECK_CUDA_ERR(cudaStreamSynchronize(sstreams[id]));
            #else
            rpp_kernel<<<batch3, 512, 0, sstreams[id]>>>(
                d_bk3[id] + batch3, d_bk3[id], bk3_max_size
            );
            red_kernel<<<batch3, traits::blockThreads, traits::l1_shmem, sstreams[id]>>>(
                d_red_out[id], d_num_red_out[id], out_max_size, d_vec16[id], d_bk3[id] + batch2, d_bk3[id], bk3_max_size, reducer->goal_norm
            );
            #endif
        }
        #if ENABLE_PROFILING && BGJL_HOST_UPK
        if (l == 0) CHECK_CUDA_ERR(cudaEventRecord(logger->red_stop[id], sstreams[id]));
        #endif
    }

    #if ENABLE_PROFILING
    #if BGJL_HOST_UPK
    int bk2_size[BGJ4_DEFAULT_BATCH2];
    int bk3_size[BGJ4_DEFAULT_BATCH3];
    CHECK_CUDA_ERR(cudaMemcpyAsync(bk2_size, d_bk2[id], batch2 * sizeof(int), cudaMemcpyDeviceToHost, sstreams[id]));
    CHECK_CUDA_ERR(cudaMemcpyAsync(bk3_size, d_bk3[id], batch3 * sizeof(int), cudaMemcpyDeviceToHost, sstreams[id]));
    #else
    int bk3_size[BGJ4_DEFAULT_BATCH3];
    CHECK_CUDA_ERR(cudaEventRecord(logger->red_stop[id], sstreams[id]));
    CHECK_CUDA_ERR(cudaMemcpyAsync(bk3_size, d_bk3[id], batch3 * sizeof(int), cudaMemcpyDeviceToHost, sstreams[id]));
    #endif
    #endif

    CHECK_CUDA_ERR(cudaMemcpyAsync(h_num_flt_out[id], d_num_red_out[id], sizeof(int), cudaMemcpyDeviceToHost, sstreams[id]));
    CHECK_CUDA_ERR(cudaMemsetAsync(d_num_red_out[id], 0, sizeof(int), sstreams[id]));
    CHECK_CUDA_ERR(cudaStreamSynchronize(sstreams[id]));
    #if ENABLE_PROFILING
    #if BGJL_HOST_UPK
    float bk2_tt;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&bk2_tt, logger->bk2_start[id], logger->red_start[id]));
    logger->ev_bk2_us += bgj4_repeat * 1000.f * bk2_tt;
    for (int i = 0; i < batch2; i++) {
        logger->ev_bk2_max   = std::max((int)logger->ev_bk2_max.load(), bk2_size[i]);
        logger->ev_bk2_ssum += bgj4_repeat * bk2_size[i];
        logger->ev_bk3_vmmas += bgj4_repeat * ceil(bk2_size[i] / 16.0) * batch3 / 16.0;
    }
    #endif
    float red_tt;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&red_tt, logger->red_start[id], logger->red_stop[id]));
    logger->ev_red_us += bgj4_repeat * 1000.f * red_tt;
    logger->ev_bk3_num += bgj4_repeat * batch2 * batch3;
    for (int i = 0; i < batch3; i++) {
        logger->ev_bk3_max = std::max((int)logger->ev_bk3_max.load(), bk3_size[i]);
        logger->ev_bk3_ssum += bgj4_repeat * bk3_size[i] * batch2;
        logger->ev_red_vmmas += bgj4_repeat * batch2 * ceil(bk3_size[i] / 16.0) * ceil(bk3_size[i] / 16.0 + 1) * 0.5;
    }
    #endif
    if ((int)h_num_flt_out[id][0] < 0) lg_err("thread %d | %d, num_red_out overflow(%d), ignored", tid, sid, h_num_flt_out[id][0]);
    int to_flt = h_num_flt_out[id][0] < out_max_size ? h_num_flt_out[id][0] : out_max_size;
    #if ENABLE_PROFILING
    logger->ev_flt_num += 1;
    logger->ev_red_msum += h_num_flt_out[id][0];
    logger->ev_red_max = std::max((int)logger->ev_red_max.load(), h_num_flt_out[id][0]);
    logger->ev_red_ssum += to_flt;
    #endif

    #if 0
    to_flt = device_remove_dup(d_red_out[id], d_num_red_out[id], to_flt, sstreams[id]);

    #if ENABLE_PROFILING
    logger->ev_red_usum += to_flt;
    #endif
    #endif

    int device_ptr = hw::gpu_ptrl(tid, sid, num_threads, threads_per_buc);
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->fff_start[id], sstreams[id]));
    #endif
    for (int i = 0; i < to_flt; i += filter_taskVecs) {
        int batch_num = to_flt - i < filter_taskVecs ? to_flt - i : filter_taskVecs;
        fpv_kernel<<<fpv_blocks, fpv_threads, fpv_shmem, sstreams[id]>>>(data[id], d_vec16[id], d_red_out[id] + 2 * i, batch_num);
        flt_kernel<<<check_traits::kernelBlocks, check_traits::blockThreads, check_traits::dynamic_shmem, sstreams[id]>>>(
            data[id], batch_num, local_data[device_ptr]
        );
        fcs_kernel<<<fcs_blocks, fcs_threads, fcs_shmem, sstreams[id]>>>(d_vec_out[id], d_score_out[id], 
            d_norm_out[id], d_u_out[id], d_num_flt_out[id], flt_out_max_size, data[id], batch_num, reducer->goal_score);
    }
    #if ENABLE_PROFILING
    CHECK_CUDA_ERR(cudaEventRecord(logger->fff_stop[id], sstreams[id]));
    #endif

    CHECK_CUDA_ERR(cudaMemsetAsync(d_bk2[id], 0, batch2 * sizeof(int), sstreams[id]));

    return 0;
}



Reducer_t::Reducer_t(Pool_hd_t *pool, bwc_manager_t *bwc, swc_manager_t *swc, ut_checker_t *ut_checker) {
    #if ENABLE_PROFILING
    this->logger = new logger_t();
    logger->clear();
    logger->ev_total_check_ptr = &total_check;
    logger->ev_total_notin_ptr = &total_notin;
    logger->ev_goal_score_ptr  = (int32_t *)&goal_score;
    logger->reducer = this;
    #endif

    pthread_spin_init(&stuck_stat_lock, PTHREAD_PROCESS_SHARED);
    pthread_spin_init(&traffic_ctrl_lock, PTHREAD_PROCESS_SHARED);

    this->set_pool(pool);
    this->set_bwc_manager(bwc);
    this->set_swc_manager(swc);
    this->set_ut_checker(ut_checker);
    ut_checker->set_stuck_stat(&total_check, &total_notin, stuck_stat_lock);
}

Reducer_t::~Reducer_t() {
    pthread_spin_destroy(&stuck_stat_lock);
    pthread_spin_destroy(&traffic_ctrl_lock);
    if (_red_pool) {
        for (int i = 0; i < _num_threads; i++) delete _red_pool[i];
        free(_red_pool);
        _red_pool = NULL;
    }
    if (_sub_threads) {
        for (int i = 0; i < _threads_per_buc * _num_threads; i++) delete _sub_threads[i];
        free(_sub_threads);
        _sub_threads = NULL;
    }
    #if ENABLE_PROFILING
    delete this->logger;
    #endif
}

int Reducer_t::set_num_threads(long num_threads) {
    if (_red_pool) {
        for (int i = 0; i < this->_num_threads; i++) delete _red_pool[i];
        free(_red_pool);
    }
    if (_threads_per_buc) {
        if (_sub_threads) {
            for (int i = 0; i < _threads_per_buc * this->_num_threads; i++) delete _sub_threads[i];
        }
    }

    this->_num_threads = num_threads;
    
    _red_pool = (thread_pool::thread_pool **)malloc(num_threads * sizeof(thread_pool::thread_pool *));
    for (int i = 0; i < num_threads; i++) _red_pool[i] = new thread_pool::thread_pool(1);
    if (_threads_per_buc) {
        _sub_threads = (thread_pool::thread_pool **)malloc(_threads_per_buc * num_threads * sizeof(thread_pool::thread_pool *));
        for (int i = 0; i < _threads_per_buc * num_threads; i++) _sub_threads[i] = new thread_pool::thread_pool(1);
    }

    return 0;
}

int Reducer_t::auto_bgj_params_set(int bgj) {
    int ret = 0;

    if (!this->_ssd_slimit) this->_ssd_slimit  = SWC_SSD_SLIMIT;
    if (!this->_dram_slimit) this->_dram_slimit = SWC_DRAM_SLIMIT;
    if (!this->_gram_slimit) this->_gram_slimit = RED_GRAM_SLIMIT;

    double exp_chunk_nbytes = (14. + _pool->CSD) * Pool_hd_t::chunk_max_nvecs;

    this->_num_sol_chunks_slimit = floor(_ssd_slimit / exp_chunk_nbytes);
    long _max_sol_chunks = ceil(_bucketer->_size_ratio * pow(4./3., _pool->CSD * .5) / Pool_hd_t::chunk_max_nvecs);
    if (_pool->pwc_manager->num_chunks() > _max_sol_chunks) _max_sol_chunks = _pool->pwc_manager->num_chunks();
    if (_num_sol_chunks_slimit > _max_sol_chunks + 1000) _num_sol_chunks_slimit = _max_sol_chunks + 1000;

    int CSD = _pool->CSD;

    if (bgj == 1) {
        this->_strategy    = strategy_bgj1;

        this->_threads_per_buc = 0;
    }
    if (bgj == 2) {
        this->_strategy   = strategy_bgj2;
        this->_alpha1     = BGJ2_DEFAULT_ALPHA1;
        this->_batch1     = BGJ2_DEFAULT_BATCH1;

        this->_threads_per_buc = 0;
    }
    if (bgj == 3) {
        this->_strategy = strategy_bgj3;
        this->_alpha1   = BGJ3_DEFAULT_ALPHA1;
        this->_batch1   = BGJ3_DEFAULT_BATCH1;
        this->_alpha2   = BGJ3_DEFAULT_ALPHA2;
        this->_batch2   = BGJ3_DEFAULT_BATCH2;
        this->bgj3_repeat = ceil(2.5 * pow(2.0, 0.07 * CSD) / 512.0);

        this->_threads_per_buc = BGJ3_DEFAULT_THREADS_PER_BUC;
    }
    if (bgj == 4) {
        this->_strategy = strategy_bgj3l;
        this->_alpha1   = BGJ3L_DEFAULT_ALPHA1;
        this->_batch1   = BGJ3L_DEFAULT_BATCH1;
        this->_alpha2   = BGJ3L_DEFAULT_ALPHA2;
        this->_batch2   = BGJ3L_DEFAULT_BATCH2;
        this->bgj3l_repeat = ceil(3.0 * pow(2.0, 0.066666666 * CSD) / 512.0);

        this->_threads_per_buc = BGJ3L_DEFAULT_THREADS_PER_BUC;
    }
    if (bgj == 5) {
        this->_strategy   = strategy_bgj4;
        this->_alpha1     = BGJ4_DEFAULT_ALPHA1;
        this->_batch1     = BGJ4_DEFAULT_BATCH1;
        this->_alpha2     = BGJ4_DEFAULT_ALPHA2;
        this->_batch2     = BGJ4_DEFAULT_BATCH2;
        this->_alpha3     = BGJ4_DEFAULT_ALPHA3;
        this->_batch3     = BGJ4_DEFAULT_BATCH3;
        this->bgj4_repeat = ceil(2.5 * pow(2.0, 0.07 * CSD) / 512.0);

        this->_threads_per_buc = BGJ4_DEFAULT_THREADS_PER_BUC;
    }

    const int cache_for_prefetch = bwc_manager_t::bwc_auto_prefetch_for_read * 
                                   bwc_manager_t::bwc_auto_prefetch_for_read_depth + 
                                   bwc_manager_t::bwc_auto_prefetch_for_write;
    const int cache_per_thread  = (traits::buc_max_size(_pool->CSD, _pool->ESD, _strategy) - 1) / Pool_hd_t::chunk_max_nvecs + 1;
    int ram_limit = (_bwc->max_cached_chunks() - cache_for_prefetch) / cache_per_thread;
    int expect_num_threads = traits::num_threads(_pool->CSD, _pool->ESD, _strategy);
    if (ram_limit < expect_num_threads && (_strategy == strategy_bgj3l || _strategy == strategy_bgj4)) {
        lg_warn("limit from bwc #cache %d < expected num threads %d", ram_limit, expect_num_threads);
        expect_num_threads = ram_limit;
    }
    if (expect_num_threads < 1) expect_num_threads = 1;
    if (!this->_num_threads) this->set_num_threads(expect_num_threads);

    if (_strategy == strategy_bgj1) {
        lg_dbg("CSD %d, strategy bgj1, #threads %ld, #swc chunks <= %ld(%.2f TB)",
                _pool->CSD, _num_threads, _num_sol_chunks_slimit, _num_sol_chunks_slimit * exp_chunk_nbytes / 1e12);
    }
    if (_strategy == strategy_bgj2) {
        lg_dbg("CSD %d, strategy bgj2, #threads %ld, #swc chunks <= %ld(%.2f TB), alpha1 %.3f, batch1 %d",
                _pool->CSD, _num_threads, _num_sol_chunks_slimit, _num_sol_chunks_slimit * exp_chunk_nbytes / 1e12, _alpha1, _batch1);
    }
    if (_strategy == strategy_bgj3) {
        lg_dbg("CSD %d, strategy bgj3, #threads %ld, #swc chunks <= %ld(%.2f TB), alpha1 %.3f, batch1 %d, alpha2 %.3f, batch2 %d * %d",
                _pool->CSD, _num_threads, _num_sol_chunks_slimit, _num_sol_chunks_slimit * exp_chunk_nbytes / 1e12, _alpha1, _batch1, _alpha2, _batch2, bgj3_repeat);
    }
    if (_strategy == strategy_bgj3l) {
        lg_dbg("CSD %d, strategy bgj3l, #threads %ld, #swc chunks <= %ld(%.2f TB), alpha1 %.3f, batch1 %d, alpha2 %.3f, batch2 %d * %d",
                _pool->CSD, _num_threads, _num_sol_chunks_slimit, _num_sol_chunks_slimit * exp_chunk_nbytes / 1e12, _alpha1, _batch1, _alpha2, _batch2, bgj3l_repeat);
    }
    if (_strategy == strategy_bgj4) {
        lg_dbg("CSD %d, strategy bgj4, #threads %ld, #swc chunks <= %ld(%.2f TB), alpha1 %.3f, batch1 %d, alpha2 %.3f, batch2 %d * %d, alpha3 %.3f, batch3 %d",
                _pool->CSD, _num_threads, _num_sol_chunks_slimit, _num_sol_chunks_slimit * exp_chunk_nbytes / 1e12, _alpha1, _batch1, _alpha2, _batch2, bgj4_repeat, _alpha3, _batch3);
    }


    return ret;
}

int Reducer_t::run() {
    int ret = 0;

    if (_red_buf) delete _red_buf;
    _red_buf = new red_buffer_holder_t(this);

    for (int tid = 0; tid < _num_threads; tid++) {
        _red_pool[tid]->push([this, tid] { _red_buf->device_init(tid); });
    }
    for (int tid = 0; tid < _num_threads; tid++) {
        _red_pool[tid]->wait_sleep();
    }

    for (int tid = 0; tid < _num_threads; tid++) {
        for (int sid = 0; sid < _threads_per_buc; sid++) {
            _sub_threads[tid * _threads_per_buc + sid]->push([this, tid, sid] { _red_buf->device_init(tid, sid); });
        }
    }
    for (int tid = 0; tid < _num_threads; tid++) {
        for (int sid = 0; sid < _threads_per_buc; sid++) {
            _sub_threads[tid * _threads_per_buc + sid]->wait_sleep();
        }
    }

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
        _red_pool[tid]->push([this, tid] { _reduce(tid); });
    }
    for (int tid = 0; tid < _num_threads; tid++) {
        _red_pool[tid]->wait_sleep();
    }

    lg_dbg("reduce done, waiting for ut checker");

    _ut_checker->input_done();

    _ut_checker->wait_work();

    lg_dbg("ut checker done");

    _signal_red_done();

    for (int tid = 0; tid < _num_threads; tid++) {
        for (int sid = 0; sid < _threads_per_buc; sid++) {
            _sub_threads[tid * _threads_per_buc + sid]->push([this, tid, sid] { _red_buf->device_done(tid, sid); });
        }
    }
    for (int tid = 0; tid < _num_threads; tid++) {
        for (int sid = 0; sid < _threads_per_buc; sid++) {
            _sub_threads[tid * _threads_per_buc + sid]->wait_sleep();
        }
    }

    for (int tid = 0; tid < _num_threads; tid++) {
        _red_pool[tid]->push([this, tid] { _red_buf->device_done(tid); });
    }
    for (int tid = 0; tid < _num_threads; tid++) {
        _red_pool[tid]->wait_sleep();
    }

    lg_report();

    delete _red_buf;
    _red_buf = NULL;

    return ret;
}

int Reducer_t::_reduce(int tid) {
    constexpr long vec_nbytes = Pool_hd_t::vec_nbytes;

    for (;;) {
        long bucket_id = -1;
        std::unique_lock<std::mutex> red_lock(_red_mtx);
        _red_cv.wait(red_lock, [this, &bucket_id] {
            if (flag & flag_stop_now) return true; 
            bucket_id = _bwc->pop_bucket();
            return bucket_id >= 0 || (flag & flag_stop);
        });
        red_lock.unlock();

        if ((flag & (flag_stop | flag_stop_now)) && bucket_id == -1) break;

        _signal_bucket_done();

        if (_swc->num_using() >= _num_sol_chunks_slimit) {
            _bwc->bucket_finalize(bucket_id);
            _signal_bucket_done();
            continue;
        }

        if (_strategy == strategy_bgj1) {
            _ld_sbuc(tid, bucket_id);

            _red_buf->bgj1_run(tid);
            
            _red_out_2_swc(tid);
        }

        if (_strategy == strategy_bgj2) {
            _red_buf->bgj2_ctr(tid, &_bucketer->ctr_record[bucket_id * vec_nbytes]);

            _ld_sbuc(tid, bucket_id);

            _red_buf->bgj2_run(tid);

            _red_out_2_swc(tid);
        }

        if (_strategy == strategy_bgj3) {
            _red_buf->bgj3_ctr(tid, &_bucketer->ctr_record[bucket_id * vec_nbytes]);

            _ld_sbuc(tid, bucket_id);

            for (int sid = 0; sid < _threads_per_buc; sid++) {
                _sub_threads[tid * _threads_per_buc + sid]->push([this, tid, sid] {
                    int count = 0;
                    int curr_num_out = 0;
                    int id = tid * _threads_per_buc + sid;
                    #if USE_GRAPH
                    for (int i = 0; i < _batch2; i++) {
                        CHECK_CUDA_ERR(cudaGraphExecKernelNodeSetParams(_red_buf->graphExecs[id], 
                                _red_buf->redKernelNodes[id][i], &_red_buf->redKernelParams[id][i]));
                    }
                    #endif
                    for (int b = sid; b < _batch1; b += _threads_per_buc) {
                        for (int i = 0; i < bgj3_repeat; i++) {
                            _red_buf->bgj3_run(tid, sid, b);
                            if (i != bgj3_repeat - 1) CHECK_CUDA_ERR(cudaStreamSynchronize(_red_buf->sstreams[tid * _threads_per_buc + sid]));
                        }

                        count++;
                        CHECK_CUDA_ERR(cudaMemcpyAsync(&curr_num_out, _red_buf->d_num_red_out[tid * _threads_per_buc + sid], sizeof(int), 
                                       cudaMemcpyDeviceToHost, _red_buf->sstreams[tid * _threads_per_buc + sid]));
                        #if ENABLE_PROFILING
                        int bk2_size[BGJ3_DEFAULT_BATCH2];
                        CHECK_CUDA_ERR(cudaMemcpyAsync(bk2_size, _red_buf->d_bk2[tid * _threads_per_buc + sid], BGJ3_DEFAULT_BATCH2 * sizeof(int), 
                                       cudaMemcpyDeviceToHost, _red_buf->sstreams[tid * _threads_per_buc + sid]));
                        #endif
                        CHECK_CUDA_ERR(cudaStreamSynchronize(_red_buf->sstreams[tid * _threads_per_buc + sid]));
                        #if ENABLE_PROFILING
                        {   
                            int id = tid * _threads_per_buc + sid;
                            float bk2_tt, red_tt;
                            CHECK_CUDA_ERR(cudaEventElapsedTime(&bk2_tt, logger->bk2_start[id], logger->bk2_stop[id]));
                            CHECK_CUDA_ERR(cudaEventElapsedTime(&red_tt, logger->bk2_stop[id], logger->red_stop[id]));
                            logger->ev_bk2_us += 1000.f * bk2_tt * bgj3_repeat;
                            logger->ev_red_us += 1000.f * red_tt * bgj3_repeat;
                            logger->ev_bk2_num += _batch2 * bgj3_repeat;
                            for (int i = 0; i < _batch2; i++) {
                                logger->ev_bk2_max   = std::max((int)logger->ev_bk2_max.load(), bk2_size[i]);
                                logger->ev_bk2_ssum += bk2_size[i] * bgj3_repeat;
                                logger->ev_red_vmmas += ceil(bk2_size[i] / 16.0) * ceil(bk2_size[i] / 16.0 + 1.0) * 0.5 * bgj3_repeat;
                            }
                        }
                        #endif
                        //if (curr_num_out > _red_buf->out_max_size) {
                        //    lg_warn("thread %d | %d, #bk2 %d, #out %d overflow(%d)", tid, sid, count, curr_num_out, _red_buf->out_max_size);
                        //}
                        if ((1 + 2.0 / sqrt(count)) * curr_num_out > _red_buf->out_max_size || b + _threads_per_buc >= _batch1) {
                            _red_out_2_swc(tid, sid);
                            count = 0;
                            curr_num_out = 0;
                        }
                    }
                });
            }

            for (int sid = 0; sid < _threads_per_buc; sid++) {
                _sub_threads[tid * _threads_per_buc + sid]->wait_sleep();
            }
        }

        if (_strategy == strategy_bgj3l || _strategy == strategy_bgj4) {
            _red_buf->bgjl_buc_ctr(tid, &_bucketer->ctr_record[bucket_id * vec_nbytes]);

            int num_chunks;
            chunk_t **working_chunks;

            for (;;) {
                pthread_spin_lock(&traffic_ctrl_lock);
                if (ld_bk0_tids < 2) {
                    ld_bk0_tids++;
                    pthread_spin_unlock(&traffic_ctrl_lock);
                    break;
                }
                pthread_spin_unlock(&traffic_ctrl_lock);
            }

            _ld_lbuc(tid, bucket_id, num_chunks, working_chunks);

            pthread_spin_lock(&traffic_ctrl_lock);
            ld_bk0_tids--;
            pthread_spin_unlock(&traffic_ctrl_lock);

            _red_lbuc(tid, working_chunks);

            for (int i = 0; i < num_chunks; i++) _bwc->read_done(working_chunks[i], bucket_id);

            free(working_chunks);
        }

        _bwc->bucket_finalize(bucket_id);

        _signal_bucket_done();

        pthread_spin_lock(&stuck_stat_lock);
        int need_signal_stuck = traits::sieving_stuck(total_check, total_notin, _pool->CSD);            
        pthread_spin_unlock(&stuck_stat_lock);
        if (need_signal_stuck) _signal_red_stuck();
    }

    return 0;
}

int Reducer_t::_red_out_2_swc(int tid, int sid) {
    int32_t size;
    int8_t *h_vec;
    int32_t *h_norm;
    uint16_t *h_score;
    uint64_t *h_u;

    if (_strategy == strategy_bgj1 || _strategy == strategy_bgj2) {
        _red_buf->bgjs_out(tid, &size, &h_vec, &h_norm, &h_score, &h_u);
    }
    if (_strategy == strategy_bgj3) {
        _red_buf->bgjm_out(tid, sid, &size, &h_vec, &h_norm, &h_score, &h_u);
    }
    if (_strategy == strategy_bgj3l || _strategy == strategy_bgj4) {
        _red_buf->bgjl_out(tid, sid, &size, &h_vec, &h_norm, &h_score, &h_u);
    }
    

    chunk_t *dst = NULL;

    while (size) {
        if (!dst) {
            if (_swc->num_using() >= _num_sol_chunks_slimit) {
                _ut_checker->trigger_batch();
                /// lg_warn("swc full, %d new sols ignored", size);
                break;
            }
            #if ENABLE_PROFILING
            struct timeval fetch_start, fetch_end;
            gettimeofday(&fetch_start, NULL);
            #endif
            dst = _swc->fetch_for_write();
            #if ENABLE_PROFILING
            gettimeofday(&fetch_end, NULL);
            logger->ev_ld_stall_us += (fetch_end.tv_sec - fetch_start.tv_sec) * 1000000 + fetch_end.tv_usec - fetch_start.tv_usec;
            #endif
            if (!dst) {
                lg_err("tid = %d, fetch_for_write failed, swc %d using, limit %d, %d new sols ignored",
                        tid, _swc->num_using(), _num_sol_chunks_slimit, size);
                break;
            } else if (_normalize_chunk(dst, _pool->CSD)) {
                lg_warn("chunk %d from swc not normalized", dst->id);
            }
        }
        int to_move = size < Pool_hd_t::chunk_max_nvecs - dst->size ? size : Pool_hd_t::chunk_max_nvecs - dst->size;
        memcpy(dst->vec + _pool->CSD * dst->size, h_vec, _pool->CSD * to_move);
        memcpy(dst->norm + dst->size, h_norm, sizeof(int32_t) * to_move);
        memcpy(dst->score + dst->size, h_score, sizeof(uint16_t) * to_move);
        memcpy(dst->u + dst->size, h_u, sizeof(uint64_t) * to_move);
        size      -= to_move;
        dst->size += to_move;
        h_vec     += to_move * _pool->CSD;
        h_norm    += to_move;
        h_score   += to_move;
        h_u       += to_move;
        if (dst->size == Pool_hd_t::chunk_max_nvecs) {
            _ut_checker->task_commit(dst);
            #if ENABLE_PROFILING
            logger->ev_ld_chunks++;
            logger->ev_st_chunks++;
            #endif
            dst = NULL;
        }
    }

    if (dst) {
        #if ENABLE_PROFILING
        logger->ev_ld_chunks++;
        logger->ev_st_chunks++;
        #endif
        _ut_checker->task_commit(dst);
    }

    return 0;
}

int Reducer_t::_ld_sbuc(int tid, int bucket_id) {
    #if ENABLE_PROFILING
    struct timeval fetch_start, fetch_end;
    gettimeofday(&fetch_start, NULL);
    #endif
    chunk_t *curr_chunk = _bwc->fetch_for_read(bucket_id);
    #if ENABLE_PROFILING
    gettimeofday(&fetch_end, NULL);
    logger->ev_ld_stall_us += (fetch_end.tv_sec - fetch_start.tv_sec) * 1000000 + fetch_end.tv_usec - fetch_start.tv_usec;
    logger->ev_ld_chunks++;
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
            task_vecs += _red_buf->bgjs_h2d(tid, curr_chunk, used);
            if (used == -1) {
                // lg_warn("bucket %d size exceeds buc_max_size(%d), truncated", bucket_id, _red_buf->buc_max_size);
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
                logger->ev_ld_chunks++;
                #endif
                used = 0;
            }
        }
        
        if (_strategy == strategy_bgj1) {
            _red_buf->bgjs_upk(tid);
        }
        if (_strategy == strategy_bgj2 || _strategy == strategy_bgj3) {
            _red_buf->bgjm_upk(tid);
        }

        if (no_more_chunks_in_buc) break;
    }

    #if ENABLE_PROFILING
    int bk0_size = _red_buf->buc_vecs[tid];
    int bk1_size[BGJ3_DEFAULT_BATCH1 + BGJ2_DEFAULT_BATCH1];
    if (_strategy == strategy_bgj2 || _strategy == strategy_bgj3) {
        CHECK_CUDA_ERR(cudaMemcpyAsync(bk1_size, _red_buf->d_bk1[tid], _batch1 * sizeof(int), 
                       cudaMemcpyDeviceToHost, _red_buf->streams[tid]));
        CHECK_CUDA_ERR(cudaStreamSynchronize(_red_buf->streams[tid]));
        logger->ev_bk1_num += _batch1;
        for (int i = 0; i < _batch1; i++) {
            logger->ev_bk1_ssum += bk1_size[i];
            logger->ev_bk1_max   = std::max((int)logger->ev_bk1_max.load(), bk1_size[i]);
            if (_strategy == strategy_bgj3) {
                logger->ev_bk2_vmmas += ceil(_batch2 / 16.0) * ceil(bk1_size[i] / 16.0);
            }
            if (_strategy == strategy_bgj2) {
                logger->ev_red_vmmas += ceil(bk1_size[i] / 16.0) * ceil(bk1_size[i] / 16.0 + 1.0) / 2;
            }
        }
    }
    logger->ev_bk0_num  += 1;
    logger->ev_bk0_ssum += bk0_size;
    logger->ev_bk0_max   = std::max((int)logger->ev_bk0_max.load(), bk0_size);
    logger->ev_bk1_vmmas += ceil(bk0_size / 16.0) * ceil(_batch1 / 16.0);
    #endif

    return 0;
}

int Reducer_t::_ld_lbuc(int tid, int bucket_id, int &num_chunks, chunk_t **&working_chunks) {
    num_chunks = 0;
    long bucket_num_chunks = _bwc->bucket_num_chunks(bucket_id);
    if (bucket_num_chunks > _red_buf->buc_max_size / 8192) bucket_num_chunks = _red_buf->buc_max_size / 8192 + 1;

    working_chunks = (chunk_t **) malloc(bucket_num_chunks * sizeof(chunk_t *));

    pthread_spinlock_t task_lock, buc_lock;
    pthread_spin_init(&task_lock, PTHREAD_PROCESS_SHARED);
    pthread_spin_init(&buc_lock, PTHREAD_PROCESS_SHARED);
    
    int num_task = 0, no_more_task = 0;
    uint64_t *task_list = (uint64_t *) malloc(bucket_num_chunks + 8);

    for (int sid = 0; sid < _threads_per_buc; sid++) {
        _sub_threads[tid * _threads_per_buc + sid]->push([&, tid, sid] {
            int id = tid * _threads_per_buc + sid;
            CHECK_CUDA_ERR(cudaMemsetAsync(_red_buf->d_num_red_out[id], 0, sizeof(int), _red_buf->sstreams[id]));
            CHECK_CUDA_ERR(cudaMemsetAsync(_red_buf->d_num_flt_out[id], 0, sizeof(int), _red_buf->sstreams[id]));
            CHECK_CUDA_ERR(cudaMemcpyAsync(_red_buf->d_ct1[id], _red_buf->h_ct1[tid], _batch1 * _red_buf->CSD16, cudaMemcpyHostToDevice, _red_buf->sstreams[id]));
            int *pos = (int *) malloc(sizeof(int) * _batch1);
            int *num = (int *) malloc(sizeof(int) * _batch1);
            for (;;) {
                uint64_t task = 0;
                for (int64_t poll_iters = 0;; poll_iters++) {
                    if (num_task || no_more_task) {
                        volatile int *num_task_vol = reinterpret_cast<volatile int *>(&num_task);
                        pthread_spin_lock(&task_lock);
                        if (num_task_vol[0]) {
                            task = task_list[--num_task_vol[0]];
                            pthread_spin_unlock(&task_lock);
                            break;
                        }
                        pthread_spin_unlock(&task_lock);
                        if (no_more_task) break;
                    }
                }

                if (task == 0 && no_more_task) break;
                int task_chunks = task >> 32;
                int task_start  = task & 0xffffffff;
                for (int i = 0; i < task_chunks; i++) {
                    chunk_t *src = working_chunks[task_start + i];
                    _red_buf->bgjl_buc_h2d(tid, sid, src);
                }
                int *buc_out;
                _red_buf->bgjl_buc_run(tid, sid);
                _red_buf->bgjl_buc_out(tid, sid, &buc_out);

                uint64_t ind_bias = (uint64_t) task_start * Pool_hd_t::chunk_max_nvecs;
                uint32_t ind_bias_hi = (ind_bias >> 32) << 24;
                uint32_t ind_bias_lo = ind_bias & 0xffffffffULL;

                /// @todo shuffle the order && multi lock
                int *buc = _red_buf->h_bk1[tid];
                pthread_spin_lock(&buc_lock);
                for (int i = 0; i < _batch1; i++) {
                    int max_num = _red_buf->bk1_max_size > buc[i] ? (_red_buf->bk1_max_size - buc[i]) : 0;
                    num[i] = buc_out[i] > max_num ? max_num : buc_out[i];
                    pos[i] = buc[i];
                    buc[i] += num[i];
                }
                pthread_spin_unlock(&buc_lock);
                for (int i = 0; i < _batch1; i++) {
                    uint32_t *dst = (uint32_t *)buc + _batch1 + 2L * (_red_buf->bk1_max_size * (long)i + (long)pos[i]);
                    uint32_t *src = (uint32_t *)buc_out + _batch1 + 2L * (_red_buf->l1_out_max_size * (long)i);
                    for (int j = 0; j < num[i]; j++) {
                        dst[2 * j + 0] = ind_bias_lo + src[2 * j + 0];
                        dst[2 * j + 1] = ind_bias_hi + src[2 * j + 1];
                    }
                }
            }
            free(pos);
            free(num);
        });
    }
    

    chunk_t *unfull_chunk = NULL;
    for (;;) {
        if (num_chunks == bucket_num_chunks) break;
        #if ENABLE_PROFILING
        struct timeval fetch_start, fetch_end;
        gettimeofday(&fetch_start, NULL);
        #endif
        chunk_t *src = _bwc->fetch_for_read(bucket_id);
        #if ENABLE_PROFILING
        gettimeofday(&fetch_end, NULL);
        logger->ev_ld_stall_us += (fetch_end.tv_sec - fetch_start.tv_sec) * 1000000 + fetch_end.tv_usec - fetch_start.tv_usec;
        logger->ev_ld_chunks++;
        #endif
        if (!src) break;
        if (_normalize_chunk(src, _pool->CSD)) {
            lg_warn("chunk %d from bwc (bucket_id %d) not normalized", src->id, bucket_id);
        }
        if (src->size == 0) {
            lg_warn("chunk %d from bwc (bucket_id %d) has size 0", src->id, bucket_id);
            _bwc->read_done(src, bucket_id);
            continue;
        }
        if (src->size < Pool_hd_t::chunk_max_nvecs) {
            if (!unfull_chunk) unfull_chunk = src;
            else {
                int to_copy = Pool_hd_t::chunk_max_nvecs - unfull_chunk->size < src->size ? 
                              Pool_hd_t::chunk_max_nvecs - unfull_chunk->size : src->size;
                memcpy(unfull_chunk->vec + _pool->CSD * unfull_chunk->size, src->vec + _pool->CSD * (src->size - to_copy), _pool->CSD * to_copy);
                memcpy(unfull_chunk->norm + unfull_chunk->size, src->norm + src->size - to_copy, sizeof(int32_t) * to_copy);
                memcpy(unfull_chunk->score + unfull_chunk->size, src->score + src->size - to_copy, sizeof(uint16_t) * to_copy);
                memcpy(unfull_chunk->u + unfull_chunk->size, src->u + src->size - to_copy, sizeof(uint64_t) * to_copy);
                memset(src->score + src->size - to_copy, 0, sizeof(uint16_t) * to_copy);
                memset(src->norm + src->size - to_copy, 0, sizeof(int32_t) * to_copy);
                src->size -= to_copy;
                unfull_chunk->size += to_copy;
                if (unfull_chunk->size == Pool_hd_t::chunk_max_nvecs) {
                    working_chunks[num_chunks++] = unfull_chunk;
                    if (src->size) unfull_chunk = src;
                    else {
                        _bwc->read_done(src, bucket_id);
                        unfull_chunk = NULL;
                    }
                } else {
                    _bwc->read_done(src, bucket_id);
                    continue;
                }
            }
        } else working_chunks[num_chunks++] = src;

        if (num_chunks % traits::taskChunks == 0 && num_chunks) {
            uint64_t task = (((uint64_t) traits::taskChunks) << 32) | (num_chunks - traits::taskChunks);
            pthread_spin_lock(&task_lock);
            task_list[num_task++] = task;
            pthread_spin_unlock(&task_lock);
        }
    }

    uint64_t not_added = num_chunks % traits::taskChunks;
    not_added = (not_added << 32) | (num_chunks - not_added);
    if (unfull_chunk) {
        if (unfull_chunk->size && num_chunks != bucket_num_chunks) {
            working_chunks[num_chunks++] = unfull_chunk;
            not_added += 1ULL << 32;
        } else _bwc->read_done(unfull_chunk, bucket_id);
    }

    pthread_spin_lock(&task_lock);
    if (not_added >> 32) task_list[num_task++] = not_added;
    no_more_task = 1;
    pthread_spin_unlock(&task_lock);

    for (int sid = 0; sid < _threads_per_buc; sid++) {
        _sub_threads[tid * _threads_per_buc + sid]->wait_sleep();
    }

    free(task_list);
    pthread_spin_destroy(&task_lock);
    pthread_spin_destroy(&buc_lock);

    #if ENABLE_PROFILING
    long bk0_size = (num_chunks - 1) * Pool_hd_t::chunk_max_nvecs + working_chunks[num_chunks - 1]->size;
    int *bk1_size = _red_buf->h_bk1[tid];
    logger->ev_bk1_num  += _batch1;
    for (int i = 0; i < _batch1; i++) {
        logger->ev_bk1_ssum += bk1_size[i];
        logger->ev_bk1_max   = std::max((int)logger->ev_bk1_max.load(), bk1_size[i]);
    }
    logger->ev_bk0_num  += 1;
    logger->ev_bk0_ssum += bk0_size;
    logger->ev_bk0_max   = std::max((long)logger->ev_bk0_max.load(), bk0_size);
    logger->ev_bk1_vmmas += ceil(bk0_size / 16.0) * ceil(_batch1 / 16.0);
    #endif

    return 0;
}

void vec_collect(int8_t *dst, int8_t **src_list, int n, int CSD, int CSD16);

int Reducer_t::_red_lbuc(int tid, chunk_t **&working_chunks) {
    int num_task = _batch1;

    pthread_spinlock_t task_lock;
    pthread_spin_init(&task_lock, PTHREAD_PROCESS_SHARED);

    for (int sid = 0; sid < _threads_per_buc; sid++) {
        _sub_threads[tid * _threads_per_buc + sid]->push([&, tid, sid] {
            int id = tid * _threads_per_buc + sid;
            int count = 0;
            #if USE_GRAPH
            for (int i = 0; i < (_strategy == strategy_bgj3l ? _batch2 : _batch3); i++) {
                CHECK_CUDA_ERR(cudaGraphExecKernelNodeSetParams(_red_buf->graphExecs[id], 
                        _red_buf->redKernelNodes[id][i], &_red_buf->redKernelParams[id][i]));
            }
            #endif

            for (;;) {
                pthread_spin_lock(&task_lock);
                int task = --num_task;
                pthread_spin_unlock(&task_lock);
                if (task < 0) break;

                int bk1_size = _red_buf->h_bk1[tid][task];
                int *bk1_ptr = _red_buf->h_bk1[tid] + _batch1 + 2L * _red_buf->bk1_max_size * (long)task;
                int32_t *hn_ptr = _red_buf->h_norm[tid * _threads_per_buc + sid];
                int8_t *hv_ptr = _red_buf->h_vec_out[tid * _threads_per_buc + sid];
                _red_buf->bgjl_ctr(tid, sid, bk1_size, bk1_ptr, working_chunks);
                #if ENABLE_PROFILING
                struct timeval collect_start, collect_end;
                gettimeofday(&collect_start, NULL);
                #endif
                for (int i = 0; i < bk1_size; i += traits::taskVecs) {
                    if (i) CHECK_CUDA_ERR(cudaStreamSynchronize(_red_buf->sstreams[id]));
                    int to_move = bk1_size - i < traits::taskVecs ? bk1_size - i : traits::taskVecs;
                    int8_t *src_list[traits::taskVecs];
                    for (int j = i; j < i + to_move; j++) {
                        uint32_t ptr = bk1_ptr[j * 2];
                        uint32_t ptr_hi = (((uint32_t)bk1_ptr[j * 2 + 1]) >> 24) << 19;
                        src_list[j - i] = working_chunks[ptr / Pool_hd_t::chunk_max_nvecs + ptr_hi]->vec + 
                                       _pool->CSD * (ptr % Pool_hd_t::chunk_max_nvecs);
                        _red_buf->h_repeat_buf[id][2 * (j - i)] = j;
                        #if BGJL_HOST_UPK
                        _red_buf->h_repeat_buf[id][2 * (j - i) + 1] = bk1_ptr[j * 2 + 1] & 0x00ffffff;
                        #else
                        hn_ptr[j - i] = bk1_ptr[j * 2 + 1] & 0x00ffffff;
                        _red_buf->h_repeat_buf[id][2 * (j - i) + 1] = hn_ptr[j - i];
                        #endif
                    }
                    vec_collect(hv_ptr, src_list, to_move, _pool->CSD, _red_buf->CSD16);
                    _red_buf->bgjl_h2d(tid, sid, to_move);
                    _red_buf->bgjl_upk(tid, sid, i);
                    CHECK_CUDA_ERR(cudaMemcpyAsync(_red_buf->repeat_buf[id] + 2 * i, _red_buf->h_repeat_buf[id], 8 * to_move, 
                                   cudaMemcpyHostToDevice, _red_buf->sstreams[id]));
                }
                CHECK_CUDA_ERR(cudaMemcpyAsync(_red_buf->repeat_buf_size[id], &bk1_size, sizeof(int), 
                               cudaMemcpyHostToDevice, _red_buf->sstreams[id]));
                #if ENABLE_PROFILING
                {   
                    long rep = _strategy == strategy_bgj3l ? bgj3l_repeat : bgj4_repeat;
                    gettimeofday(&collect_end, NULL);
                    logger->ev_collect_us += (collect_end.tv_sec - collect_start.tv_sec) * 1000000 + 
                                              collect_end.tv_usec - collect_start.tv_usec;
                    logger->ev_bk2_num += rep * _batch2;
                    logger->ev_bk2_vmmas += rep * ceil(bk1_size / 16.0) * (_batch2 / 16.0);
                }
                #endif
                if (_strategy == strategy_bgj3l) _red_buf->bgj3l_run(tid, sid);
                if (_strategy == strategy_bgj4) _red_buf->bgj4_run(tid, sid);
                
                count++;
                CHECK_CUDA_ERR(cudaMemcpyAsync(_red_buf->h_num_flt_out[id], _red_buf->d_num_flt_out[id], sizeof(int), 
                                cudaMemcpyDeviceToHost, _red_buf->sstreams[id]));
                CHECK_CUDA_ERR(cudaStreamSynchronize(_red_buf->sstreams[id]));
                #if ENABLE_PROFILING
                {
                    float fff_tt;
                    CHECK_CUDA_ERR(cudaEventElapsedTime(&fff_tt, logger->fff_start[id], logger->fff_stop[id]));
                    logger->ev_fff_us += 1000.f * fff_tt;
                }
                #endif
                //if (_red_buf->h_num_flt_out[id][0] > _red_buf->flt_out_max_size) {
                //    lg_warn("thread %d | %d, #bk2 %d, #out %d overflow(%d)", tid, sid, count, _red_buf->h_num_flt_out[id][0], _red_buf->flt_out_max_size);
                //}
                if ((1 + 2.0 / sqrt(count)) * _red_buf->h_num_flt_out[id][0] > _red_buf->flt_out_max_size) {
                    _red_out_2_swc(tid, sid);
                    count = 0;
                }
            }

            if (count) {
                CHECK_CUDA_ERR(cudaMemcpyAsync(_red_buf->h_num_flt_out[id], _red_buf->d_num_flt_out[id], sizeof(int), 
                                cudaMemcpyDeviceToHost, _red_buf->sstreams[id]));
                CHECK_CUDA_ERR(cudaStreamSynchronize(_red_buf->sstreams[id]));
                if (_red_buf->h_num_flt_out[id][0] > _red_buf->flt_out_max_size) {
                    lg_warn("thread %d | %d, #bk2 %d, #out %d overflow(%d)", tid, sid, count, _red_buf->h_num_flt_out[id][0], _red_buf->flt_out_max_size);
                }
                if (_red_buf->h_num_flt_out[id][0]) _red_out_2_swc(tid, sid);
            }
        });
    }

    for (int sid = 0; sid < _threads_per_buc; sid++) {
        _sub_threads[tid * _threads_per_buc + sid]->wait_sleep();
    }

    pthread_spin_destroy(&task_lock);

    return 0;
}

int Reducer_t::_signal_bucket_done() {
    std::unique_lock<std::mutex> lock(_bucketer->_buc_mtx);
    _bucketer->_buc_cv.notify_all();
    return 0;
}

int Reducer_t::_signal_red_done() {
    std::unique_lock<std::mutex> lock(_bucketer->_buc_mtx);
    _bucketer->flag |= Bucketer_t::flag_final;
    _bucketer->_buc_cv.notify_all();
    return 0;
}

int Reducer_t::_signal_red_stuck() {
    std::unique_lock<std::mutex> lock(_bucketer->_buc_mtx);
    _bucketer->flag |= Bucketer_t::flag_stuck;
    _bucketer->_buc_cv.notify_all();
    return 0;
}
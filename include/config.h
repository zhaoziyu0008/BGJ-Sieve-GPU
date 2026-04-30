#ifndef __CONFIG_H
#define __CONFIG_H

///////////////// hardware config /////////////////
#define MULTI_SSD 1
#define MULTI_GPU 1
struct hw {
    static constexpr int ssd_num = 2;
    static constexpr const char *ssd_name_list[ssd_num] = {"0", "1"};
    static inline const char *ssd_name(int chunk_id) {
        return (chunk_id & 1) ? "1" : "0";
    }
    static constexpr int gpu_num = 8;
    static constexpr const int gpu_id_list[gpu_num] = {0, 1, 2, 3, 4, 5, 6, 7};
    static inline int gpu_ptr(int tid, int num_threads) {
        return (tid * gpu_num) / num_threads;
    }
    static inline int gpu_ptrl(int tid, int sid, int num_threads, int threads_per_buc) {
        if (sid == -1) return gpu_ptr(tid, num_threads);
        else return (tid * threads_per_buc + sid) % gpu_num;
    }
    static inline int gpu_id(int tid, int num_threads) {
        return (tid * gpu_num) / num_threads;
    }
};

///////////////// profiling config /////////////////

#define ENABLE_PROFILING         1
#define AUTO_REPORT_DURATION  1800
#define POOL_HD_LOG_LEVEL   ll_info
#define REDUCER_LOG_LEVEL   ll_dbg
#define BUCKETER_LOG_LEVEL  ll_dbg
#define PWC_LOG_LEVEL       ll_warn
#define BWC_LOG_LEVEL       ll_warn
#define SWC_LOG_LEVEL       ll_warn
#define DHB_LOG_LEVEL       ll_dbg
#define DHR_LOG_LEVEL       ll_dbg


///////////////// cuda config /////////////////
#define MAX_NUM_DEVICE 8
#define USE_GRAPH 0
#define USE_HUGE_PAGE 0
#define BGJL_HOST_UPK 1


///////////////// pwc config /////////////////
#define ONE_TIME_IO                     1
#define PWC_DEFAULT_LOADING_THREADS     6
#define PWC_DEFAULT_SYNCING_THREADS     6
#define PWC_SSD_SLIMIT                  (10000ULL << 30)
#define PWC_DRAM_SLIMIT                 (8ULL << 30)
#define PWC_DEFAULT_MAX_CACHED_CHUNKS   (PWC_DRAM_SLIMIT / 8192ULL / 190ULL)
#define PWC_MAX_PARALLEL_SYNC_CHUNKS    5


///////////////// bwc config /////////////////
#define BWC_DEFAULT_LOADING_THREADS     8
#define BWC_DEFAULT_SYNCING_THREADS     6
#define BWC_SSD_SLIMIT                  (15000ULL << 30)
#define BWC_DRAM_SLIMIT                 (262ULL << 30)
#define BWC_DEFAULT_MAX_CACHED_CHUNKS   (BWC_DRAM_SLIMIT / 8192ULL / 190ULL)
#define BWC_MAX_PARALLEL_SYNC_CHUNKS    5
#define BWC_MAX_BUCKETS                 4192


///////////////// swc config /////////////////
#define SWC_DEFAULT_LOADING_THREADS     5
#define SWC_DEFAULT_SYNCING_THREADS     3
#define SWC_SSD_SLIMIT                  (5000ULL << 30)
#define SWC_DRAM_SLIMIT                 (36ULL << 30)
#define SWC_DEFAULT_MAX_CACHED_CHUNKS   (SWC_DRAM_SLIMIT / 8192ULL / 190ULL)
#define SWC_MAX_PARALLEL_SYNC_CHUNKS    5


///////////////// red config /////////////////
#define RED_MIN_CSD16                   128     /* change with kernel choosing code tegother */
#define RED_MAX_NUM_THREADS             32             
#define RED_GRAM_SLIMIT                 (22ULL << 30)

#define BGJ1_RED_DEFAULT_NUM_THREADS    32
#define BGJ2_RED_DEFAULT_NUM_THREADS    32
#define BGJ3_RED_DEFAULT_NUM_THREADS    16
#define BGJ3L_RED_DEFAULT_NUM_THREADS   6
#define BGJ4_RED_DEFAULT_NUM_THREADS    4

#define BGJ2_DEFAULT_ALPHA1             0.290               
#define BGJ2_DEFAULT_BATCH1             4096

#define BGJ3_DEFAULT_ALPHA1             0.190
#define BGJ3_DEFAULT_ALPHA2             0.295
#define BGJ3_DEFAULT_BATCH1             256
#define BGJ3_DEFAULT_BATCH2             512
#define BGJ3_DEFAULT_THREADS_PER_BUC    4

#define BGJ3L_DEFAULT_ALPHA1            0.190
#define BGJ3L_DEFAULT_ALPHA2            0.295
#define BGJ3L_DEFAULT_BATCH1            256
#define BGJ3L_DEFAULT_BATCH2            512
#define BGJ3L_DEFAULT_THREADS_PER_BUC   8

#define BGJ4_DEFAULT_ALPHA1             0.180
#define BGJ4_DEFAULT_ALPHA2             0.230
#define BGJ4_DEFAULT_ALPHA3             0.300
#define BGJ4_DEFAULT_BATCH1             128
#define BGJ4_DEFAULT_BATCH2             64
#define BGJ4_DEFAULT_BATCH3             16
#define BGJ4_DEFAULT_THREADS_PER_BUC    8


///////////////// buc config /////////////////
#define BUC_MIN_CSD16                   128     /* change with kernel choosing code tegother */
#define BUC_DEFAULT_NUM_THREADS         16
#define BUC_GRAM_SLIMIT                 (1ULL << 30)

#define BGJ1_L0_MIN_ALPHA0              0.310
#define BGJ1_L0_MAX_ALPHA0              0.310
#define BGJ2_L0_MIN_ALPHA0              0.245
#define BGJ2_L0_MAX_ALPHA0              0.245
#define BGJ3_L0_MIN_ALPHA0              0.210
#define BGJ3_L0_MAX_ALPHA0              0.210
#define BGJ3L_L0_MIN_ALPHA0             0.210
#define BGJ3L_L0_MAX_ALPHA0             0.210
#define BGJ4_L0_MIN_ALPHA0              0.165
#define BGJ4_L0_MAX_ALPHA0              0.185

#define BGJ1_L0_BATCH_RATIO             0.5
#define BGJ2_L0_BATCH_RATIO             0.5
#define BGJ3_L0_BATCH_RATIO             0.5
#define BGJ3L_L0_BATCH_RATIO            0.5
#define BGJ4_L0_BATCH_RATIO             0.5

#define BGJ_L0_MAX_BATCH0               2048    /* change with kernel choosing code tegother */
#define BGJ_L0_MIN_BATCH0               16      /* change with kernel choosing code tegother */



///////////////// bgj config /////////////////
#define BGJ_DEFAULT_SATURATION_RADIUS   4./3.
#define BGJ_DEFAULT_SATURATION_RATIO    0.375
#define BGJ_DEFAULT_IMPROVE_RATIO       0.71
#define BGJ_CENTER_IMPROVE_RATIO        0.77

#define BGJ1_SIZE_RATIO                 3.2
#define BGJ2_SIZE_RATIO                 3.2
#define BGJ3_SIZE_RATIO                 3.2
#define BGJ3L_SIZE_RATIO                3.2
#define BGJ4_SIZE_RATIO                 3.2


///////////////// ut config /////////////////
#define UT_DEFAULT_NUM_THREADS          16
#define UT_TABLE_DRAM_SLIMIT            (1500ULL << 30)
#define UT_BUFFER_DRAM_SLIMIT           (300ULL << 30)
#define UT_DEFAULT_MAX_CHUNKS           (UT_BUFFER_DRAM_SLIMIT / 8192ULL / 190ULL)
#define UT_DEFAULT_MAX_UIDS             (UT_BUFFER_DRAM_SLIMIT / 8192ULL / 32ULL)
#define UT_DEFAULT_BATCH_RATIO          0.01


///////////////// dh config /////////////////
#define DH_MAX_BATCH                    2048
#define DH_MIN_BATCH                    256
#define DHB_DEFAULT_NUM_THREADS         24
#define DHR_DEFAULT_NUM_THREADS         32
#define DHB_GRAM_SLIMIT                 (1ULL << 30)
#define DHR_GRAM_SLIMIT                 (22ULL << 30)
#define DH_BSIZE_RATIO                  (ESD <= 40 ? 30.0 : 50.0)
#define DH_REPORT_DURATION              1800
#define SPLIT_DHR                       1

#endif

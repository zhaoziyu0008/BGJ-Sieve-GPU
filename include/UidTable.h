#ifndef __UIDTABLE_H
#define __UIDTABLE_H

#include "config.h"

#include <pthread.h>
#include <limits>

#include "../dep/phmap/phmap.h"

struct UidTable {
    static constexpr uint64_t ut_split = 8192;
    
    typedef uint64_t                        Utype;
    typedef uint64_t                        Usize;
    typedef pthread_spinlock_t              Ulock;
    typedef phmap::flat_hash_set<uint64_t>  Uset;
    
    UidTable();
    ~UidTable();

    int set_dirname(const char *dirname);
    int set_num_threads(int num_threads);

    inline uint64_t coeff(long i) { return _coeffs[i]; } 
    inline uint64_t normalize(uint64_t uid) { return uid > std::numeric_limits<Utype>::max() / 2  + 1 ? -uid : uid; }
    
    uint64_t size();
    uint64_t hold();

    int insert(uint64_t uid);
    int erase(uint64_t uid);
    int check(uint64_t uid);

    int dump_table(long table_id);
    int load_table(long table_id);
    int reset_hash_function(long CSD);

    #if MULTI_SSD
    char _dir[32] = {};
    #else
    char _prefix[32] = {};
    #endif

    Utype   _coeffs[256];
    Usize   _size[ut_split];
    Ulock   _locks[ut_split];
    Uset    *_tables[ut_split];

    int _num_threads = 1;
};

#endif
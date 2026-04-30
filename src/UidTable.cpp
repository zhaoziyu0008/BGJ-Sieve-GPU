#include "../include/UidTable.h"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>

#include "../dep/phmap/phmap_dump.h"

uint64_t sampleU() {
    uint64_t ret = 0;
    ret |= rand() & 0xffff;
    ret <<= 16;
    ret |= rand() & 0xffff;
    ret <<= 16;
    ret |= rand() & 0xffff;
    ret <<= 16;
    ret |= rand() & 0xffff;
    return ret;
}

UidTable::UidTable() {
    for (long i = 0; i < ut_split; i++) _size[i] = 0;
    for (long i = 0; i < ut_split; i++) pthread_spin_init(&_locks[i], PTHREAD_PROCESS_SHARED);
    for (long i = 0; i < ut_split; i++) _tables[i] = new Uset();
    this->set_dirname(".uid");
}

UidTable::~UidTable() {
    for (long i = 0; i < ut_split; i++) delete _tables[i];
    for (long i = 0; i < ut_split; i++) pthread_spin_destroy(&_locks[i]);
}

uint64_t UidTable::size() {
    uint64_t ret = 0;
    for (long i = 0; i < ut_split; i++) ret += _size[i];
    return ret;
}

uint64_t UidTable::hold() {
    uint64_t ret = 0;
    for (long i = 0; i < ut_split; i++) if (_tables[i]) ret++;
    return ret;
}

int UidTable::set_dirname(const char *dirname) {
    if (dirname == NULL) {
        fprintf(stderr, "[Error] UidTable::set_dirname: NULL input\n");
        return -1;
    }
    #if MULTI_SSD
    int perfix_len = snprintf(_dir, 32, "%s", dirname);
    #else
    int perfix_len = snprintf(_prefix, 32, "%s/.Ufile-", dirname);
    #endif
    if (perfix_len >= 32) {
        fprintf(stderr, "[Error] UidTable::set_dirname: dirname too long\n");
        return -1;
    }

    struct stat st;
    if (stat(dirname, &st) == 0) {
        if (S_ISDIR(st.st_mode)) {
            #if MULTI_SSD
            for (int I = 0; I < hw::ssd_num; I++) {
                char subdirpath[256];
                snprintf(subdirpath, 256, "%s/%s", dirname, hw::ssd_name_list[I]); 
                DIR *dir = opendir(subdirpath);
                if (dir == NULL) {
                    perror("[Error] UidTable::set_dirname: opendir failed");
                    abort();
                }

                struct dirent *entry;
                while ((entry = readdir(dir)) != NULL) {
                    if (strncmp(entry->d_name, ".Ufile-", 7) == 0) {
                        char filepath[512];
                        snprintf(filepath, 512, "%s/%s", subdirpath, entry->d_name);
                        if (remove(filepath) != 0) {
                            perror("[Error] UidTable::set_dirname: remove failed");
                            closedir(dir);
                            abort();
                        }
                    }
                }
                closedir(dir);
            }
            #else
            DIR *dir = opendir(dirname);
            if (dir == NULL) {
                perror("[Error] UidTable::set_dirname: opendir failed");
                return -1;
            }
            struct dirent *entry;
            while ((entry = readdir(dir)) != NULL) {
                if (strncmp(entry->d_name, ".Ufile-", 7) == 0) {
                    char filepath[PATH_MAX];
                    snprintf(filepath, sizeof(filepath), "%s/%s", dirname, entry->d_name);
                    if (remove(filepath) != 0) {
                        perror("[Error] UidTable::set_dirname: remove failed");
                        closedir(dir);
                        return -1;
                    }
                }
            }
            closedir(dir);
            #endif
        } else {
            fprintf(stderr, "[Error] UidTable::set_dirname: %s is not a directory\n", dirname);
            #if MULTI_SSD
            abort();
            #else
            return -1;
            #endif
        }
    } else {
        #if MULTI_SSD
        fprintf(stderr, "[Error] UidTable::set_dirname: directory %s does not exist\n", dirname);
        abort();
        #else
        if (mkdir(dirname, 0755) != 0) {
            perror("[Error] UidTable::set_dirname: mkdir failed");
            return -1;
        }
        #endif
    }

    return 0;
}

int UidTable::set_num_threads(int num_threads) {
    if (num_threads <= 0) {
        fprintf(stderr, "[Error] UidTable::set_num_threads: invalid input\n");
        return -1;
    }
    _num_threads = num_threads;
    return 0;
}

int UidTable::insert(uint64_t uid) {
    uid = this->normalize(uid);
    int pos = uid % ut_split;
    if (_tables[pos] == NULL) return -1;
    pthread_spin_lock(&_locks[pos]);
    int ret = _tables[pos]->insert(uid).second ? 1 : 0;
    if (ret) _size[pos]++;
    pthread_spin_unlock(&_locks[pos]);
    return ret;
}

int UidTable::erase(uint64_t uid) {
    uid = this->normalize(uid);
    int pos = uid % ut_split;
    if (_tables[pos] == NULL) return -1;
    pthread_spin_lock(&_locks[pos]);
    int ret = _tables[pos]->erase(uid) ? 1 : 0;
    if (ret) _size[pos]--;
    pthread_spin_unlock(&_locks[pos]);
    return ret;
}

int UidTable::check(uint64_t uid) {
    uid = this->normalize(uid);
    int pos = uid % ut_split;
    if (_tables[pos] == NULL) return -1;
    pthread_spin_lock(&_locks[pos]);
    int ret = _tables[pos]->count(uid) ? 1 : 0;
    pthread_spin_unlock(&_locks[pos]);
    return ret;
}

int UidTable::dump_table(long table_id) {
    if (table_id < 0 || table_id >= ut_split) return -2;
    
    char filename[64];
    #if MULTI_SSD
    snprintf(filename, 64, "%s/%s/.Ufile-%04ld", _dir, hw::ssd_name(table_id), table_id);
    #else
    snprintf(filename, 64, "%s%04ld", _prefix, table_id);
    #endif

    phmap::BinaryOutputArchive ar(filename);

    pthread_spin_lock(&_locks[table_id]);

    if (!_tables[table_id]->phmap_dump(ar)) {
        pthread_spin_unlock(&_locks[table_id]);
        fprintf(stderr, "[Error] UidTable::dump_table: dump failed\n");
        return -1;
    }

    delete _tables[table_id];
    _tables[table_id] = NULL;
    pthread_spin_unlock(&_locks[table_id]);

    return 0;

}

int UidTable::load_table(long table_id) {
    if (table_id < 0 || table_id >= ut_split) return -2;

    char filename[64];
    #if MULTI_SSD
    snprintf(filename, 64, "%s/%s/.Ufile-%04ld", _dir, hw::ssd_name(table_id), table_id);
    #else
    snprintf(filename, 64, "%s%04ld", _prefix, table_id);
    #endif

    phmap::BinaryInputArchive ar(filename);

    pthread_spin_lock(&_locks[table_id]);

    if (_tables[table_id]) _tables[table_id]->clear();
    else _tables[table_id] = new Uset();

    if (!_tables[table_id]->phmap_load(ar)) {
        pthread_spin_unlock(&_locks[table_id]);
        fprintf(stderr, "[Error] UidTable::load_table: load failed\n");
        return -1;
    }

    remove(filename);

    pthread_spin_unlock(&_locks[table_id]);

    return 0;
}

int UidTable::reset_hash_function(long CSD) {
    for (long i = 0; i < CSD; i++) _coeffs[i] = sampleU();
    
    #pragma omp parallel for num_threads(_num_threads)
    for (long i = 0; i < ut_split; i++) {
        pthread_spin_lock(&_locks[i]);
        if (_tables[i]) {
            delete _tables[i];
        } else {
            char filename[64];
            #if MULTI_SSD
            snprintf(filename, 64, "%s/%s/.Ufile-%04ld", _dir, hw::ssd_name(i), i);
            #else
            snprintf(filename, 64, "%s%04ld", _prefix, i);
            #endif
            remove(filename);
        }
        _tables[i] = new Uset();
        _size[i] = 0;
        pthread_spin_unlock(&_locks[i]);
    }

    this->insert(0);
    for (long i = 0; i < CSD; i++) this->insert(_coeffs[i]);

    return 0;
}
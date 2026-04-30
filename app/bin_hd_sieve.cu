#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../include/pool_hd.h"

struct task_config_t {
    static constexpr int task_final_sieve = 1;
    static constexpr int task_local_pump  = 2;
    static constexpr int task_dual_hash   = 3;
    static constexpr int task_bkz         = 4;
    static constexpr long val_not_assigned = -999;
    static constexpr long default_final_sieve_ssd = 60;
    static constexpr long default_bkz_sieve_ssd = 100;
    static constexpr long min_dim_for_free = 20;

    int task = 0;
    char basis_file[256] = {};
    char output_file[256] = {};
    /* for final sieve only */
    long min_lifting_dim = val_not_assigned;
    long target_sieving_dim = val_not_assigned;
    long current_sieving_dim = val_not_assigned;
    /* for local pump only */
    long ind_l = val_not_assigned;
    long ind_r = val_not_assigned;
    long start_sieving_dim = 50;
    long max_sieving_dim = val_not_assigned;
    long enable_dual_hash = 0;
    long dual_hash_ratio = val_not_assigned;
    long enable_down_sieve = 0;
    /* for dual hash only */
    long target_position = val_not_assigned;
    /* for bkz only */
    long bkz_sieving_dim = val_not_assigned;
    long bkz_start_index = 0;
    long bkz_jump_step = 8;
    long bkz_dim_for_free = 35;
    long bkz_enable_dual_hash = 1;
    long bkz_dual_hash_ratio = 300;

    int show_help(const char *argv0);
    int parse_args(int argc, char **argv);
    int run();

    private:
    int _run_final_sieve();
    int _run_local_pump();
    int _run_dual_hash();
    int _run_bkz();
};

void run_command_file(const char* filename) {
    std::ifstream infile(filename);
    std::string line;
    
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::vector<std::string> args;
        std::stringstream ss(line);
        std::string arg;
        
        while (ss >> arg) {
            args.push_back(arg);
        }
        
        std::vector<char*> argv;
        for (auto& s : args) {
            argv.push_back(&s[0]);
        }
        
        printf("Executing command: %s\n", line.c_str());
        task_config_t task;
        if (task.parse_args(argv.size(), argv.data()) == 0) {
            if (task.run()) break;
        } else {
            break;
        }

        for (int i = 0; i < hw::ssd_num; i++) {
            char cmd[256];
            snprintf(cmd, 256, "find \".pool/%s/\" -name \".*_*\" -type f -delete 2>/dev/null", hw::ssd_name(i));
            system(cmd);
        }
    }
}

int main(int argc, char** argv) {
    if (argc == 2 && strstr(argv[1], ".cmd")) {
        run_command_file(argv[1]);
        _destory_ck_allocator();
        return 0;
    }

    printf("Cmd: ");
    for (int i = 0; i < argc; i++) {
        printf("%s ", argv[i]);
    }
    printf("\n");

    task_config_t task;
    if (task.parse_args(argc, argv) == 0) {
        task.run();
    }
    
    _destory_ck_allocator();

    return 0;
}


int task_config_t::show_help(const char *argv0) {
    printf("Usage: %s --input INPUT_FILE --task TASK [OPTIONS]\n", argv0);
    printf("Tasks:\n");

    printf("  sieve:\n");
    printf("    --TSD           Set target sieving dimension\n");
    printf("    --CSD           Set current sieving dimension\n");
    printf("    --MLD           Set min lifting dimension\n\n");
    
    printf("  pump:\n");
    printf("    --ind_l         Set left index of sieving context\n");
    printf("    --ind_r         Set right index of sieving context\n");
    printf("    --MSD           Set max sieving dimension\n");
    printf("    --SSD           Set start sieving dimension\n");
    printf("    --DH            Set dual hash ratio\n");
    printf("    --DS            Set down sieve\n");
    printf("    --output        Set output file\n\n");
    
    printf("  dh:\n");
    printf("    --POS           Set target position\n\n");

    printf("  bkz:\n");
    printf("    --BSD           Set BKZ sieving dimension\n");
    printf("    --JUMP          Set jump step\n");
    printf("    --D4F           Set dimension for free\n");
    printf("    --BDH           Set dual hash ratio\n");
    printf("    --STI           Set start index\n\n");

    return 0;
}

int task_config_t::parse_args(int argc, char **argv) {
    if (argc < 2) {
        show_help(argv[0]);
        return -1;
    }

    for (int i = 1; i < argc; i++) {
        if (strcasecmp(argv[i], "--help") == 0 || strcasecmp(argv[i], "-h") == 0) {
            show_help(argv[0]);
            return -1;
        }
    }

    for (int i = 1; i < argc; i++) {
        if (strcasecmp(argv[i], "--task") == 0 || strcasecmp(argv[i], "-t") == 0) {
            if (++i >= argc) {
                printf("Error: missing value for %s\n", argv[i-1]);
                return -1;
            }
            std::string task_name = argv[i];
            if (task_name == "sieve" || task_name == "s") {
                task = task_final_sieve;
            } else if (task_name == "pump" || task_name == "p") {
                task = task_local_pump;
            } else if (task_name == "dh" || task_name == "d") {
                task = task_dual_hash;
            } else if (task_name == "bkz" || task_name == "b") {
                task = task_bkz;
            } else {
                printf("Error: unknown task: %s\n", task_name.c_str());
                return -1;
            }
        }
    }

    if (task == 0) {
        printf("Error: task not specified\n");
        return -1;
    }

    for (int i = 1; i < argc; i++) {
        if (strcasecmp(argv[i], "--input") == 0 || strcasecmp(argv[i], "-i") == 0) {
            if (++i >= argc) {
                printf("Error: missing value for %s\n", argv[i-1]);
                return -1;
            }
            strcpy(basis_file, argv[i]);
        }
    }

    if (basis_file[0] == '\0') {
        if (argv[1][0] != '-') {
            strcpy(basis_file, argv[1]);
        } else {
            printf("Error: input file not specified\n");
            return -1;
        }
    }

    if (task == task_final_sieve) {
        for (int i = 1; i < argc; i++) {
            if (strcasecmp(argv[i], "--TSD") == 0 || strcasecmp(argv[i], "-T") == 0) {
                if (++i >= argc) {
                    printf("Error: missing value for %s\n", argv[i-1]);
                    return -1;
                }
                target_sieving_dim = atoi(argv[i]);
            } else if (strcasecmp(argv[i], "--CSD") == 0 || strcasecmp(argv[i], "-C") == 0) {
                if (++i >= argc) {
                    printf("Error: missing value for %s\n", argv[i-1]);
                    return -1;
                }
                current_sieving_dim = atoi(argv[i]);
            } else if (strcasecmp(argv[i], "--MLD") == 0 || strcasecmp(argv[i], "-M") == 0) {
                if (++i >= argc) {
                    printf("Error: missing value for %s\n", argv[i-1]);
                    return -1;
                }
                min_lifting_dim = atoi(argv[i]);
            } else if (i > 1 && strcasecmp(argv[i], "--input") && strcasecmp(argv[i], "-i") && 
                                strcasecmp(argv[i], "--task") && strcasecmp(argv[i], "-t") && argv[i][0] == '-') {
                printf("Error: invalid option for final sieve task: %s\n", argv[i]);
                return -1;
            }
        }
    }

    if (task == task_local_pump) {
        for (int i = 1; i < argc; i++) {
            if (strcasecmp(argv[i], "--ind_l") == 0 || strcasecmp(argv[i], "-l") == 0) {
                if (++i >= argc) {
                    printf("Error: missing value for %s\n", argv[i-1]);
                    return -1;
                }
                ind_l = atoi(argv[i]);
            } else if (strcasecmp(argv[i], "--ind_r") == 0 || strcasecmp(argv[i], "-r") == 0) {
                if (++i >= argc) {
                    printf("Error: missing value for %s\n", argv[i-1]);
                    return -1;
                }
                ind_r = atoi(argv[i]);
            } else if (strcasecmp(argv[i], "--output") == 0 || strcasecmp(argv[i], "-o") == 0) {
                if (++i >= argc) {
                    printf("Error: missing value for %s\n", argv[i-1]);
                    return -1;
                }
                strcpy(output_file, argv[i]);
            } else if (strcasecmp(argv[i], "--SSD") == 0) {
                if (++i >= argc) {
                    printf("Error: missing value for %s\n", argv[i-1]);
                    return -1;
                }
                start_sieving_dim = atoi(argv[i]);
            } else if (strcasecmp(argv[i], "--MSD") == 0) {
                if (++i >= argc) {
                    printf("Error: missing value for %s\n", argv[i-1]);
                    return -1;
                }
                max_sieving_dim = atoi(argv[i]);
            } else if (strcasecmp(argv[i], "--DH") == 0) {
                enable_dual_hash = 1;
                if (i + 1 < argc && argv[i+1][0] != '-') {
                    dual_hash_ratio = atoi(argv[++i]);
                }
            } else if (strcasecmp(argv[i], "--DS") == 0) {
                enable_down_sieve = 1;
            } else if (i > 1 && strcasecmp(argv[i], "--input") && strcasecmp(argv[i], "-i") && 
                                strcasecmp(argv[i], "--task") && strcasecmp(argv[i], "-t") && argv[i][0] == '-') {
                printf("Error: invalid option for pump task: %s\n", argv[i]);
                return -1;
            }
        }
        if (max_sieving_dim == val_not_assigned) {
            printf("Error: max sieving dimension not specified\n");
            return -1;
        }
        if (output_file[0] == '\0') {
            snprintf(output_file, sizeof(output_file), "%sr", basis_file);
            if (access(output_file, F_OK) == 0) {
                printf("Error: output file unspecified and %s exists\n", output_file);
                return -1;
            } else {
                printf("output file set to default %s\n", output_file);
            }
        }
    }

    if (task == task_dual_hash) {
        for (int i = 1; i < argc; i++) {
            if (strcasecmp(argv[i], "--POS") == 0 || strcasecmp(argv[i], "-P") == 0) {
                if (++i >= argc) {
                    printf("Error: missing value for %s\n", argv[i-1]);
                    return -1;
                }
                target_position = atoi(argv[i]);
            } else if (i > 1 && strcasecmp(argv[i], "--input") && strcasecmp(argv[i], "-i") && 
                                strcasecmp(argv[i], "--task") && strcasecmp(argv[i], "-t") && argv[i][0] == '-') {
                printf("Error: invalid option for dual hash task: %s\n", argv[i]);
                return -1;
            }
        }
    }
    
    if (task == task_bkz) {
        for (int i = 1; i < argc; i++) {
            if (strcasecmp(argv[i], "--BSD") == 0 || strcasecmp(argv[i], "-B") == 0) {
                if (++i >= argc) {
                    printf("Error: missing value for %s\n", argv[i-1]);
                    return -1;
                }
                bkz_sieving_dim = atoi(argv[i]);
            } else if (strcasecmp(argv[i], "--JUMP") == 0 || strcasecmp(argv[i], "-J") == 0) {
                if (++i >= argc) {
                    printf("Error: missing value for %s\n", argv[i-1]);
                    return -1;
                }
                bkz_jump_step = atoi(argv[i]);
            } else if (strcasecmp(argv[i], "--D4F") == 0 || strcasecmp(argv[i], "-F") == 0) {
                if (++i >= argc) {
                    printf("Error: missing value for %s\n", argv[i-1]);
                    return -1;
                }
                bkz_dim_for_free = atoi(argv[i]);
            } else if (strcasecmp(argv[i], "--BDH") == 0) {
                if (++i >= argc) {
                    printf("Error: missing value for %s\n", argv[i-1]);
                    return -1;
                }
                bkz_dual_hash_ratio = atoi(argv[i]);
            } else if (strcasecmp(argv[i], "--STI") == 0) {
                if (++i >= argc) {
                    printf("Error: missing value for %s\n", argv[i-1]);
                    return -1;
                }
                bkz_start_index = atoi(argv[i]);
            } else if (strcasecmp(argv[i], "--output") == 0 || strcasecmp(argv[i], "-o") == 0) {
                if (++i >= argc) {
                    printf("Error: missing value for %s\n", argv[i-1]);
                    return -1;
                }
                strcpy(output_file, argv[i]);
            } else if (i > 1 && strcasecmp(argv[i], "--input") && strcasecmp(argv[i], "-i") && 
                                strcasecmp(argv[i], "--task") && strcasecmp(argv[i], "-t") && argv[i][0] == '-') {
                printf("Error: invalid option for BKZ task: %s\n", argv[i]);
                return -1;
            }
        }
        if (output_file[0] == '\0') {
            snprintf(output_file, sizeof(output_file), "%sr", basis_file);
            if (access(output_file, F_OK) == 0) {
                printf("Error: output file unspecified and %s exists\n", output_file);
                return -1;
            } else {
                printf("output file set to default %s\n", output_file);
            }
        }
    }

    return 0;
}

#include <time.h>
#include <dirent.h>
#include <sys/stat.h>

int _sieve(Pool_hd_t *p) {
    int ret = 0;
    if (p->CSD < 112) {
        ret = p->bgj1_Sieve_hd();
    } else if (p->CSD < 135) {
        ret = p->bgj2_Sieve_hd();
    } else if (p->CSD <= 143) {
        ret = p->bgj2_Sieve_hd();
    } else {
        ret = p->bgj3l_Sieve_hd();
    }
    return ret;
}

int _time_stamp(char dst[256]) {
    time_t now = time(NULL);
    struct tm *local_time = localtime(&now);
    strftime(dst, 256, "%Y-%m-%d-%H-%M-%S", local_time);

    return 0;
}

int _sync_basis(Lattice_QP *L, const char *basis_file, int ind_l, int ind_r) {
    char ts[256];
    _time_stamp(ts);
    char filename[256];
    snprintf(filename, 256, ".%s-%d-%d-%s", basis_file, ind_l, ind_r, ts);
    L->store(filename);

    return 0;
}

int _dir_empty(const char *dirname) {
    DIR *dir = opendir(dirname);
    if (!dir) {
        perror("opendir failed");
        return -1;
    }

    struct dirent *entry;
    int is_empty = 1;

    while ((entry = readdir(dir))) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
        is_empty = 0;
        break;
    }

    closedir(dir);
    return is_empty;
}

int _file_chunk000000(char *filename, const char *dirname) {
    DIR *dir = opendir(dirname);
    if (!dir) {
        fprintf(stderr, "Cannot open directory %s\n", dirname);
        return -1;
    }

    struct dirent *entry;

    while ((entry = readdir(dir))) {
        size_t name_len = strlen(entry->d_name);
        if (name_len >= 6 && strcmp(entry->d_name + name_len - 6, "000000") == 0) {
            if (filename[0]) {
                fprintf(stderr, "Multiple files ending with 000000 found\n");
                closedir(dir);
                return -1;
            } else snprintf(filename, 256, "%s", entry->d_name);
        }
    }
    closedir(dir);

    return 0;
}

int _get_real_context(int *ind_l, int *ind_r, uint64_t &hash) {
    char dirname[256];
    snprintf(dirname, sizeof(dirname), ".pool/%s", hw::ssd_name(0));

    char metadata[12];
    char filename[256] = {};

    if (_file_chunk000000(filename, dirname) == -1) {
        return -1;
    }

    if (filename[0] == '\0') {
        for (int i = 0; i < hw::ssd_num; i++) {
            snprintf(dirname, sizeof(dirname), ".pool/%s", hw::ssd_name(i));
            if (_dir_empty(dirname) != 1) {
                printf("%s not clean but file for chunk 000000 not found?\n", dirname);
                return -1;
            }
        }
        return 0;
    }

    // Read metadata
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/%s", dirname, filename);
    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open file %s\n", filepath);
        return -1;
    }

    size_t bytes_read = fread(metadata, 1, 12, fp);
    fclose(fp);

    if (bytes_read != 12) {
        fprintf(stderr, "Failed to read 12 bytes from chunk 000000\n");
        return -1;
    }

    uint8_t read_l = *((uint8_t *)(&metadata[10]));
    uint8_t read_r = *((uint8_t *)(&metadata[11]));
    hash = *((uint64_t *)(&metadata[2]));

    *ind_l = read_l;
    *ind_r = read_r;

    return 0;
}

void _print_memory_usage() {
    char statm_path[64];
    sprintf(statm_path, "/proc/%d/statm", getpid());
    FILE* fp = fopen(statm_path, "r");
    if (fp) {
        long virt, res, shr;
        if (fscanf(fp, "%ld %ld %ld", &virt, &res, &shr) == 3) {
            // Convert pages to MB (4KB pages)
            printf("Memory Usage - VIRT: %.2f GB, RES: %.2f GB, SHR: %.2f GB\n",
                   virt * 4. / 1024. / 1024., res * 4. / 1024. / 1024., shr * 4. / 1024. / 1024.);
        }
        fclose(fp);
    }
}

int task_config_t::run() {
    const char *dir_list[] = {".bucket", ".sol", ".uid", ".pool"};

    for (int i = 0; i < ((task == task_local_pump || task == task_bkz) ? 4 : 
                  (task == task_final_sieve || task == task_dual_hash) ? 3 : 0); i++) {
        for (int j = 0; j < hw::ssd_num; j++) {
            char dirname[256];
            snprintf(dirname, sizeof(dirname), "%s/%s", dir_list[i], hw::ssd_name(j));
            if (_dir_empty(dirname) != 1) {
                printf("%s not clean, clean it before start\n", dirname);
                return -1;
            }
        }
    }

    if (task == task_final_sieve) {
        return _run_final_sieve();
    } else if (task == task_local_pump) {
        return _run_local_pump();
    } else if (task == task_dual_hash) {
        return _run_dual_hash();
    } else if (task == task_bkz) {
        return _run_bkz();
    } else {
        printf("Invalid task %d, nothing done\n", task);
    }
    return -1;
}

int task_config_t::_run_final_sieve() {
    Lattice_QP L(this->basis_file);
    if (!L.NumRows()) {
        printf("[Error] basis load fail, nothing done\n");
        return -1;
    }

    if (target_sieving_dim == val_not_assigned) {
        target_sieving_dim = 176;
    }

    if (min_lifting_dim == val_not_assigned) {
        min_lifting_dim = 176;
    }

    int real_ind_l = -1, real_ind_r = -1;
    uint64_t real_hash = 0;
    if (_get_real_context(&real_ind_l, &real_ind_r, real_hash) == -1) {
        printf("[Error] wrong pool format? nothing done\n");
        return -1;
    }

    int need_sample = 0;

    if (real_ind_l == -1 && real_ind_r == -1) {
        if (current_sieving_dim == val_not_assigned) {
            current_sieving_dim = default_final_sieve_ssd;
            need_sample = 1;
        } else {
            printf("[Error] sieving context not detected but current sieving dimenson specified\n");
            return -1;
        }
    } else {
        if (real_ind_r != L.NumRows()) {
            printf("[Error] sieving context [%d, %d] but lattice dim %ld?", real_ind_l, real_ind_r, L.NumRows());
            return -1;
        }
        if (current_sieving_dim == val_not_assigned) current_sieving_dim = real_ind_r - real_ind_l;
        else {
            if (current_sieving_dim != real_ind_r - real_ind_l) {
                printf("[Error] sieving context [%d, %d] detected but current sieving dimenson specified %ld\n", real_ind_l, real_ind_r, current_sieving_dim);
                return -1;
            }
        }
    }

    Pool_hd_t pool(&L);
    pool.set_sieving_context(L.NumRows() - current_sieving_dim, L.NumRows());
    pool.set_boost_depth(0);

    pool.set_num_threads(8);

    if (need_sample) {
        pool.sampling(3.2 * pow(4./3., pool.CSD * .5) - 5);
    } else {
        if (real_hash) pool.basis_hash = real_hash;
        pool.pwc_manager->set_pool(&pool);
        pool.load(3);
    }

    pool.check(3);

    for (;;) {
        int ret = _sieve(&pool);
        if (ret == 1) {
            break;
        }
        if (pool.CSD == 100) {
            if (pool.check_dim_lose() == -1) {
                printf("[Error] sieve stuck at dimension 100, aborted\n");
                fflush(stdout);
                return -1;
            }
        }
        if (pool.CSD >= min_lifting_dim) pool.show_min_lift(pool.index_l <= 40 ? 0 : pool.index_l - 40);
        if (pool.CSD < target_sieving_dim) pool.extend_left();
        else break;
    }

    return 0;
}

int task_config_t::_run_local_pump() {
    Lattice_QP L(this->basis_file);
    if (!L.NumRows()) {
        printf("[Error] basis load fail, nothing done\n");
        return -1;
    }

    if (ind_l == val_not_assigned) ind_l = 0;
    if (ind_r == val_not_assigned) ind_r = L.NumRows();
    
    double start_pot = L.Pot();
    struct timeval pump_start, pump_stop;
    gettimeofday(&pump_start, NULL);
    struct timeval sieve_start, sieve_stop;
    double last_sieve_time = 0.0;

    if (ind_l < 0 || ind_r < 0 || ind_l > L.NumRows() || ind_r > L.NumRows() || ind_l >= ind_r) {
        printf("invalid sieving context [%ld, %ld)\n", ind_l, ind_r);
        return -1;
    }
    
    if (max_sieving_dim >= ind_r - ind_l) {
        max_sieving_dim = ind_r - ind_l - 1;
    }

    if (max_sieving_dim < 40) {
        printf("max sieving dimension too small(%ld), nothing done\n", max_sieving_dim);
        return -1;
    }

    if (start_sieving_dim > max_sieving_dim) start_sieving_dim = max_sieving_dim;

    if (enable_dual_hash && dual_hash_ratio == val_not_assigned) {
        dual_hash_ratio = 10 * (ind_r - ind_l - max_sieving_dim);
        if (dual_hash_ratio < 200) dual_hash_ratio = 200;
    }

    Lattice_QP *L_loc = L.b_loc_QP(ind_l, ind_r);
    char ts[256];
    _time_stamp(ts);
    char loc_basis_file[256];
    snprintf(loc_basis_file, 256, ".%s-%ld-%ld-%s", basis_file, ind_l, ind_r, ts);
    L_loc->store(loc_basis_file);
    delete L_loc;

    Lattice_QP L_locs(loc_basis_file);
    Pool_hd_t pool(&L_locs);
    pool.set_sieving_context(L_locs.NumRows() - start_sieving_dim, L_locs.NumRows());
    pool.set_boost_depth(0);

    pool.set_num_threads(8);

    pool.sampling(BGJ1_SIZE_RATIO * pow(4./3., pool.CSD * .5) - 5);

    while (pool.CSD <= max_sieving_dim) {
        gettimeofday(&sieve_start, NULL);
        int ret = _sieve(&pool);
        gettimeofday(&sieve_stop, NULL);

        last_sieve_time += (sieve_stop.tv_sec - sieve_start.tv_sec) + (sieve_stop.tv_usec - sieve_start.tv_usec) * 1e-6;
        {
            if ((ret == -1 && pool.CSD < 80) || pool.CSD == 80) {
                if (pool.check_dim_lose() == -1) {
                    printf("sieve stuck, aborted\n");
                    fflush(stdout);
                    return -1;
                }
            }
            pool.store();
            if (pool.CSD < max_sieving_dim) {
                pool.extend_left();
                last_sieve_time = 0.0;
            }
            else break;
        }
    }

    pool.down_sieve_flag = 1;

    int ind = 0;
    while (pool.CSD >= 60) {
        long pos = -1;
        if (enable_dual_hash) {
            double dh_expect_time = enable_down_sieve ? (last_sieve_time * dual_hash_ratio * 1e-3) : 
                                                        (pow(10.0, pool.CSD * 0.1 - 11.4) * (pool.index_l - ind) / 32.0);
            printf("dual hash expect time = %.2fs\n", dh_expect_time);
            pool.dh_insert(ind++, 1.2, dh_expect_time, &pos);
            last_sieve_time = 0.0;
        } else pool.insert(ind++, 1.2, &pos);
        L_locs.store(".bkz.tmp_basis");
        if (ind <= pos) ind = pos + 1;
        if (ind >= pool.index_l) ind = pool.index_l - 1;
        pool.shrink(BGJ4_SIZE_RATIO * pow(4./3., pool.CSD * .5) - 5);

        if (pool.pwc_manager->num_vec() < 0.7 * (BGJ4_SIZE_RATIO * pow(4./3., pool.CSD * .5) - 5)) {
            break;
        }

        if (!enable_down_sieve) {
            if (ind > 10) break;
            continue;
        }

        for (;;) {
            gettimeofday(&sieve_start, NULL);
            int ret = _sieve(&pool);
            gettimeofday(&sieve_stop, NULL);
            last_sieve_time += (sieve_stop.tv_sec - sieve_start.tv_sec) + (sieve_stop.tv_usec - sieve_start.tv_usec) * 1e-6;
            break;
        }
    }

    _sync_basis(&L_locs, basis_file, ind_l, ind_r);
    if (L_locs.NumRows() < 90) L_locs.LLL_DEEP_QP(0.99);

    L.trans_to(ind_l, ind_r, &L_locs);
    L.compute_gso_QP();
    L.size_reduce();
    /// L.LLL_QP();
    L.compute_gso_QP();
    gettimeofday(&pump_stop, NULL);
    double pump_time = (pump_stop.tv_sec - pump_start.tv_sec) + (pump_stop.tv_usec - pump_start.tv_usec) * 1e-6;
    double d_pot = L.Pot() - start_pot;
    printf("pump time = %.2fs, dPot = %.2f\n", pump_time, -d_pot);
    L.show_dist_vec();

    L.store(output_file);

    return 0;
}

int task_config_t::_run_dual_hash() {
    Lattice_QP L(this->basis_file);
    if (!L.NumRows()) {
        printf("[Error] basis load fail, nothing done\n");
        return -1;
    }

    int real_ind_l = -1, real_ind_r = -1;
    uint64_t real_hash = 0;
    if (_get_real_context(&real_ind_l, &real_ind_r, real_hash) == -1) {
        printf("[Error] wrong pool format? nothing done\n");
        return -1;
    }

    if (real_ind_l == -1 && real_ind_r == -1) {
        printf("[Error] sieving context not detected for dual hash\n");
        return -1;
    }

    if (real_ind_r != L.NumRows()) {
        printf("[Error] sieving context [%d, %d] but lattice dim %ld?", real_ind_l, real_ind_r, L.NumRows());
        return -1;
    }

    if (target_position == val_not_assigned) target_position = real_ind_l < 40 ? 0 : real_ind_l - 40;
    if (target_position < 0 || target_position >= real_ind_l) {
        printf("[Error] invalid target position %ld\n", target_position);
        return -1;
    }

    Pool_hd_t pool(&L);
    pool.set_sieving_context(real_ind_l, real_ind_r);
    pool.set_boost_depth(0);

    pool.set_num_threads(8);

    if (real_hash) {
        pool.basis_hash = real_hash;
        pool.pwc_manager->set_pool(&pool);
    }

    pool.load(3);

    pool.check(3);

    pool.dh_final(target_position, 1.2);

    return 0;
}

int task_config_t::_run_bkz() {
    Lattice_QP L(this->basis_file);
    if (!L.NumRows()) {
        printf("[Error] basis load fail, nothing done\n");
        return -1;
    }

    if (bkz_dual_hash_ratio == 0) bkz_enable_dual_hash = 0;

    long block_size = bkz_dim_for_free + bkz_sieving_dim;

    for (long index = bkz_start_index; index < L.NumRows(); index += bkz_jump_step) {
        char curr_basis_name[256];
        char next_basis_name[256];
        snprintf(curr_basis_name, 256, ".%s_%ld", basis_file, index);
        snprintf(next_basis_name, 256, ".%s_%ld", basis_file, index + bkz_jump_step);
        
        if (index == bkz_start_index) L.store(curr_basis_name);

        int last_pump = index + bkz_sieving_dim + 2 >= L.NumRows();
        
        task_config_t pump_task;
        pump_task.task = task_local_pump;
        memcpy(pump_task.basis_file, curr_basis_name, 256);
        memcpy(pump_task.output_file, next_basis_name, 256);
        pump_task.ind_l = index;
        pump_task.ind_r = (index + block_size) > L.NumRows() ? L.NumRows() : (index + block_size);
        pump_task.start_sieving_dim = default_bkz_sieve_ssd < bkz_sieving_dim ? 
                                      default_bkz_sieve_ssd : bkz_sieving_dim;
        pump_task.max_sieving_dim = bkz_sieving_dim;
        pump_task.enable_dual_hash = bkz_enable_dual_hash;
        pump_task.dual_hash_ratio = bkz_dual_hash_ratio;
        pump_task.enable_down_sieve = 1;
        if (pump_task.run() == -1) break;
        for (int i = 0; i < hw::ssd_num; i++) {
            char cmd[256];
            snprintf(cmd, 256, "find \".pool/%s/\" -name \".*_*\" -type f -delete 2>/dev/null", hw::ssd_name(i));
            system(cmd);
        }

        if (last_pump) {
            char cmd[256];
            snprintf(cmd, 256, "mv %s %s", next_basis_name, output_file);
            system(cmd);
            break;
        }
    }

    return 0;
}

#include "../include/lattice.h"
#include "../include/utils.h"
#include "../include/vec.h"
#include "../include/quad.h"
#include <fstream>

//constructions and distruction
Lattice_QP::Lattice_QP(long n, long m){
    this->n = n;
    this->m = m;
    n8 = ((n + 7) / 8) * 8;
    m8 = ((m + 7) / 8) * 8;
    alloc();
}
Lattice_QP::Lattice_QP(NTL::Mat<double>& L){
    this->n = L.NumRows();
    this->m = L.NumCols();
    n8 = ((n + 7) / 8) * 8;
    m8 = ((m + 7) / 8) * 8;
    alloc();

    for (long i = 0; i < n; i++){
        for (long j = 0; j < m; j++){
            NTL::quad_float x(L[i][j]);
            b.hi[i][j] = x.hi;
            b.lo[i][j] = x.lo;
        }
    }
}
Lattice_QP::Lattice_QP(NTL::Mat<NTL::ZZ>& L){
    this->n = L.NumRows();
    this->m = L.NumCols();
    n8 = ((n + 7) / 8) * 8;
    m8 = ((m + 7) / 8) * 8;
    alloc();

    for (long i = 0; i < n; i++){
        for (long j = 0; j < m; j++){
            NTL::quad_float x =  NTL::conv<NTL::quad_float>(L[i][j]);
            b.hi[i][j] = x.hi;
            b.lo[i][j] = x.lo;
        }
    }
}
Lattice_QP::Lattice_QP(NTL::Mat<NTL::quad_float>& L){
    this->n = L.NumRows();
    this->m = L.NumCols();
    n8 = ((n + 7) / 8) * 8;
    m8 = ((m + 7) / 8) * 8;
    alloc();

    for (long i = 0; i < n; i++){
        for (long j = 0; j < m; j++){
            b.hi[i][j] = L[i][j].hi;
            b.lo[i][j] = L[i][j].lo;
        }
    }
}
Lattice_QP::Lattice_QP(const char *filename){
    NTL::Mat<NTL::quad_float> L;
    std::ifstream data(filename, std::ios::in);
    data >> L;
    if (!L.NumRows()){
        std::cerr << "[Error] Lattice_QP::Lattice_QP(const char *filename): incorrect file format!\n";
        return;
    }
    this->n = L.NumRows();
    this->m = L.NumCols();
    n8 = ((n + 7) / 8) * 8;
    m8 = ((m + 7) / 8) * 8;
    alloc();

    for (long i = 0; i < n; i++){
        for (long j = 0; j < m; j++){
            b.hi[i][j] = L[i][j].hi;
            b.lo[i][j] = L[i][j].lo;
        }
    }
}
Lattice_QP::~Lattice_QP(){
    clear();
}
int Lattice_QP::set_size(long n, long m){
    clear();
    this->n = n;
    this->m = m;
    n8 = ((n + 7) / 8) * 8;
    m8 = ((m + 7) / 8) * 8;
    return alloc();
}

//alloc and clear
int Lattice_QP::alloc(){
    if ((m <= 0) || (n <= 0)) {
        std::cerr << "[Error] Lattice_QP::alloc: please set the size before alloc\n";
        return 0;
    }
    b = NEW_MAT_QP(n, m);
    miu = NEW_MAT_QP(n, n);
    b_star = NEW_MAT_QP(n, m);
    B = NEW_VEC_QP(n);

    if (!B.hi || !B.lo || !b_star.hi || !b_star.lo || !miu.hi || !miu.lo || !b.hi || !b.lo) {
        std::cerr << "[Error] Lattice_FP::alloc: allocation failed\n";
        clear();
        return 0;
    }
    return 1;
}
int Lattice_QP::clear(){
    if (b.hi || b.lo) FREE_MAT_QP(b);
    if (b_star.hi || b_star.lo) FREE_MAT_QP(b_star);
    if (miu.hi || miu.lo) FREE_MAT_QP(miu);
    if (B.hi || B.lo) FREE_VEC_QP(B);
    return 1;
}

//tools
long Lattice_QP::pump_red_msd(){
    double pottt = this->Pot();
    double detn = this->detn();
    long ret = 60;
    double min_gap = 1e30;
    for (long dim = 60; dim < 180 && dim <= this->n; dim++){
        if (fabs(pot(detn, this->n, dim) - pottt) < min_gap) {
            min_gap = fabs(pot(detn, this->n, dim) - pottt);
            ret = dim;
        }
    }
    if (ret <= 60) {
        fprintf(stderr, "[Warning] Lattice_QP::pump_red_msd: not BKZ-60 reduced yet.\n");
    }
    return ret;
}
void Lattice_QP::show_dist_vec(){
    if (!gso_computed) compute_gso_QP();
    std::cout << "[";
    for (long i = 0; i < n-1; i++){
        std::cout << sqrt(B.hi[i]) << ", ";
    }
    std::cout << sqrt(B.hi[n-1]) << "]\n";
}
void Lattice_QP::show_miu(){
    if (!gso_computed) compute_gso_QP();
    std::cout << "[";
    for (long i = 0; i < n; i ++){
        std::cout << "[";
        for (long j = 0; j < n-1; j++){
            std::cout << miu.hi[i][j] << " ";
        }
        std::cout << miu.hi[i][n-1] << "]\n";
    }
    std::cout << "]\n";
}
void Lattice_QP::show_length(){
    std::cout << "[";
    for (long i = 0; i < n-1; i++){
        std::cout << sqrt(dot(b.hi[i], b.hi[i], n8)) << " ";
    }
    std::cout << sqrt(dot(b.hi[n-1], b.hi[n-1], n8)) << "]\n";
}
double Lattice_QP::detn(){
    if (!gso_computed) compute_gso_QP();
    double detn = 1.0;
    for (long i = 0; i < n; i++){
        detn *= pow(B.hi[i], 0.5/n);
    }
    return detn;
}
double Lattice_QP::gh(){
    return this->detn() * gh_coeff(n);
}
double Lattice_QP::gh(long ind_l, long ind_r){
    if (!gso_computed) compute_gso_QP();
    double ret = 1.0;
    double e = 0.5/(ind_r - ind_l);
    for (long i = ind_l; i < ind_r; i++){
        ret *= pow(B.hi[i], e);
    }
    return ret * gh_coeff(ind_r - ind_l);
}
double Lattice_QP::Pot(){
    if (!gso_computed) compute_gso_QP();
    double pot = 0.0;
    for (long i = 0; i < n; i++){
        pot += (n-i) * log2(B.hi[i]);
    }
    return pot;
}

void Lattice_QP::store(const char *filename){
    std::ofstream hout(filename, std::ios::out);
    hout << *this <<std::endl;
}
int Lattice_QP::to_int(){
    bool fpwarn = false;
    bool fperr = false;
    for (long i = 0; i < n; i++){
        if (fperr) break;
        for (long j = 0; j < m; j++){
            if (fabs(b.hi[i][j]-round(b.hi[i][j])) > 0.05){
                fpwarn = true;
                if (fabs(b.hi[i][j] - round(b.hi[i][j])) > 0.2) {
                    fperr = true;
                    break;
                }
            }
        }
    }
    if (fperr){
        std::cerr << "[Error] Lattice_QP::round: floating point precision error, aborted\n";
        return 0;
    }
    if (fpwarn){
        std::cerr << "[Warning] Lattice_QP::round float point precision warning\n";
    }
    gso_computed = 0;
    for (long i = 0; i < n; i++){
        for (long j = 0; j < m; j++){
            NTL::quad_float x = NTL::to_quad_float(round(b.hi[i][j]));
            b.hi[i][j] = x.hi;
            b.lo[i][j] = x.lo;
        }
    }
    return 1;
}


void Lattice_QP::tail_shuffle(long l){
    l = (l < n) ? l : n;
    VEC_QP tmp = NEW_VEC_QP(m8);
    //shuffle [n-l, n]
    for (long epoch = 0; epoch < 13; epoch ++){
        for (long i = n-l; i < n; i++){
            long j = rand()%l + n-l;
            if (i != j){
                copy(tmp.hi, tmp.lo, b.hi[i], b.lo[i], m8);
                copy(b.hi[i], b.lo[i], b.hi[j], b.lo[j], m8);
                copy(b.hi[j], b.lo[j], tmp.hi, tmp.lo, m8);
            }
        }
        for (long i = n-l; i < n; i++){
            long j = rand()%l + n-l;
            if (i != j){
                long q = rand()%5 - 2;
                red(b.hi[i], b.lo[i], b.hi[j], b.lo[j], NTL::to_quad_float(q), m8);
            }
        }
    }
    compute_gso_QP(n-l);
    if (l <= 50) {
        LLL_QP(0.99, n-l, n);
        LLL_DEEP_QP(0.99, n-l, n);
    } else {
        LLL_QP(0.7, n-l, n-l+30);
        LLL_QP(0.7, n-l+15, n-l+50);
        LLL_QP(0.7, n-50, n-20);
        LLL_QP(0.7, n-30, n);
        LLL_QP(0.9, n-l, n-l+30);
        LLL_QP(0.9, n-l+15, n-l+50);
        LLL_QP(0.9, n-50, n-20);
        LLL_QP(0.9, n-30, n);
        LLL_QP(0.99, n-l, n-l+50);
        LLL_QP(0.99, n-50, n);
        LLL_QP(0.99, n-l, n);
        LLL_DEEP_QP(0.9, n-l, n);
        LLL_DEEP_QP(0.97, n-l, n);
        LLL_DEEP_QP(0.99, n-l, n);
    }
}

std::ostream& operator << (std::ostream& os, Lattice_QP& L){
    long n = L.NumRows();
    long m = L.NumCols();
    NTL::quad_float x;
    x.SetOutputPrecision(40);
    MAT_QP b_ = L.get_b();
    os << "[";
    for (long i = 0; i < n; i++){
        os << "[";
        for (long j = 0; j < m-1; j++){
            NTL::quad_float y(b_.hi[i][j], b_.lo[i][j]);
            os << y << " ";
        }
        NTL::quad_float y(b_.hi[i][m-1], b_.lo[i][m-1]);
        os << y << "]\n";
    }
    os << "]";
    return os;
}



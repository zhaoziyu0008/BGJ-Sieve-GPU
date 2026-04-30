#ifndef __LATTICE_H
#define __LATTICE_H

#define GSO_COMPUTED_QP 2L

#include <NTL/LLL.h>
#include <string.h>
#include "quad.h"



// We only use quad_float, which is necessary in dimension > 500
class Lattice_QP {
    public:
        //constructions
            Lattice_QP(){};
            //create a lattice of size (n, m) with all entries 0
            Lattice_QP(long n, long m);
            //create a lattice form NTL matrix
            Lattice_QP(NTL::Mat<double>& L);
            Lattice_QP(NTL::Mat<NTL::ZZ>& L);
            Lattice_QP(NTL::Mat<NTL::quad_float>& L);
            //read a lattice from the file
            Lattice_QP(const char *filename);
            ~Lattice_QP();
            int set_size(long n, long m);
        
        //basic info
            //number of vectors
            inline long NumRows();
            //length of each vectors
            inline long NumCols();
            //if the data is changed, you should set gso status by hand.
            //the lattice basis
            inline MAT_QP get_b();
            //the gso data, b[i] = \sum_{j=0}^{i} b_star[j] * miu[i][j];
            //miu[i][i] = 1.0, B[i] = dot(b_star[i], b_star[i]);
            inline MAT_QP get_miu();
            inline MAT_QP get_b_star();
            inline VEC_QP get_B();
            //if the gso data is corrent, the gso_status should be 2, else 0.
            inline void set_gso_status(long status);
            inline long get_gso_status();

        //gso
            //compute the gso data
            void compute_gso_QP();
            //compute the gso data, and assume the gso data on [0, index_l] is already correct
            void compute_gso_QP(long index_l);
            //apply a linear transform on the lattice, which transform the local projected lattice on [ind_l, ind_r] to L
            int trans_to(long ind_l, long ind_r, Lattice_QP *L);
            //transform the lattice by matrix A
            void trans_by(long **A, long index_l, long index_r);
            void trans_by(double **A, long index_l, long index_r);
            //get a local projected lattice on [ind_l, ind_r]}
            Lattice_QP *b_loc_QP(long ind_l, long ind_r);
            // success or nothing done and output an error message, L_src will be corrupted
            int reconstruct(Lattice_QP *L_src);

        //lattice reduction
            int size_reduce();
            int LLL_QP(double delta = 0.99);
            int LLL_DEEP_QP(double delta = 0.99);
            //local version
            int size_reduce(long index);
            int size_reduce(long l, long r);
            int LLL_QP(double delta, long ind_l, long ind_r);
            int LLL_DEEP_QP(double delta, long ind_l, long ind_r);
            //shuffle the last l vectors, to prevent stuck when sieving
            void tail_shuffle(long l);

        //dual
            int dual_QP(NTL::Mat<NTL::quad_float>& Ld);
            int dual_QP(NTL::Mat<double>& Ld);
            Lattice_QP *dual_QP();
            void usd();
            int dual_size_red();

        //tools
            long pump_red_msd();
            void show_dist_vec();
            void show_miu();
            void show_length();
            //return det(L)^{1/n}
            double detn();
            //return the gaussian heuristic of L
            double gh();
            double gh(long ind_l, long ind_r);
            //return lg2 of Pot(L)
            double Pot();
            //store the lattice
            void store(const char *filename);
            //round the lattice to integer, cut the tail of things like 1.0000000001
            int to_int();
    private:
        long n;
        long m;
        long n8;
        long m8;
        
        int alloc();
        int clear();
        void LLL_reduce(long k, long l);
        void LLL_swap(long k);

        //gso data
        long gso_computed = 0;
        MAT_QP b = {NULL, NULL};
        MAT_QP miu = {NULL, NULL};
        MAT_QP b_star = {NULL, NULL};
        VEC_QP B = {NULL, NULL};
};

// so we can use "cout << L" to print L
std::ostream& operator << (std::ostream& os, Lattice_QP& L);

inline long Lattice_QP::NumRows(){
    return n;
}
inline long Lattice_QP::NumCols(){
    return m;
}
inline MAT_QP Lattice_QP::get_b(){
    return b;
}
inline MAT_QP Lattice_QP::get_miu(){
    return miu;
}
inline MAT_QP Lattice_QP::get_b_star(){
    return b_star;
}
inline VEC_QP Lattice_QP::get_B(){
    return B;
}
inline void Lattice_QP::set_gso_status(long status){
    if ((status < 0) || (status > 2)) {
        std::cerr << "[Error] Lattice_FP::set_gso_status: wrong status code, aborted\n";
        return;
    }
    gso_computed = status;
    return;
}
inline long Lattice_QP::get_gso_status(){
    return gso_computed;
}

#endif

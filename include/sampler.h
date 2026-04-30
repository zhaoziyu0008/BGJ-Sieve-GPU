#ifndef __SAMPLER_H
#define __SAMPLER_H

#include <stdint.h>
#include <math.h>
#include <random>
#include <NTL/LLL.h>

class Lattice_QP;

/* random generator. */
uint64_t Uniform_u64();

class DGS1d{
    public:
        DGS1d(){this->gen64.seed(std::random_device{}());}
        DGS1d(int seed){this->gen64.seed(seed);}
        ~DGS1d(){}
        int discrete_gaussian(double mu, double sigma2);
        void set_seed(int seed){this->gen64.seed(seed);}
        inline uint64_t Uniform_u64(){return gen64();}
    private:
        std::mt19937_64 gen64;
};

class NaiveDGS{
    public:
        NaiveDGS(){this->baseDGS = new DGS1d();}
        NaiveDGS(int seed){this->baseDGS = new DGS1d(seed);}
        NaiveDGS(Lattice_QP *_L){this->baseDGS = new DGS1d(); this->L = _L;}
        NaiveDGS(Lattice_QP *_L, int seed){this->baseDGS = new DGS1d(seed); this->L = _L;}
        ~NaiveDGS(){this->baseDGS->~DGS1d(); delete this->baseDGS;}
        void set_seed(int seed){this->baseDGS->set_seed(seed);}
        void set_L(Lattice_QP *_L){this->L = _L;}
        double *gen_vec(double sigma2, long ind_l, long ind_r, int sr = 1);
        long *gen_coeff(double sigma2, long ind_l, long ind_r);
        double *gen_vec(double sigma2, int sr = 1){return this->gen_vec(sigma2, 0, this->L->NumRows(), sr);}
        long *gen_coeff(double sigma2){return this->gen_coeff(sigma2, 0, this->L->NumRows());}
    private:
        DGS1d *baseDGS;
        Lattice_QP *L;
};


NTL::Vec<double> random_vec(long n, double l);
NTL::Mat<double> random_vec(long n, double l, long num);
Lattice_QP *random_qary_lattice(long dim, long q, long m);


#endif
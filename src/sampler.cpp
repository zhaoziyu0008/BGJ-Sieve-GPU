#include "../include/lattice.h"
#include "../include/sampler.h"
#include "../include/utils.h"
#include "../include/vec.h"



#define POWER264 18446744073709551616.0 

#define ADJUST_SIGMA_WHEN_TOOSMALL 1

std::mt19937_64 gen64(std::random_device{}());

int DGS1d::discrete_gaussian(double mu, double sigma2){
    long mu0 = round(mu);
    double bias = mu - mu0;
    double isigma2 = 0.5/sigma2;
    long Bound = ceil(8.48528136 * sqrt(sigma2));
    uint64_t count = 0;
    while (1){
#if ADJUST_SIGMA_WHEN_TOOSMALL
        count++;
        if (count > 16777216ULL) {
            count = 0;
            sigma2 = 0.5/isigma2;
            isigma2 *= 0.5;
            fprintf(stderr, "[Warning] DGS1d::discrete_gaussian: sigma2 = %f too small, adjust to (mu, sigma2) = (%f, %f).\n", sigma2, mu, 0.5/isigma2);
        }
#endif
        long x = this->Uniform_u64() % (2 * Bound + 1);
        x -= Bound;
        double p = exp(-(x - bias) * (x - bias) * isigma2);
        if (this->Uniform_u64() / POWER264 < p) return x + mu0;
    }
}

double *NaiveDGS::gen_vec(double sigma2, long ind_l, long ind_r, int sr){
    if (this->L->get_gso_status() != GSO_COMPUTED_QP) this->L->compute_gso_QP();
    long d = ind_r - ind_l;
    double **b = this->L->get_b().hi;
    double *B = this->L->get_B().hi;
    double **miu = this->L->get_miu().hi;

    long *coeff = (long *) NEW_VEC(d+ind_l, sizeof(long));
    double *res = (double *) NEW_VEC(d+ind_l, sizeof(double));
    double *vec = (double *) NEW_VEC(this->L->NumCols(), sizeof(double));

    for (long i = ind_r - 1; i >= ind_l; i--){
        coeff[i] = this->baseDGS->discrete_gaussian(res[i], sigma2/B[i]);
        for (long j = 0; j < i; j++) res[j] -= coeff[i] * miu[i][j];
        red(vec, b[i], -coeff[i], this->L->NumCols());
    }
    if (sr){
        for (long i = ind_l-1; i >= 0; i--){
            coeff[i] = round(res[i]);
            for (long j = 0; j < i; j++) res[j] -= coeff[i] * miu[i][j];
            red(vec, b[i], -coeff[i], this->L->NumCols());
        }
    }
    FREE_VEC(res);
    FREE_VEC(coeff);
    return vec;
}
long *NaiveDGS::gen_coeff(double sigma2, long ind_l, long ind_r){
    if (this->L->get_gso_status() != GSO_COMPUTED_QP) this->L->compute_gso_QP();
    long d = ind_r - ind_l;
    double *B = this->L->get_B().hi;
    double **miu = this->L->get_miu().hi;

    long *coeff = (long *) NEW_VEC(d, sizeof(long));
    double *res = (double *) NEW_VEC(d, sizeof(double));

    for (long i = d - 1; i >= 0; i--){
        coeff[i] = this->baseDGS->discrete_gaussian(res[i], sigma2/B[i+ind_l]);
        for (long j = 0; j < i; j++) res[j] -= coeff[i] * miu[i+ind_l][j+ind_l];
    }

    FREE_VEC(res);
    return coeff;
}

uint64_t Uniform_u64(){
    return gen64();
}

NTL::Vec<double> random_vec(long n, double l){
    DGS1d R;
    NTL::Vec<double> v;
    v.SetLength(n);
    for (long i = 0; i < n; i++){
        v[i] = R.discrete_gaussian(0.0, 42950988369.0);
    }
    double xx = length(v);
    xx = l / xx;
    for (long i = 0; i < n; i++) v[i] *= xx;
    return v;
}
NTL::Mat<double> random_vec(long n, double l, long num){
    DGS1d R;
    NTL::Mat<double> v;
    v.SetDims(num, n);
    for (long i = 0; i < num; i++){
        for (long j = 0; j < n; j++) v[i][j] = R.discrete_gaussian(0.0, 42950988369.0);
        double xx = length(v[i]);
        xx = l / xx;
        for (long j = 0; j < n; j++) v[i][j] *= xx;
    }
    
    return v;
}
Lattice_QP *random_qary_lattice(long dim, long q, long m){
    Lattice_QP *L = new Lattice_QP(dim, dim);
    long i = 0;
    for (; i < m; i++) L->get_b().hi[i][i] = q;
    for (; i < dim; i++) {
        L->get_b().hi[i][i] = 1.0;
        for (long j = 0; j < m; j++) L->get_b().hi[i][j] = gen64()%q;
    }
    return L;
}


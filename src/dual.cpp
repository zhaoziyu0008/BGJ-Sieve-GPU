#include "../include/lattice.h"
#include "../include/utils.h"
#include "../include/vec.h"
#include "../include/quad.h"


int Lattice_QP::dual_QP(NTL::Mat<NTL::quad_float>& Ld){
    MAT_QP tmp = NEW_MAT_QP(n, m);
    NTL::quad_float *Bi = new NTL::quad_float[n];

    if (!gso_computed) compute_gso_QP();
    for (long i = 0; i < n; i++){
        copy(tmp.hi[i], tmp.lo[i], b_star.hi[i], b_star.lo[i], m8);
        NTL::quad_float x(B.hi[i], B.lo[i]);
        Bi[i] = 1.0/x;
    }
    for (long i = 0; i < n; i++){
        mul(tmp.hi[i], tmp.lo[i], Bi[i], m8);
    }
    for (long i = n-1; i > 0; i--){
        for (long j = i-1; j >= 0; j--){
            NTL::quad_float x(miu.hi[i][j], miu.lo[i][j]);
            red(tmp.hi[j], tmp.lo[j], tmp.hi[i], tmp.lo[i], x, m8);
        }
    }
    Ld.SetDims(n, m);
    for (long i = 0; i < n; i++){
        for (long j = 0; j < m; j++){
            Ld[i][j].hi = tmp.hi[i][j];
            Ld[i][j].lo = tmp.lo[i][j];
        }
    }
    FREE_MAT_QP(tmp);
    delete[] Bi;
    return 1;
}
int Lattice_QP::dual_QP(NTL::Mat<double>& Ld){
    MAT_QP tmp = NEW_MAT_QP(n, m);
    NTL::quad_float *Bi = new NTL::quad_float[n];

    if (!gso_computed) compute_gso_QP();
    for (long i = 0; i < n; i++){
        copy(tmp.hi[i], tmp.lo[i], b_star.hi[i], b_star.lo[i], m8);
        NTL::quad_float x(B.hi[i], B.lo[i]);
        Bi[i] = 1.0/x;
    }
    for (long i = 0; i < n; i++){
        mul(tmp.hi[i], tmp.lo[i], Bi[i], m8);
    }
    for (long i = n-1; i > 0; i--){
        for (long j = i-1; j >= 0; j--){
            NTL::quad_float x(miu.hi[i][j], miu.lo[i][j]);
            red(tmp.hi[j], tmp.lo[j], tmp.hi[i], tmp.lo[i], x, m8);
        }
    }
    Ld.SetDims(n, m);
    for (long i = 0; i < n; i++){
        for (long j = 0; j < m; j++){
            Ld[i][j] = tmp.hi[i][j];
        }
    }
    FREE_MAT_QP(tmp);
    delete[] Bi;
    return 1;
}
Lattice_QP *Lattice_QP::dual_QP(){
    Lattice_QP *L = new Lattice_QP(n, m);
    MAT_QP tmp = L->get_b();
    NTL::quad_float *Bi = new NTL::quad_float[n];

    if (!gso_computed) compute_gso_QP();
    for (long i = 0; i < n; i++){
        copy(tmp.hi[i], tmp.lo[i], b_star.hi[i], b_star.lo[i], m8);
        NTL::quad_float x(B.hi[i], B.lo[i]);
        Bi[i] = 1.0/x;
    }
    for (long i = 0; i < n; i++){
        mul(tmp.hi[i], tmp.lo[i], Bi[i], m8);
    }
    for (long i = n-1; i > 0; i--){
        for (long j = i-1; j >= 0; j--){
            NTL::quad_float x(miu.hi[i][j], miu.lo[i][j]);
            red(tmp.hi[j], tmp.lo[j], tmp.hi[i], tmp.lo[i], x, m8);
        }
    }
    delete[] Bi;
    return L;
}
void Lattice_QP::usd(){
	double *tmp = (double *) NEW_VEC(m, sizeof(double));
	for (long i = 0; i < n/2; i++){
		copy(tmp, b.hi[i], m);
		copy(b.hi[i], b.hi[n-1-i], m);
		copy(b.hi[n-1-i], tmp, m);
		copy(tmp, b.lo[i], m);
		copy(b.lo[i], b.lo[n-1-i], m);
		copy(b.lo[n-1-i], tmp, m);
	}
	FREE_VEC(tmp);
	gso_computed = 0;
}
int Lattice_QP::dual_size_red() {
    Lattice_QP *L_dual = this->dual_QP();
    if (L_dual == NULL) return -1;
    L_dual->usd();
    L_dual->compute_gso_QP();
    for (long i = 0; i < n; i++) {
        for (long j = i - 1; j >= 0; j--) {
            if (fabs(L_dual->miu.hi[i][j]) > 0.5){
                NTL::quad_float q = NTL::to_quad_float(round(L_dual->miu.hi[i][j]));
                red(L_dual->b.hi[i], L_dual->b.lo[i], L_dual->b.hi[j], L_dual->b.lo[j],q,m8);
                red(L_dual->miu.hi[i],L_dual->miu.lo[i],L_dual->miu.hi[j],L_dual->miu.lo[j],q,j+1);
                red(this->b.hi[n-1-j], this->b.lo[n-1-j], this->b.hi[n-1-i], this->b.lo[n-1-i], -q, m8);
                red(this->miu.hi[n-1-j], this->miu.lo[n-1-j], this->miu.hi[n-1-i], this->miu.lo[n-1-i], -q, n-i);
            }
        }
    }
    delete L_dual;
    return 1;
}
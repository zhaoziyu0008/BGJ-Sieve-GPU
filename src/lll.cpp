#include "../include/lattice.h"
#include "../include/vec.h"
#include "../include/utils.h"
#include "../include/quad.h"

#define TRANSTO_USE_QP 1

void Lattice_QP::compute_gso_QP(){
	for (long i = 0; i < n; i++){
		copy(b_star.hi[i], b_star.lo[i], b.hi[i], b.lo[i], m8);
	}
	NTL::quad_float x = dot(b_star.hi[0], b_star.lo[0], b_star.hi[0], b_star.lo[0], m8);
	B.hi[0] = x.hi;
	B.lo[0] = x.lo;
	for (long i = 0; i < n-1; i++){
		x = 1.0/x;
		for (long j = i+1; j < n; j++){
			NTL::quad_float y = dot(b_star.hi[j], b_star.lo[j], b_star.hi[i], b_star.lo[i], m8) * x;
			miu.hi[j][i] = y.hi;
			miu.lo[j][i] = y.lo;
			red(b_star.hi[j], b_star.lo[j], b_star.hi[i], b_star.lo[i], y, m8);
		}
		x = dot(b_star.hi[i+1], b_star.lo[i+1], b_star.hi[i+1], b_star.lo[i+1], m8);
		B.hi[i+1] = x.hi;
		B.lo[i+1] = x.lo;
	}
	for (long i = 0; i < n; i++){
        miu.hi[i][i] = 1.0;
		miu.lo[i][i] = 0.0;
    }
	gso_computed = GSO_COMPUTED_QP;
}
void Lattice_QP::compute_gso_QP(long index_l){
	for (long i = index_l; i < n; i++){
		copy(b_star.hi[i], b_star.lo[i], b.hi[i], b.lo[i], m8);
	}
	NTL::quad_float x;
	for (long i = 0; i < index_l; i++){
		x.hi = B.hi[i];
		x.lo = B.lo[i];
		x = 1.0/x;
		for (long j = index_l; j < n; j++){
			NTL::quad_float y = dot(b_star.hi[j], b_star.lo[j], b_star.hi[i], b_star.lo[i], m8) * x;
			miu.hi[j][i] = y.hi;
			miu.lo[j][i] = y.lo;
			red(b_star.hi[j], b_star.lo[j], b_star.hi[i], b_star.lo[i], y, m8);
		}
	}
	x = dot(b_star.hi[index_l], b_star.lo[index_l], b_star.hi[index_l], b_star.lo[index_l], m8);
	B.hi[index_l] = x.hi;
	B.lo[index_l] = x.lo;
	for (long i = index_l; i < n-1; i++){
		x = 1.0/x;
		for (long j = i+1; j < n; j++){
			NTL::quad_float y = dot(b_star.hi[j], b_star.lo[j], b_star.hi[i], b_star.lo[i], m8) * x;
			miu.hi[j][i] = y.hi;
			miu.lo[j][i] = y.lo;
			red(b_star.hi[j], b_star.lo[j], b_star.hi[i], b_star.lo[i], y, m8);
		}
		x = dot(b_star.hi[i+1], b_star.lo[i+1], b_star.hi[i+1], b_star.lo[i+1], m8);
		B.hi[i+1] = x.hi;
		B.lo[i+1] = x.lo;
	}
	for (long i = index_l; i < n; i++){
        miu.hi[i][i] = 1.0;
		miu.lo[i][i] = 0.0;
    }
}

int Lattice_QP::trans_to(long ind_l, long ind_r, Lattice_QP *L){
	if (!gso_computed) compute_gso_QP();
	long dim = ind_r - ind_l;
	long **A = (long **)NEW_MAT(dim, dim, sizeof(long));
    bool fpwarn = false;
	bool fperr = false;

    #if TRANSTO_USE_QP
    Lattice_QP *L_copy = new Lattice_QP(L->NumRows(), L->NumCols());
    for (long i = 0; i < L->NumRows(); i++) {
        copy_avx2(L_copy->get_b().hi[i], L->get_b().hi[i], L->NumCols());
        copy_avx2(L_copy->get_b().lo[i], L->get_b().lo[i], L->NumCols());
    }
    double **b_loc_hi = L->get_b().hi;
    double **b_loc_lo = L->get_b().lo;
    NTL::quad_float *B_loc = (NTL::quad_float *) malloc(dim * sizeof(NTL::quad_float));
    for (long j = 0; j < dim; j++) {
        B_loc[j] = NTL::sqrt(NTL::quad_float(B.hi[j+ind_l], B.lo[j+ind_l]));
    }
    for (long j = dim - 1; j >= 0; j--) {
        if (fperr) break;
        NTL::quad_float x = NTL::quad_float(1.0) / B_loc[j];
        for (long i = 0; i < dim; i++) {
            NTL::quad_float y = NTL::quad_float(b_loc_hi[i][j], b_loc_lo[i][j]) * x;
            if (fabs(y.hi - round(y.hi)) > 0.05) {
                fpwarn = true;
                if (fabs(y.hi - round(y.hi)) > 0.2) {
                    fperr = true;
                    break;
                }
            }
            A[i][j] = round(y.hi);
            //to optimize
            for (long k = 0; k < dim; k++){
                NTL::quad_float z = NTL::quad_float(b_loc_hi[i][k], b_loc_lo[i][k]);
                z -= A[i][j] * NTL::quad_float(miu.hi[j+ind_l][k+ind_l], miu.lo[j+ind_l][k+ind_l]) * B_loc[k];
                b_loc_hi[i][k] = z.hi;
                b_loc_lo[i][k] = z.lo;
            }
        }
    }
    free(B_loc);
    if (fpwarn || fperr) {
        char output1[256];
        char output2[256];
        sprintf(output1, ".RAW-%ld-%ld-%lx", ind_l, ind_r, (long)this);
        sprintf(output2, ".DST-%lx", (long)L_copy);
        store(output1);
        L_copy->store(output2);
    }
    delete L_copy;
    #else
    double **b_loc = L->get_b().hi;
	double *B_loc = (double *) NEW_VEC(dim, sizeof(double));
	for (long j = 0; j < dim; j++){
		B_loc[j] = sqrt(B.hi[j+ind_l]);
	}
	for (long j = dim - 1; j >= 0; j--){
		if (fperr) break;
		double x = 1.0 / B_loc[j];
		for (long i = 0; i < dim; i++){
			double y = b_loc[i][j] * x;
			if (fabs(y - round(y)) > 0.05){
				fpwarn = true;
				if (fabs(y - round(y)) > 0.2){
					fperr = true;
					break;
				}
			}
			A[i][j] = round(y);
			//to optimize
			for (long k = 0; k < dim; k++){
				b_loc[i][k] -= A[i][j] * miu.hi[j+ind_l][k+ind_l] * B_loc[k];
			}
		}
	}
    FREE_VEC((void *)B_loc);
    #endif

	if (fperr){
		std::cerr << "[Error] Lattice_QP::trans_to: floating point precision error, aborted.\n";
        return 0;
	}
	if (fpwarn){
		std::cerr << "[Warning] Lattice_QP::trans_to: floating point precision warning.\n";
	}
	trans_by(A, ind_l, ind_r);
	FREE_MAT((void **)A);
	return 1;
}
int Lattice_QP::reconstruct(Lattice_QP *L_src) {
    Lattice_QP *backup = new Lattice_QP(this->NumRows(), this->NumCols());
    for (long i = 0; i < this->NumRows(); i++) {
        copy_avx2(backup->get_b().hi[i], this->get_b().hi[i], this->NumCols());
        copy_avx2(backup->get_b().lo[i], this->get_b().lo[i], this->NumCols());
    }
    int ret = L_src->trans_to(0, this->NumRows(), backup);
    delete backup;
    if (ret) {
        for (long i = 0; i < this->NumRows(); i++) {
            copy_avx2(this->get_b().hi[i], L_src->get_b().hi[i], this->NumCols());
            copy_avx2(this->get_b().lo[i], L_src->get_b().lo[i], this->NumCols());
        }
        this->gso_computed = 0;
    }

    return ret;
}
Lattice_QP *Lattice_QP::b_loc_QP(long ind_l, long ind_r){
	if (!gso_computed) compute_gso_QP();
	long dim = ind_r - ind_l;
	Lattice_QP *zret = new Lattice_QP(dim, dim);
	MAT_QP b_ret = zret->get_b();
	//to optimize
	for (long j = 0; j < dim; j++){
		NTL::quad_float y(B.hi[j+ind_l], B.lo[j+ind_l]);
		y = sqrt(y);
		for (long i = 0; i < dim; i++){
			NTL::quad_float x(miu.hi[i + ind_l][j + ind_l], miu.lo[i + ind_l][j + ind_l]);	
			NTL::quad_float z = x * y;
			b_ret.hi[i][j] = z.hi;
			b_ret.lo[i][j] = z.lo;
		}
	}
	return zret;
}

int Lattice_QP::size_reduce(long index){
	for (long i = index-1; i >= 0; i--){
        LLL_reduce(index, i);
    }
	return 1;
}
int Lattice_QP::size_reduce(long l, long r) {
	for (long i = l; i < r; i++) {
		for (long j = l-1; j >= 0; j--) {
			LLL_reduce(i, j);
		}
	}
	return 1;
}
int Lattice_QP::LLL_QP(double delta, long ind_l, long ind_r){
	if (!gso_computed) compute_gso_QP();
    long k = ind_l;
	long num_swap = 0;

	while (k < ind_r){
		if (k == ind_l){
			for (long l = k-1; l >= 0; l--){
				LLL_reduce(k, l);
			}
			k++;
			continue;
		}
		//can be remove?
		if ((num_swap % 100000)==99999){
			num_swap++;
			compute_gso_QP();
		}
		LLL_reduce(k, k-1);
		if (B.hi[k] < (delta-miu.hi[k][k-1]*miu.hi[k][k-1])*B.hi[k-1]){
			LLL_swap(k);
			num_swap++;
			k--;
		}else{			
			for (long l = k-2; l >=0; l--){
				LLL_reduce(k, l);
			}
			k++;
		}
	}
    gso_computed = 0;
    return 1;
}
int Lattice_QP::LLL_DEEP_QP(double delta, long ind_l, long ind_r){
	if (!gso_computed) compute_gso_QP();
    long k = ind_l;
	long num_swap = 0;
	for (long l = k-1; l >= 0; l--){
		LLL_reduce(k, l);
	}
	k++;
	while (k < ind_r){
		//can be remove?
		if ((num_swap % 100000)==99999){
			num_swap++;
			compute_gso_QP();
		}
		//for stablity, only do this for l >= ind_l?
		for (long l = k-1; l >= 0; l--){
			LLL_reduce(k, l);
		}
		double distance = tri_dot_slow(miu.hi[k]+ind_l, miu.hi[k]+ind_l, B.hi+ind_l, k-ind_l+1);
		for (long j = ind_l; j < k; j++){
			if (distance < B.hi[j]*delta){
				for (long l = k; l > j; l--){
					LLL_swap(l);
				}
				//std::cerr << "insert " << k << " to " << j << std::endl;
				k = j;
				if (k == ind_l){
					for (long l = k-1; l >= 0; l--){
						LLL_reduce(k, l);
					}
				}
				break;
			}else{
				distance -= miu.hi[k][j]*miu.hi[k][j]*B.hi[j];
			}
		}
		k++;
	}
    gso_computed = 0;
    return 1;
}
int Lattice_QP::size_reduce(){
	for (long i = 1; i < n; i++){
		size_reduce(i);
	}
	return 1;
}
int Lattice_QP::LLL_QP(double delta){
	return LLL_QP(delta, 0, n);
}
int Lattice_QP::LLL_DEEP_QP(double delta){
	return LLL_DEEP_QP(delta, 0, n);
}

void Lattice_QP::LLL_reduce(long k, long l){
	if (fabs(miu.hi[k][l]) > 0.5){
		NTL::quad_float q = NTL::to_quad_float(round(miu.hi[k][l]));
		red(b.hi[k],b.lo[k],b.hi[l],b.lo[l],q,m8);
		red(miu.hi[k],miu.lo[k],miu.hi[l],miu.lo[l],q,l+1);
	}
}
void Lattice_QP::LLL_swap(long k){
	double *tmp_hi = b.hi[k];
	double *tmp_lo = b.lo[k];
	b.hi[k] = b.hi[k-1];
	b.lo[k] = b.lo[k-1];
	b.hi[k-1] = tmp_hi;
	b.lo[k-1] = tmp_lo;

	double *tmpp_hi = miu.hi[k];
	double *tmpp_lo = miu.lo[k];
	miu.hi[k] = miu.hi[k-1];
	miu.lo[k] = miu.lo[k-1];
	miu.hi[k-1] = tmpp_hi;
	miu.lo[k-1] = tmpp_lo;

	NTL::quad_float mu(miu.hi[k-1][k-1], miu.lo[k-1][k-1]);
	NTL::quad_float ttt(B.hi[k-1], B.lo[k-1]);
	NTL::quad_float BBB(B.hi[k], B.lo[k]);
	BBB += mu * mu * ttt;
	ttt /= BBB;

	NTL::quad_float tmp(B.hi[k],B.lo[k]);
	tmp *= ttt;
	B.hi[k] = tmp.hi;
	B.lo[k] = tmp.lo;
	tmp = mu * ttt;
	miu.hi[k][k-1] = tmp.hi;
	miu.lo[k][k-1] = tmp.lo;
	B.hi[k-1] = BBB.hi;
	B.lo[k-1] = BBB.lo;

	long l = n-k-1;
	tmp = -tmp;
	VEC_QP muk = NEW_VEC_QP(l);
	VEC_QP muk_ = NEW_VEC_QP(l);
	for (long s = k+1; s < n; s++){
		muk.hi[s-k-1] = miu.hi[s][k];
		muk.lo[s-k-1] = miu.lo[s][k];
		muk_.hi[s-k-1] = miu.hi[s][k-1];
		muk_.lo[s-k-1] = miu.lo[s][k-1];
	}
	red(muk_, muk, mu, l);			//this is new miu[s][k]
	red(muk, muk_, tmp, l);			//this is new miu[s][k-1]
	for (long s = k+1; s < n; s++){
		miu.hi[s][k] = muk_.hi[s-k-1];
		miu.lo[s][k] = muk_.lo[s-k-1];
		miu.hi[s][k-1] = muk.hi[s-k-1];
		miu.lo[s][k-1] = muk.lo[s-k-1];
	}
	FREE_VEC_QP(muk);
	FREE_VEC_QP(muk_);

	/*for (long s = k+1; s < n;s++){
		ttt = miu[s][k];
		miu[s][k] = miu[s][k-1] - mu * ttt;
		miu[s][k-1] = ttt + tmp * miu[s][k];
	}*/

    miu.hi[k-1][k-1] = 1.0;
	miu.lo[k-1][k-1] = 0.0;
    miu.hi[k-1][k] = 0.0;
	miu.lo[k-1][k] = 0.0;
    miu.hi[k][k] = 1.0;
	miu.lo[k][k] = 0.0;
}
void Lattice_QP::trans_by(long **A, long index_l, long index_r){
	long dim = index_r - index_l;
	MAT_QP tmp = NEW_MAT_QP(dim, m8);
	for (long i = 0; i < dim; i++){
		for (long j = 0; j < dim; j++){
			NTL::quad_float q(-A[i][j]);
			red(tmp.hi[i], tmp.lo[i], b.hi[j + index_l], b.lo[j + index_l], q, m8);
		}
	}
	for (long i = 0; i < dim; i++){
		copy(b.hi[i + index_l], b.lo[i + index_l], tmp.hi[i], tmp.lo[i], m8);
	}
	FREE_MAT_QP(tmp);
	gso_computed = 0;
}
void Lattice_QP::trans_by(double **A, long index_l, long index_r){
	long dim = index_r - index_l;
	bool fperr = false;
	bool fpwarn = false;
	for (long i = 0; i < dim; i++){
		if (fperr) break;
		for (long j = 0; j < dim; j++){
			if (fabs(A[i][j] - round(A[i][j])) > 0.05) {
				fpwarn = true;
				if (fabs(A[i][j] - round(A[i][j])) > 0.2){
					fperr = true;
					break;
				}
			}
		}
	}
	if (fperr) {
		std::cerr << "[Error] Lattice_QP::transformby: floating point error, aborted!\n";
		return;
	}
	if (fpwarn) {
		std::cerr << "[Warning] Lattice_QP::transformby: floating point warning!\n";
	}
	MAT_QP tmp = NEW_MAT_QP(dim, m8);
	for (long i = 0; i < dim; i++){
		for (long j = 0; j < dim; j++){
			NTL::quad_float q(-round(A[i][j]));
			red(tmp.hi[i], tmp.lo[i], b.hi[j + index_l], b.lo[j + index_l], q, m8);
		}
	}
	for (long i = 0; i < dim; i++){
		copy(b.hi[i + index_l], b.lo[i + index_l], tmp.hi[i], tmp.lo[i], m8);
	}
	FREE_MAT_QP(tmp);
	gso_computed = 0;
}
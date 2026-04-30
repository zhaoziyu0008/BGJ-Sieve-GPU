#include "../include/utils.h"
#include "../include/vec.h"


void **NEW_MAT(long n, long m, long size){
    long M = ((m * size - 1) / 64 + 1) * 64;

    void **ptr = new void*[n+1];
    ptr[0] = calloc (n * M + 64, 1);
    if (ptr[0] == NULL) return NULL;
    void *ptr_start = (void *) ((((long)(ptr[0])-1)/64+1)*64);
    for (long i = 0; i < n; i++){
        ptr[i+1] = (void *)((long)ptr_start + M * i);
    }
    return &ptr[1];
}
void *NEW_VEC(long n, long size){
    long M = ((n * size - 1) / 64 + 1) * 64;
    void *ptr = calloc (M + 64 + 8, 1);
    if (ptr == NULL) return NULL;
    void *ptr_start = (void *) ((((long)(ptr)+7)/64 + 1) * 64);
    void **pptr = (void **)((long)ptr_start - 8);
    pptr[0] = ptr;
    return ptr_start;
}
void FREE_MAT(void **ptr){
    ptr -= 1;
    free(ptr[0]);
    delete[] ptr;
}
void FREE_VEC(void *ptr){
    long pptr = ((long*)ptr)[-1];
    free((void *) pptr);
}


VEC_QP NEW_VEC_QP(long n){
    VEC_QP ret;
    ret.hi = (double *)NEW_VEC(n, sizeof(double));
    ret.lo = (double *)NEW_VEC(n, sizeof(double));
    return ret;
}
MAT_QP NEW_MAT_QP(long n, long m){
    MAT_QP ret;
    ret.hi = (double **)NEW_MAT(n, m, sizeof(double));
    ret.lo = (double **)NEW_MAT(n, m, sizeof(double));
    return ret;
}
void FREE_VEC_QP(VEC_QP vq){
    FREE_VEC(vq.hi);
    FREE_VEC(vq.lo);
}
void FREE_MAT_QP(MAT_QP mq){
    FREE_MAT((void **)mq.hi);
    FREE_MAT((void **)mq.lo);
}

double gh_coeff(long n){
    double a = 1.0;
    if (n % 2 == 0){
        long m = n/2;
        for (long i = 1; i < m + 1; i++){
            a *= pow(i, 1.0 / n);
        }
    }else{
        long m = (n - 1) / 2;
        for (long i = 0; i < m + 1; i++){
            a *= pow(i + 0.5,1.0 / n);
        }
        a *= pow(3.14159265357989324,0.5 / n);
    }
    a /= sqrt(3.14159265357989324);
    return a;
}
double pot(double detn, long n, long blocksize){
    static const double lg2_hkz60[60] = {
        0.9997048107870072, 0.9888638766629481, 0.9562663489009612, 0.9319398107243426, 0.909156793172772,
        0.8803454368707301, 0.8435075691264613, 0.8128331581835154, 0.7831677539426122, 0.755991831521689,
        0.7471182699111201, 0.6961654432904928, 0.6643027209316872, 0.6497071417249046, 0.598309722947419,
        0.5682305937217957, 0.5526302700248892, 0.4983718955048759, 0.4732039241738746, 0.422447856817323, 
        0.4126873776493885, 0.3708719445216065, 0.3171451449693261, 0.2843094054674554, 0.256299314268641, 
        0.2189843706821939, 0.1939248745950290, 0.1482889181738402, 0.1105088674778068, 0.074771219098067, 
        0.0255207855804460,-0.0191835228211998,-0.0488911620203144,-0.0914465657064089,-0.135940194374428, 
        -0.1591103541091127,-0.2127974667090677,-0.2507680846759403,-0.2889139179358686,-0.325297912864509, 
        -0.3770203233561157,-0.4287773463558759,-0.4640664741943877,-0.5080055431217686,-0.548198587016565,
        -0.5865048708456515,-0.6367355533270002,-0.6705125689673702,-0.7226340430239490,-0.760355605697006, 
        -0.8072697063689745,-0.8403953930555745,-0.8677829624847366,-0.9264530695756915,-0.984041648117367,
        -1.0242485168137299,-1.0436644439143734,-1.0938330797395408,-1.1382621093309973,-1.184466424901700,
    };  // average lg2 of sqrt(B) for random hkz-reduced basis 

    double *lg2_dist_vec = new double[n];
    double lg2_det = 0.0;
    for (long i = 0; i < blocksize - 60; i++){
        lg2_dist_vec[i+n-blocksize] = lg2_det / (blocksize - i) + log2(gh_coeff(blocksize-i));
        lg2_det -= lg2_dist_vec[i+n-blocksize];
    }
    lg2_det /= 60.0;
    for (long i = 0; i < 60; i++){
        lg2_dist_vec[i+n-60] = lg2_hkz60[i] + lg2_det;
    }
    lg2_det = 0.0;
    double lg2_gg = log2(gh_coeff(blocksize))*(blocksize/(blocksize-1.0));
    for (long i = 0; i < n - blocksize; i++){
        lg2_det -= lg2_dist_vec[n-i-1];
        long ind = n - blocksize - i - 1;
        lg2_dist_vec[ind] = lg2_det /(blocksize-1) + lg2_gg;
        lg2_det += lg2_dist_vec[ind];
    }
    double bias = 0.0;
    for (long i = 0; i < n; i++){
        bias += lg2_dist_vec[i];
    }
    bias /= n;
    bias -= log2(detn);
    for (long i = 0; i < n; i++){
        lg2_dist_vec[i] -= bias;
    }

    double ret = 0.0;
    for (long i = 0; i < n; i++){
        ret += (n-i) * lg2_dist_vec[i];
    }
    delete[] lg2_dist_vec;
    ret *= 2.0;
    return ret;
}

void int_inv(long **Ai, double **src, long n){
    long n8 = ((n+7)/8)*8;
    double *miu_store = (double *) calloc (sizeof(double)*n*n8 + 64, 1);
    double *b_star_store = (double *) calloc (sizeof(double)*n*n8 + 64, 1);
    double *miu_store_start = (double *) ((((long)(miu_store)-1)/64+1)*64);
    double *b_star_store_start = (double *) ((((long)(b_star_store)-1)/64+1)*64);
    double *B = new double[n];
    double **miu = new double*[n];
    double **b_star = new double*[n];
    for (long i = 0; i < n; i++){
        miu[i] = miu_store_start + i * n8;
        b_star[i] = b_star_store_start + i * n8;
    }
    for (long i = 0; i < n; i++){
        for (long j = 0; j < n; j++){
            b_star[i][j] = src[i][j];
        }
    }

    B[0] = 1/dot(b_star[0], b_star[0], n8);
    for (long i = 0; i < n-1; i++){
        for (long j = i+1; j < n; j++){
            miu[j][i] = dot(b_star[j], b_star[i], n8)*B[i];
            red(b_star[j], b_star[i], miu[j][i], n8);
        }
        B[i+1] = 1/dot(b_star[i+1], b_star[i+1], n8);
    }
    for (long i = 0; i < n; i++){
        mul(b_star[i], B[i], n8);
    }
    for (long i = n-1; i > 0; i--){
        for (long j = i-1; j >= 0; j--){
            red(b_star[j], b_star[i], miu[i][j], n8);
        }
    }
    
    for (long i = 0; i < n; i++){
        for (long j = 0; j < n; j++){
            Ai[j][i] = round(b_star[i][j]);
        }
    }

    delete[] B;
    delete[] miu;
    delete[] b_star;
    free(miu_store);
    free(b_star_store);
}

double dot(NTL::Vec<double>& src1, NTL::Vec<double>& src2){
    double ret = 0.0;
    long n = src1.length();
    for (long i = 0; i < n; i++){
        ret += src1[i] * src2[i];
    }
    return ret;
}
double length(NTL::Vec<double>& src){
    return sqrt(dot(src, src));
}

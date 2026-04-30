#ifndef __UTILS_H
#define __UTILS_H

#include <NTL/LLL.h>
#include "quad.h"
#include "lattice.h"

#ifndef _CEIL8
#define _CEIL8(_x) (((_x) + 7) & (~7))
#else
#error _CEIL8 is already defined
#endif

#ifndef _CEIL64
#define _CEIL64(_x) (((_x) + 63) & (~63))
#else
#error _CEIL64 is already defined
#endif

#define PTRCEIL64(_ptr) ((((long)(_ptr) - 1) / 64 + 1) * 64)

#define PRINT_VEC(_vec, _m) do {            \
    std::cout << "[";                       \
    for (long __i = 0; __i < _m-1; __i++){  \
        std::cout << _vec[__i] << " ";      \
    }                                       \
    if (_m > 0) {                           \
        std::cout << _vec[_m-1] << "]\n";   \
    } else {                                \
        std::cout << "]\n";                 \
    }                                       \
} while (0)

#define PRINT_MAT(__mat, _n, _m) do {                               \
    std::cout << "[";                                               \
    for (long _im = 0; _im < _n; _im++) PRINT_VEC(__mat[_im], _m);  \
    std::cout << "]\n";                                             \
} while (0)

/* create an aligned Matrix/Vector, each element has size bytes
 * for example if we want to create an aligned double matrix/vec,
 * use the following:
 *    double **mat = (double **)NEW_MAT(NumRows, NumCols, sizeof(double));
 *    double *vec = (double *)NEW_VEC(length, sizeof(double));
 * if the length or NumCols above can not divided by 8, several zeros
 * will be added in the tail, so for example if you want to compute
 * the dot product of the i-th and j-th row of a mat, use the following:
 *    dot(mat[i], mat[j], NumCols)
 * fanally you need to free them by:
 *    FREE_MAT(mat);
 *    FREE_VEC(vec);
*/
void **NEW_MAT(long n, long m, long size);
void *NEW_VEC(long n, long size);
void FREE_MAT(void **ptr);
void FREE_VEC(void *ptr);
inline void FREE_MAT(double **ptr){FREE_MAT((void **)ptr);}
inline void FREE_VEC(double *ptr){FREE_VEC((void *)ptr);}
inline void FREE_MAT(float **ptr){FREE_MAT((void **)ptr);}
inline void FREE_VEC(float *ptr){FREE_VEC((void *)ptr);}
inline void FREE_MAT(long **ptr){FREE_MAT((void **)ptr);}
inline void FREE_VEC(long *ptr){FREE_VEC((void *)ptr);}

/* create an aligned NTL::quad_float Matrix/Vector. */
VEC_QP NEW_VEC_QP(long n);
MAT_QP NEW_MAT_QP(long n, long m);
void FREE_VEC_QP(VEC_QP vq);
void FREE_MAT_QP(MAT_QP mq);


/* some functions about lattice. */
//return gamma(n/2+1)^(1/n)/sqrt(pi)
double gh_coeff(long n);                            
//return the expect log(pot) of a n-dim lattice with det = detn^n after BKZ-blocksize 
double pot(double detn, long n, long blocksize);    
void int_inv(long **Ai, double **src, long n);

/* ntl vec operations */
double dot(NTL::Vec<double>& src1, NTL::Vec<double>& src2);
double length(NTL::Vec<double>& src);
#endif
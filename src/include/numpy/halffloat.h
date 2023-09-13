#ifndef NPYMATH_HALFFLOAT_H_
#define NPYMATH_HALFFLOAT_H_

#include <Python.h>
#include "numpy/npy_math.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Half-precision routines
 */

/* Conversions */
float npy_half_to_float(npymath_half h);
double npy_half_to_double(npymath_half h);
npymath_half npy_float_to_half(float f);
npymath_half npy_double_to_half(double d);
/* Comparisons */
int npy_half_eq(npymath_half h1, npymath_half h2);
int npy_half_ne(npymath_half h1, npymath_half h2);
int npy_half_le(npymath_half h1, npymath_half h2);
int npy_half_lt(npymath_half h1, npymath_half h2);
int npy_half_ge(npymath_half h1, npymath_half h2);
int npy_half_gt(npymath_half h1, npymath_half h2);
/* faster *_nonan variants for when you know h1 and h2 are not NaN */
int npy_half_eq_nonan(npymath_half h1, npymath_half h2);
int npy_half_lt_nonan(npymath_half h1, npymath_half h2);
int npy_half_le_nonan(npymath_half h1, npymath_half h2);
/* Miscellaneous functions */
int npy_half_iszero(npymath_half h);
int npy_half_isnan(npymath_half h);
int npy_half_isinf(npymath_half h);
int npy_half_isfinite(npymath_half h);
int npy_half_signbit(npymath_half h);
npymath_half npy_half_copysign(npymath_half x, npymath_half y);
npymath_half npy_half_spacing(npymath_half h);
npymath_half npy_half_nextafter(npymath_half x, npymath_half y);
npymath_half npy_half_divmod(npymath_half x, npymath_half y, npymath_half *modulus);

/*
 * Half-precision constants
 */

#define NPY_HALF_ZERO   (0x0000u)
#define NPY_HALF_PZERO  (0x0000u)
#define NPY_HALF_NZERO  (0x8000u)
#define NPY_HALF_ONE    (0x3c00u)
#define NPY_HALF_NEGONE (0xbc00u)
#define NPY_HALF_PINF   (0x7c00u)
#define NPY_HALF_NINF   (0xfc00u)
#define NPY_HALF_NAN    (0x7e00u)

#define NPY_MAX_HALF    (0x7bffu)

/*
 * Bit-level conversions
 */

npymath_uint16 npy_floatbits_to_halfbits(npymath_uint32 f);
npymath_uint16 npy_doublebits_to_halfbits(npymath_uint64 d);
npymath_uint32 npy_halfbits_to_floatbits(npymath_uint16 h);
npymath_uint64 npy_halfbits_to_doublebits(npymath_uint16 h);

#ifdef __cplusplus
}
#endif

#endif  /* NPYMATH_HALFFLOAT_H_ */

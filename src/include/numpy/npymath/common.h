#ifndef NPYMATH_COMMON_H_
#define NPYMATH_COMMON_H_

/* need Python.h for Py_intptr_t */
#include <Python.h>

/* numpconfig.h is auto-generated */
#include "numpy/npymath/config.h"
#ifdef NPYMATH_HAVE_CONFIG_H
#include "numpy/npymath/block.h"
#endif

/*
 * using static inline modifiers when defining npy_math functions
 * allows the compiler to make optimizations when possible
 */
#ifndef NPYMATH_INLINE_MATH
#if defined(NPYMATH_INTERNAL_BUILD) && NPYMATH_INTERNAL_BUILD
    #define NPYMATH_INLINE_MATH 1
#else
    #define NPYMATH_INLINE_MATH 0
#endif
#endif

#ifndef NPY_INLINE_MATH
#if defined(NPY_INTERNAL_BUILD) && NPY_INTERNAL_BUILD
    #define NPY_INLINE_MATH 1
#else
    #define NPY_INLINE_MATH 0
#endif
#endif


/*
 * give a hint to the compiler which branch is more likely or unlikely
 * to occur, e.g. rare error cases:
 *
 * if (NPYMATH_UNLIKELY(failure == 0))
 *    return NULL;
 *
 * the double !! is to cast the expression (e.g. NULL) to a boolean required by
 * the intrinsic
 */
#ifdef NPYMATH_HAVE___BUILTIN_EXPECT
#define NPYMATH_LIKELY(x) __builtin_expect(!!(x), 1)
#define NPYMATH_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define NPYMATH_LIKELY(x) (x)
#define NPYMATH_UNLIKELY(x) (x)
#endif

#ifdef PY_LONG_LONG
typedef PY_LONG_LONG npymath_longlong;
typedef unsigned PY_LONG_LONG npymath_ulonglong;
#else
typedef long npymath_longlong;
typedef unsigned long npymath_ulonglong;
#endif

/*
 * `NPYMATH_SIZEOF_LONGDOUBLE` isn't usually equal to sizeof(long double).
 * In some certain cases, it may forced to be equal to sizeof(double)
 * even against the compiler implementation and the same goes for
 * `complex long double`.
 *
 * Therefore, avoid `long double`, use `npymath_longdouble` instead,
 * and when it comes to standard math functions make sure of using
 * the double version when `NPYMATH_SIZEOF_LONGDOUBLE` == `NPYMATH_SIZEOF_DOUBLE`.
 * For example:
 *   npymath_longdouble *ptr, x;
 *   #if NPYMATH_SIZEOF_LONGDOUBLE == NPYMATH_SIZEOF_DOUBLE
 *       npymath_longdouble r = modf(x, ptr);
 *   #else
 *       npymath_longdouble r = modfl(x, ptr);
 *   #endif
 *
 * See https://github.com/numpy/numpy/issues/20348
 */
#if NPYMATH_SIZEOF_LONGDOUBLE == NPYMATH_SIZEOF_DOUBLE
    #define longdouble_t double
    typedef double npymath_longdouble;
#else
    #define longdouble_t long double
    typedef long double npymath_longdouble;
#endif

#ifndef Py_USING_UNICODE
#error Must use Python with unicode enabled.
#endif


typedef signed char npymath_byte;
typedef unsigned char npymath_ubyte;
typedef unsigned short npymath_ushort;
typedef unsigned int npymath_uint;
typedef unsigned long npymath_ulong;

/* These are for completeness */
typedef short npymath_short;
typedef int npymath_int;
typedef long npymath_long;
typedef float npymath_float;
typedef double npymath_double;

#if defined(__cplusplus)

#ifdef NUMPY_BUILD

typedef struct _npy_cdouble npymath_cdouble;
typedef struct _npy_cfloat npymath_cfloat;
typedef struct _npy_clongdouble npymath_clongdouble;

#else

typedef struct
{
    double _Val[2];
} npymath_cdouble;

typedef struct
{
    float _Val[2];
} npymath_cfloat;

typedef struct
{
    long double _Val[2];
} npymath_clongdouble;

#endif

#else

#include <complex.h>

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
typedef _Dcomplex npymath_cdouble;
typedef _Fcomplex npymath_cfloat;
typedef _Lcomplex npymath_clongdouble;
#else /* !defined(_MSC_VER) || defined(__INTEL_COMPILER) */
typedef double _Complex npymath_cdouble;
typedef float _Complex npymath_cfloat;
typedef longdouble_t _Complex npymath_clongdouble;
#endif

#endif

        /* Need to find the number of bits for each type and
           make definitions accordingly.

           C states that sizeof(char) == 1 by definition

           So, just using the sizeof keyword won't help.

           It also looks like Python itself uses sizeof(char) quite a
           bit, which by definition should be 1 all the time.

           Idea: Make Use of CHAR_BIT which should tell us how many
           BITS per CHARACTER
        */

        /* Include platform definitions -- These are in the C89/90 standard */
#include <limits.h>

#define NPYMATH_SIZEOF_BYTE 1

#define NPYMATH_BITSOF_CHAR CHAR_BIT
#define NPYMATH_BITSOF_BYTE (NPYMATH_SIZEOF_BYTE * CHAR_BIT)
#define NPYMATH_BITSOF_SHORT (NPYMATH_SIZEOF_SHORT * CHAR_BIT)
#define NPYMATH_BITSOF_INT (NPYMATH_SIZEOF_INT * CHAR_BIT)
#define NPYMATH_BITSOF_LONG (NPYMATH_SIZEOF_LONG * CHAR_BIT)
#define NPYMATH_BITSOF_LONGLONG (NPYMATH_SIZEOF_LONGLONG * CHAR_BIT)
#define NPYMATH_BITSOF_FLOAT (NPYMATH_SIZEOF_FLOAT * CHAR_BIT)
#define NPYMATH_BITSOF_DOUBLE (NPYMATH_SIZEOF_DOUBLE * CHAR_BIT)
#define NPYMATH_BITSOF_LONGDOUBLE (NPYMATH_SIZEOF_LONGDOUBLE * CHAR_BIT)

#if NPYMATH_BITSOF_LONG == 8
#define NPYMATH_INT8 NPY_LONG
        typedef unsigned long npymath_uint8;
#elif NPYMATH_BITSOF_LONG == 16
#define NPYMATH_INT16 NPY_LONG
        typedef long npymath_int16;
        typedef unsigned long npymath_uint16;
#elif NPYMATH_BITSOF_LONG == 32
#define NPYMATH_INT32 NPY_LONG
        typedef long npymath_int32;
        typedef unsigned long npymath_uint32;
#elif NPYMATH_BITSOF_LONG == 64
#define NPYMATH_INT64 NPY_LONG
        typedef long npymath_int64;
        typedef unsigned long npymath_uint64;
#endif

#if NPYMATH_BITSOF_LONGLONG == 8
#  ifndef NPYMATH_INT8
#    define NPYMATH_INT8 NPY_LONGLONG
        typedef npymath_ulonglong npymath_uint8;
#  endif
#elif NPYMATH_BITSOF_LONGLONG == 16
#  ifndef NPYMATH_INT16
#    define NPYMATH_INT16 NPY_LONGLONG
        typedef npymath_longlong npymath_int16;
        typedef npymath_ulonglong npymath_uint16;
#  endif
#elif NPYMATH_BITSOF_LONGLONG == 32
#  ifndef NPYMATH_INT32
#    define NPYMATH_INT32 NPY_LONGLONG
        typedef npymath_longlong npymath_int32;
        typedef npymath_ulonglong npymath_uint32;
#  endif
#elif NPYMATH_BITSOF_LONGLONG == 64
#  ifndef NPYMATH_INT64
#    define NPYMATH_INT64 NPY_LONGLONG
        typedef npymath_longlong npymath_int64;
        typedef npymath_ulonglong npymath_uint64;
#  endif
#endif

#if NPYMATH_BITSOF_INT == 8
#ifndef NPYMATH_INT8
#define NPYMATH_INT8 NPY_INT
        typedef unsigned int npymath_uint8;
#endif
#elif NPYMATH_BITSOF_INT == 16
#ifndef NPYMATH_INT16
#define NPYMATH_INT16 NPY_INT
        typedef int npymath_int16;
        typedef unsigned int npymath_uint16;
#endif
#elif NPYMATH_BITSOF_INT == 32
#ifndef NPYMATH_INT32
#define NPYMATH_INT32 NPY_INT
        typedef int npymath_int32;
        typedef unsigned int npymath_uint32;
#endif
#elif NPYMATH_BITSOF_INT == 64
#ifndef NPYMATH_INT64
#define NPYMATH_INT64 NPY_INT
        typedef int npymath_int64;
        typedef unsigned int npymath_uint64;
#endif
#endif

#if NPYMATH_BITSOF_SHORT == 8
#ifndef NPYMATH_INT8
#define NPYMATH_INT8 NPY_SHORT
        typedef unsigned short npymath_uint8;
#endif
#elif NPYMATH_BITSOF_SHORT == 16
#ifndef NPYMATH_INT16
#define NPYMATH_INT16 NPY_SHORT
        typedef short npymath_int16;
        typedef unsigned short npymath_uint16;
#endif
#elif NPYMATH_BITSOF_SHORT == 32
#ifndef NPYMATH_INT32
#define NPYMATH_INT32 NPY_SHORT
        typedef short npymath_int32;
        typedef unsigned short npymath_uint32;
#endif
#elif NPYMATH_BITSOF_SHORT == 64
#ifndef NPYMATH_INT64
#define NPYMATH_INT64 NPY_SHORT
        typedef short npymath_int64;
        typedef unsigned short npymath_uint64;
#endif
#endif


#if NPYMATH_BITSOF_CHAR == 8
#ifndef NPYMATH_INT8
#define NPYMATH_INT8 NPY_BYTE
        typedef unsigned char npymath_uint8;
#endif
#elif NPYMATH_BITSOF_CHAR == 16
#ifndef NPYMATH_INT16
#define NPYMATH_INT16 NPY_BYTE
        typedef signed char npymath_int16;
        typedef unsigned char npymath_uint16;
#endif
#elif NPYMATH_BITSOF_CHAR == 32
#ifndef NPYMATH_INT32
#define NPYMATH_INT32 NPY_BYTE
        typedef signed char npymath_int32;
        typedef unsigned char npymath_uint32;
#endif
#elif NPYMATH_BITSOF_CHAR == 64
#ifndef NPYMATH_INT64
#define NPYMATH_INT64 NPY_BYTE
        typedef signed char npymath_int64;
        typedef unsigned char npymath_uint64;
#endif
#endif

/* half/float16 isn't a floating-point type in C */
#define NPY_FLOAT16 NPY_HALF
typedef npymath_uint16 npymath_half;

/* End of typedefs for numarray style bit-width names */

#endif  /* NPYMATH_COMMON_H_ */

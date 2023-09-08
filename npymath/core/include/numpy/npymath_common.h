#ifndef NPYMATH_COMMON_H_
#define NPYMATH_COMMON_H_

/* need Python.h for npy_intp */
#include <Python.h>

/* numpconfig.h is auto-generated */
#include "npymathconfig.h"
#ifdef HAVE_NPY_CONFIG_H
#include <npymath_config.h>
#endif

/*
 * using static inline modifiers when defining npy_math functions
 * allows the compiler to make optimizations when possible
 */
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
#ifdef HAVE___BUILTIN_EXPECT
#define NPYMATH_LIKELY(x) __builtin_expect(!!(x), 1)
#define NPYMATH_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define NPYMATH_LIKELY(x) (x)
#define NPYMATH_UNLIKELY(x) (x)
#endif

/*
 * This is to typedef npy_intp to the appropriate pointer size for this
 * platform.  Py_intptr_t, Py_uintptr_t are defined in pyport.h.
 */
typedef Py_intptr_t npy_intp;

#ifdef PY_LONG_LONG
typedef PY_LONG_LONG npy_longlong;
typedef unsigned PY_LONG_LONG npy_ulonglong;
#else
typedef long npy_longlong;
typedef unsigned long npy_ulonglong;
#endif

/*
 * `NPYMATH_SIZEOF_LONGDOUBLE` isn't usually equal to sizeof(long double).
 * In some certain cases, it may forced to be equal to sizeof(double)
 * even against the compiler implementation and the same goes for
 * `complex long double`.
 *
 * Therefore, avoid `long double`, use `npy_longdouble` instead,
 * and when it comes to standard math functions make sure of using
 * the double version when `NPYMATH_SIZEOF_LONGDOUBLE` == `NPYMATH_SIZEOF_DOUBLE`.
 * For example:
 *   npy_longdouble *ptr, x;
 *   #if NPYMATH_SIZEOF_LONGDOUBLE == NPYMATH_SIZEOF_DOUBLE
 *       npy_longdouble r = modf(x, ptr);
 *   #else
 *       npy_longdouble r = modfl(x, ptr);
 *   #endif
 *
 * See https://github.com/numpy/numpy/issues/20348
 */
#if NPYMATH_SIZEOF_LONGDOUBLE == NPYMATH_SIZEOF_DOUBLE
    #define longdouble_t double
    typedef double npy_longdouble;
#else
    #define longdouble_t long double
    typedef long double npy_longdouble;
#endif

#ifndef Py_USING_UNICODE
#error Must use Python with unicode enabled.
#endif


typedef signed char npy_byte;
typedef unsigned char npy_ubyte;
typedef unsigned short npy_ushort;
typedef unsigned int npy_uint;
typedef unsigned long npy_ulong;

/* These are for completeness */
typedef char npy_char;
typedef short npy_short;
typedef int npy_int;
typedef long npy_long;
typedef float npy_float;
typedef double npy_double;

#ifdef __cplusplus
extern "C++" {
#endif
#include <complex.h>
#ifdef __cplusplus
}
#endif

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER) && defined(__cplusplus)
typedef struct 
{
    double _Val[2];
} npy_cdouble;

typedef struct
{
    float _Val[2];
} npy_cfloat;

typedef struct
{
    long double _Val[2];
} npy_clongdouble;
#elif defined(_MSC_VER) && !defined(__INTEL_COMPILER) /* && !defined(__cplusplus) */
typedef _Dcomplex npy_cdouble;
typedef _Fcomplex npy_cfloat;
typedef _Lcomplex npy_clongdouble;
#else /* !defined(_MSC_VER) || defined(__INTEL_COMPILER) */
typedef double _Complex npy_cdouble;
typedef float _Complex npy_cfloat;
typedef longdouble_t _Complex npy_clongdouble;
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
#define NPY_INT8 NPY_LONG
#define NPY_UINT8 NPY_ULONG
        typedef long npy_int8;
        typedef unsigned long npy_uint8;
#elif NPYMATH_BITSOF_LONG == 16
#define NPY_INT16 NPY_LONG
#define NPY_UINT16 NPY_ULONG
        typedef long npy_int16;
        typedef unsigned long npy_uint16;
#elif NPYMATH_BITSOF_LONG == 32
#define NPY_INT32 NPY_LONG
#define NPY_UINT32 NPY_ULONG
        typedef long npy_int32;
        typedef unsigned long npy_uint32;
        typedef unsigned long npy_ucs4;
#elif NPYMATH_BITSOF_LONG == 64
#define NPY_INT64 NPY_LONG
#define NPY_UINT64 NPY_ULONG
        typedef long npy_int64;
        typedef unsigned long npy_uint64;
#define MyPyLong_FromInt64 PyLong_FromLong
#define MyPyLong_AsInt64 PyLong_AsLong
#elif NPYMATH_BITSOF_LONG == 128
#define NPY_INT128 NPY_LONG
#define NPY_UINT128 NPY_ULONG
        typedef long npy_int128;
        typedef unsigned long npy_uint128;
#endif

#if NPYMATH_BITSOF_LONGLONG == 8
#  ifndef NPY_INT8
#    define NPY_INT8 NPY_LONGLONG
#    define NPY_UINT8 NPY_ULONGLONG
        typedef npy_longlong npy_int8;
        typedef npy_ulonglong npy_uint8;
#  endif
#  define NPY_MAX_LONGLONG NPY_MAX_INT8
#  define NPY_MIN_LONGLONG NPY_MIN_INT8
#  define NPY_MAX_ULONGLONG NPY_MAX_UINT8
#elif NPYMATH_BITSOF_LONGLONG == 16
#  ifndef NPY_INT16
#    define NPY_INT16 NPY_LONGLONG
#    define NPY_UINT16 NPY_ULONGLONG
        typedef npy_longlong npy_int16;
        typedef npy_ulonglong npy_uint16;
#  endif
#  define NPY_MAX_LONGLONG NPY_MAX_INT16
#  define NPY_MIN_LONGLONG NPY_MIN_INT16
#  define NPY_MAX_ULONGLONG NPY_MAX_UINT16
#elif NPYMATH_BITSOF_LONGLONG == 32
#  ifndef NPY_INT32
#    define NPY_INT32 NPY_LONGLONG
#    define NPY_UINT32 NPY_ULONGLONG
        typedef npy_longlong npy_int32;
        typedef npy_ulonglong npy_uint32;
        typedef npy_ulonglong npy_ucs4;
#  endif
#  define NPY_MAX_LONGLONG NPY_MAX_INT32
#  define NPY_MIN_LONGLONG NPY_MIN_INT32
#  define NPY_MAX_ULONGLONG NPY_MAX_UINT32
#elif NPYMATH_BITSOF_LONGLONG == 64
#  ifndef NPY_INT64
#    define NPY_INT64 NPY_LONGLONG
#    define NPY_UINT64 NPY_ULONGLONG
        typedef npy_longlong npy_int64;
        typedef npy_ulonglong npy_uint64;
#    define MyPyLong_FromInt64 PyLong_FromLongLong
#    define MyPyLong_AsInt64 PyLong_AsLongLong
#  endif
#  define NPY_MAX_LONGLONG NPY_MAX_INT64
#  define NPY_MIN_LONGLONG NPY_MIN_INT64
#  define NPY_MAX_ULONGLONG NPY_MAX_UINT64
#elif NPYMATH_BITSOF_LONGLONG == 128
#  ifndef NPY_INT128
#    define NPY_INT128 NPY_LONGLONG
#    define NPY_UINT128 NPY_ULONGLONG
        typedef npy_longlong npy_int128;
        typedef npy_ulonglong npy_uint128;
#  endif
#  define NPY_MAX_LONGLONG NPY_MAX_INT128
#  define NPY_MIN_LONGLONG NPY_MIN_INT128
#  define NPY_MAX_ULONGLONG NPY_MAX_UINT128
#elif NPYMATH_BITSOF_LONGLONG == 256
#  define NPY_INT256 NPY_LONGLONG
#  define NPY_UINT256 NPY_ULONGLONG
        typedef npy_longlong npy_int256;
        typedef npy_ulonglong npy_uint256;
#  define NPY_MAX_LONGLONG NPY_MAX_INT256
#  define NPY_MIN_LONGLONG NPY_MIN_INT256
#  define NPY_MAX_ULONGLONG NPY_MAX_UINT256
#endif

#if NPYMATH_BITSOF_INT == 8
#ifndef NPY_INT8
#define NPY_INT8 NPY_INT
#define NPY_UINT8 NPY_UINT
        typedef int npy_int8;
        typedef unsigned int npy_uint8;
#endif
#elif NPYMATH_BITSOF_INT == 16
#ifndef NPY_INT16
#define NPY_INT16 NPY_INT
#define NPY_UINT16 NPY_UINT
        typedef int npy_int16;
        typedef unsigned int npy_uint16;
#endif
#elif NPYMATH_BITSOF_INT == 32
#ifndef NPY_INT32
#define NPY_INT32 NPY_INT
#define NPY_UINT32 NPY_UINT
        typedef int npy_int32;
        typedef unsigned int npy_uint32;
        typedef unsigned int npy_ucs4;
#endif
#elif NPYMATH_BITSOF_INT == 64
#ifndef NPY_INT64
#define NPY_INT64 NPY_INT
#define NPY_UINT64 NPY_UINT
        typedef int npy_int64;
        typedef unsigned int npy_uint64;
#    define MyPyLong_FromInt64 PyLong_FromLong
#    define MyPyLong_AsInt64 PyLong_AsLong
#endif
#elif NPYMATH_BITSOF_INT == 128
#ifndef NPY_INT128
#define NPY_INT128 NPY_INT
#define NPY_UINT128 NPY_UINT
        typedef int npy_int128;
        typedef unsigned int npy_uint128;
#endif
#endif

#if NPYMATH_BITSOF_SHORT == 8
#ifndef NPY_INT8
#define NPY_INT8 NPY_SHORT
#define NPY_UINT8 NPY_USHORT
        typedef short npy_int8;
        typedef unsigned short npy_uint8;
#endif
#elif NPYMATH_BITSOF_SHORT == 16
#ifndef NPY_INT16
#define NPY_INT16 NPY_SHORT
#define NPY_UINT16 NPY_USHORT
        typedef short npy_int16;
        typedef unsigned short npy_uint16;
#endif
#elif NPYMATH_BITSOF_SHORT == 32
#ifndef NPY_INT32
#define NPY_INT32 NPY_SHORT
#define NPY_UINT32 NPY_USHORT
        typedef short npy_int32;
        typedef unsigned short npy_uint32;
        typedef unsigned short npy_ucs4;
#endif
#elif NPYMATH_BITSOF_SHORT == 64
#ifndef NPY_INT64
#define NPY_INT64 NPY_SHORT
#define NPY_UINT64 NPY_USHORT
        typedef short npy_int64;
        typedef unsigned short npy_uint64;
#    define MyPyLong_FromInt64 PyLong_FromLong
#    define MyPyLong_AsInt64 PyLong_AsLong
#endif
#elif NPYMATH_BITSOF_SHORT == 128
#ifndef NPY_INT128
#define NPY_INT128 NPY_SHORT
#define NPY_UINT128 NPY_USHORT
        typedef short npy_int128;
        typedef unsigned short npy_uint128;
#endif
#endif


#if NPYMATH_BITSOF_CHAR == 8
#ifndef NPY_INT8
#define NPY_INT8 NPY_BYTE
#define NPY_UINT8 NPY_UBYTE
        typedef signed char npy_int8;
        typedef unsigned char npy_uint8;
#endif
#elif NPYMATH_BITSOF_CHAR == 16
#ifndef NPY_INT16
#define NPY_INT16 NPY_BYTE
#define NPY_UINT16 NPY_UBYTE
        typedef signed char npy_int16;
        typedef unsigned char npy_uint16;
#endif
#elif NPYMATH_BITSOF_CHAR == 32
#ifndef NPY_INT32
#define NPY_INT32 NPY_BYTE
#define NPY_UINT32 NPY_UBYTE
        typedef signed char npy_int32;
        typedef unsigned char npy_uint32;
        typedef unsigned char npy_ucs4;
#endif
#elif NPYMATH_BITSOF_CHAR == 64
#ifndef NPY_INT64
#define NPY_INT64 NPY_BYTE
#define NPY_UINT64 NPY_UBYTE
        typedef signed char npy_int64;
        typedef unsigned char npy_uint64;
#    define MyPyLong_FromInt64 PyLong_FromLong
#    define MyPyLong_AsInt64 PyLong_AsLong
#endif
#elif NPYMATH_BITSOF_CHAR == 128
#ifndef NPY_INT128
#define NPY_INT128 NPY_BYTE
#define NPY_UINT128 NPY_UBYTE
        typedef signed char npy_int128;
        typedef unsigned char npy_uint128;
#endif
#endif



#if NPYMATH_BITSOF_DOUBLE == 32
#ifndef NPY_FLOAT32
#define NPY_FLOAT32 NPY_DOUBLE
#define NPY_COMPLEX64 NPY_CDOUBLE
        typedef double npy_float32;
        typedef npy_cdouble npy_complex64;
#endif
#elif NPYMATH_BITSOF_DOUBLE == 64
#ifndef NPY_FLOAT64
#define NPY_FLOAT64 NPY_DOUBLE
#define NPY_COMPLEX128 NPY_CDOUBLE
        typedef double npy_float64;
        typedef npy_cdouble npy_complex128;
#endif
#elif NPYMATH_BITSOF_DOUBLE == 80
#ifndef NPY_FLOAT80
#define NPY_FLOAT80 NPY_DOUBLE
#define NPY_COMPLEX160 NPY_CDOUBLE
        typedef double npy_float80;
        typedef npy_cdouble npy_complex160;
#endif
#elif NPYMATH_BITSOF_DOUBLE == 96
#ifndef NPY_FLOAT96
#define NPY_FLOAT96 NPY_DOUBLE
#define NPY_COMPLEX192 NPY_CDOUBLE
        typedef double npy_float96;
        typedef npy_cdouble npy_complex192;
#endif
#elif NPYMATH_BITSOF_DOUBLE == 128
#ifndef NPY_FLOAT128
#define NPY_FLOAT128 NPY_DOUBLE
#define NPY_COMPLEX256 NPY_CDOUBLE
        typedef double npy_float128;
        typedef npy_cdouble npy_complex256;
#endif
#endif



#if NPYMATH_BITSOF_FLOAT == 32
#ifndef NPY_FLOAT32
#define NPY_FLOAT32 NPY_FLOAT
#define NPY_COMPLEX64 NPY_CFLOAT
        typedef float npy_float32;
        typedef npy_cfloat npy_complex64;
#endif
#elif NPYMATH_BITSOF_FLOAT == 64
#ifndef NPY_FLOAT64
#define NPY_FLOAT64 NPY_FLOAT
#define NPY_COMPLEX128 NPY_CFLOAT
        typedef float npy_float64;
        typedef npy_cfloat npy_complex128;
#endif
#elif NPYMATH_BITSOF_FLOAT == 80
#ifndef NPY_FLOAT80
#define NPY_FLOAT80 NPY_FLOAT
#define NPY_COMPLEX160 NPY_CFLOAT
        typedef float npy_float80;
        typedef npy_cfloat npy_complex160;
#endif
#elif NPYMATH_BITSOF_FLOAT == 96
#ifndef NPY_FLOAT96
#define NPY_FLOAT96 NPY_FLOAT
#define NPY_COMPLEX192 NPY_CFLOAT
        typedef float npy_float96;
        typedef npy_cfloat npy_complex192;
#endif
#elif NPYMATH_BITSOF_FLOAT == 128
#ifndef NPY_FLOAT128
#define NPY_FLOAT128 NPY_FLOAT
#define NPY_COMPLEX256 NPY_CFLOAT
        typedef float npy_float128;
        typedef npy_cfloat npy_complex256;
#endif
#endif

/* half/float16 isn't a floating-point type in C */
#define NPY_FLOAT16 NPY_HALF
typedef npy_uint16 npy_half;
typedef npy_half npy_float16;

#if NPYMATH_BITSOF_LONGDOUBLE == 32
#ifndef NPY_FLOAT32
#define NPY_FLOAT32 NPY_LONGDOUBLE
#define NPY_COMPLEX64 NPY_CLONGDOUBLE
        typedef npy_longdouble npy_float32;
        typedef npy_clongdouble npy_complex64;
#endif
#elif NPYMATH_BITSOF_LONGDOUBLE == 64
#ifndef NPY_FLOAT64
#define NPY_FLOAT64 NPY_LONGDOUBLE
#define NPY_COMPLEX128 NPY_CLONGDOUBLE
        typedef npy_longdouble npy_float64;
        typedef npy_clongdouble npy_complex128;
#endif
#elif NPYMATH_BITSOF_LONGDOUBLE == 80
#ifndef NPY_FLOAT80
#define NPY_FLOAT80 NPY_LONGDOUBLE
#define NPY_COMPLEX160 NPY_CLONGDOUBLE
        typedef npy_longdouble npy_float80;
        typedef npy_clongdouble npy_complex160;
#endif
#elif NPYMATH_BITSOF_LONGDOUBLE == 96
#ifndef NPY_FLOAT96
#define NPY_FLOAT96 NPY_LONGDOUBLE
#define NPY_COMPLEX192 NPY_CLONGDOUBLE
        typedef npy_longdouble npy_float96;
        typedef npy_clongdouble npy_complex192;
#endif
#elif NPYMATH_BITSOF_LONGDOUBLE == 128
#ifndef NPY_FLOAT128
#define NPY_FLOAT128 NPY_LONGDOUBLE
#define NPY_COMPLEX256 NPY_CLONGDOUBLE
        typedef npy_longdouble npy_float128;
        typedef npy_clongdouble npy_complex256;
#endif
#elif NPYMATH_BITSOF_LONGDOUBLE == 256
#define NPY_FLOAT256 NPY_LONGDOUBLE
#define NPY_COMPLEX512 NPY_CLONGDOUBLE
        typedef npy_longdouble npy_float256;
        typedef npy_clongdouble npy_complex512;
#endif

/* End of typedefs for numarray style bit-width names */

#endif  /* NPYMATH_COMMON_H_ */

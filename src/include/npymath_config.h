#ifndef NPYMATH_CONFIG_H_
#define NPYMATH_CONFIG_H_

#include "npymath_meson_config.h"
#include "npymathconfig.h"
#include "utils.h"

/* blocklist */

/* Disable broken functions on z/OS */
#if defined (__MVS__)

#define NPY_BLOCK_POWF
#define NPY_BLOCK_EXPF
#undef NPYMATH_HAVE___THREAD

#endif

/* Disable broken MS math functions */
#if defined(__MINGW32_VERSION)

#define NPY_BLOCK_ATAN2
#define NPY_BLOCK_ATAN2F
#define NPY_BLOCK_ATAN2L

#define NPY_BLOCK_HYPOT
#define NPY_BLOCK_HYPOTF
#define NPY_BLOCK_HYPOTL

#endif

#if defined(_MSC_VER)

#undef NPYMATH_HAVE_CASIN
#undef NPYMATH_HAVE_CASINF
#undef NPYMATH_HAVE_CASINL
#undef NPYMATH_HAVE_CASINH
#undef NPYMATH_HAVE_CASINHF
#undef NPYMATH_HAVE_CASINHL
#undef NPYMATH_HAVE_CATAN
#undef NPYMATH_HAVE_CATANF
#undef NPYMATH_HAVE_CATANL
#undef NPYMATH_HAVE_CATANH
#undef NPYMATH_HAVE_CATANHF
#undef NPYMATH_HAVE_CATANHL
#undef NPYMATH_HAVE_CSQRT
#undef NPYMATH_HAVE_CSQRTF
#undef NPYMATH_HAVE_CSQRTL
#undef NPYMATH_HAVE_CLOG
#undef NPYMATH_HAVE_CLOGF
#undef NPYMATH_HAVE_CLOGL
#undef NPYMATH_HAVE_CACOS
#undef NPYMATH_HAVE_CACOSF
#undef NPYMATH_HAVE_CACOSL
#undef NPYMATH_HAVE_CACOSH
#undef NPYMATH_HAVE_CACOSHF
#undef NPYMATH_HAVE_CACOSHL

#endif

/* MSVC _hypot messes with fp precision mode on 32-bit, see gh-9567 */
#if defined(_MSC_VER) && !defined(_WIN64)

#undef NPYMATH_HAVE_CABS
#undef NPYMATH_HAVE_CABSF
#undef NPYMATH_HAVE_CABSL

#define NPY_BLOCK_HYPOT
#define NPY_BLOCK_HYPOTF
#define NPY_BLOCK_HYPOTL

#endif


/* Intel C for Windows uses POW for 64 bits longdouble*/
#if defined(_MSC_VER) && defined(__INTEL_COMPILER)
#if NPYMATH_SIZEOF_LONGDOUBLE == 8
#define NPY_BLOCK_POWL
#endif
#endif /* defined(_MSC_VER) && defined(__INTEL_COMPILER) */

/* powl gives zero division warning on OS X, see gh-8307 */
#if defined(__APPLE__)
#define NPY_BLOCK_POWL
#endif

#ifdef __CYGWIN__
/* Loss of precision */
#undef NPYMATH_HAVE_CASINHL
#undef NPYMATH_HAVE_CASINH
#undef NPYMATH_HAVE_CASINHF

/* Loss of precision */
#undef NPYMATH_HAVE_CATANHL
#undef NPYMATH_HAVE_CATANH
#undef NPYMATH_HAVE_CATANHF

/* Loss of precision and branch cuts */
#undef NPYMATH_HAVE_CATANL
#undef NPYMATH_HAVE_CATAN
#undef NPYMATH_HAVE_CATANF

/* Branch cuts */
#undef NPYMATH_HAVE_CACOSHF
#undef NPYMATH_HAVE_CACOSH

/* Branch cuts */
#undef NPYMATH_HAVE_CSQRTF
#undef NPYMATH_HAVE_CSQRT

/* Branch cuts and loss of precision */
#undef NPYMATH_HAVE_CASINF
#undef NPYMATH_HAVE_CASIN
#undef NPYMATH_HAVE_CASINL

/* Branch cuts */
#undef NPYMATH_HAVE_CACOSF
#undef NPYMATH_HAVE_CACOS

/* log2(exp2(i)) off by a few eps */
#define NPY_BLOCK_LOG2

/* np.power(..., dtype=np.complex256) doesn't report overflow */
#undef NPYMATH_HAVE_CPOWL
#undef NPYMATH_HAVE_CEXPL

#include <cygwin/version.h>
#if CYGWIN_VERSION_DLL_MAJOR < 3003
// rather than blocklist cabsl, hypotl, modfl, sqrtl, error out
#error cygwin < 3.3 not supported, please update
#endif
#endif

/* Disable broken gnu trig functions */
#if defined(NPYMATH_HAVE_FEATURES_H)
#include <features.h>

#if defined(__GLIBC__)
#if !__GLIBC_PREREQ(2, 18)

#undef NPYMATH_HAVE_CASIN
#undef NPYMATH_HAVE_CASINF
#undef NPYMATH_HAVE_CASINL
#undef NPYMATH_HAVE_CASINH
#undef NPYMATH_HAVE_CASINHF
#undef NPYMATH_HAVE_CASINHL
#undef NPYMATH_HAVE_CATAN
#undef NPYMATH_HAVE_CATANF
#undef NPYMATH_HAVE_CATANL
#undef NPYMATH_HAVE_CATANH
#undef NPYMATH_HAVE_CATANHF
#undef NPYMATH_HAVE_CATANHL
#undef NPYMATH_HAVE_CACOS
#undef NPYMATH_HAVE_CACOSF
#undef NPYMATH_HAVE_CACOSL
#undef NPYMATH_HAVE_CACOSH
#undef NPYMATH_HAVE_CACOSHF
#undef NPYMATH_HAVE_CACOSHL

#endif  /* __GLIBC_PREREQ(2, 18) */
#else   /* defined(__GLIBC) */
/* musl linux?, see issue #25092 */

#undef HAVE_CASIN
#undef HAVE_CASINF
#undef HAVE_CASINL
#undef HAVE_CASINH
#undef HAVE_CASINHF
#undef HAVE_CASINHL
#undef HAVE_CATAN
#undef HAVE_CATANF
#undef HAVE_CATANL
#undef HAVE_CATANH
#undef HAVE_CATANHF
#undef HAVE_CATANHL
#undef HAVE_CACOS
#undef HAVE_CACOSF
#undef HAVE_CACOSL
#undef HAVE_CACOSH
#undef HAVE_CACOSHF
#undef HAVE_CACOSHL

#endif  /* defined(__GLIBC) */
#endif  /* defined(NPYMATH_HAVE_FEATURES_H) */

#endif  /* NPYMATH_CONFIG_H_ */

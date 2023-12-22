#ifndef NPYMATHCONFIG_H_
#define NPYMATHCONFIG_H_

#include "numpy/npymath/_config.h"

/*
 * On Mac OS X, because there is only one configuration stage for all the archs
 * in universal builds, any macro which depends on the arch needs to be
 * hardcoded.
 *
 * Note that distutils/pip will attempt a universal2 build when Python itself
 * is built as universal2, hence this hardcoding is needed even if we do not
 * support universal2 wheels anymore (see gh-22796).
 * This code block can be removed after we have dropped the setup.py based
 * build completely.
 */
#ifdef __APPLE__
    #undef NPYMATH_SIZEOF_LONG

    #ifdef __LP64__
        #define NPYMATH_SIZEOF_LONG         8
    #else
        #define NPYMATH_SIZEOF_LONG         4
    #endif

    #undef NPYMATH_SIZEOF_LONGDOUBLE
    #undef NPYMATH_SIZEOF_COMPLEX_LONGDOUBLE
    #ifdef NPYMATH_HAVE_LDOUBLE_IEEE_DOUBLE_LE
      #undef NPYMATH_HAVE_LDOUBLE_IEEE_DOUBLE_LE
    #endif
    #ifdef NPYMATH_HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE
      #undef NPYMATH_HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE
    #endif

    #if defined(__arm64__)
        #define NPYMATH_SIZEOF_LONGDOUBLE         8
        #define NPYMATH_SIZEOF_COMPLEX_LONGDOUBLE 16
        #define NPYMATH_HAVE_LDOUBLE_IEEE_DOUBLE_LE 1
    #elif defined(__x86_64)
        #define NPYMATH_SIZEOF_LONGDOUBLE         16
        #define NPYMATH_SIZEOF_COMPLEX_LONGDOUBLE 32
        #define NPYMATH_HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE 1
    #elif defined (__i386)
        #define NPYMATH_SIZEOF_LONGDOUBLE         12
        #define NPYMATH_SIZEOF_COMPLEX_LONGDOUBLE 24
    #elif defined(__ppc__) || defined (__ppc64__)
        #define NPYMATH_SIZEOF_LONGDOUBLE         16
        #define NPYMATH_SIZEOF_COMPLEX_LONGDOUBLE 32
    #else
        #error "unknown architecture"
    #endif
#endif

#endif  /* NPYMATHCONFIG_H_ */

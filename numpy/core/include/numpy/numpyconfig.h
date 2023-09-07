#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_NUMPYCONFIG_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_NUMPYCONFIG_H_

#include "_numpyconfig.h"

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
    #undef NPY_SIZEOF_LONG

    #ifdef __LP64__
        #define NPY_SIZEOF_LONG         8
    #else
        #define NPY_SIZEOF_LONG         4
    #endif

    #undef NPY_SIZEOF_LONGDOUBLE
    #undef NPY_SIZEOF_COMPLEX_LONGDOUBLE
    #ifdef HAVE_LDOUBLE_IEEE_DOUBLE_LE
      #undef HAVE_LDOUBLE_IEEE_DOUBLE_LE
    #endif
    #ifdef HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE
      #undef HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE
    #endif

    #if defined(__arm64__)
        #define NPY_SIZEOF_LONGDOUBLE         8
        #define NPY_SIZEOF_COMPLEX_LONGDOUBLE 16
        #define HAVE_LDOUBLE_IEEE_DOUBLE_LE 1
    #elif defined(__x86_64)
        #define NPY_SIZEOF_LONGDOUBLE         16
        #define NPY_SIZEOF_COMPLEX_LONGDOUBLE 32
        #define HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE 1
    #elif defined (__i386)
        #define NPY_SIZEOF_LONGDOUBLE         12
        #define NPY_SIZEOF_COMPLEX_LONGDOUBLE 24
    #elif defined(__ppc__) || defined (__ppc64__)
        #define NPY_SIZEOF_LONGDOUBLE         16
        #define NPY_SIZEOF_COMPLEX_LONGDOUBLE 32
    #else
        #error "unknown architecture"
    #endif
#endif

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NPY_NUMPYCONFIG_H_ */

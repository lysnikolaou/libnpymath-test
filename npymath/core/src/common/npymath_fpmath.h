#ifndef NPYMATH_FPMATH_H_
#define NPYMATH_FPMATH_H_

#include "npymath_config.h"

#include "numpy/npymath_cpu.h"
#include "numpy/npymath_common.h"

#if !(defined(HAVE_LDOUBLE_IEEE_QUAD_BE) || \
      defined(HAVE_LDOUBLE_IEEE_QUAD_LE) || \
      defined(HAVE_LDOUBLE_IEEE_DOUBLE_LE) || \
      defined(HAVE_LDOUBLE_IEEE_DOUBLE_BE) || \
      defined(HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE) || \
      defined(HAVE_LDOUBLE_INTEL_EXTENDED_12_BYTES_LE) || \
      defined(HAVE_LDOUBLE_MOTOROLA_EXTENDED_12_BYTES_BE) || \
      defined(HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_BE) || \
      defined(HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_LE))
    #error No long double representation defined
#endif

/* for back-compat, also keep old name for double-double */
#ifdef HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_LE
    #define HAVE_LDOUBLE_DOUBLE_DOUBLE_LE
#endif
#ifdef HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_BE
    #define HAVE_LDOUBLE_DOUBLE_DOUBLE_BE
#endif

#endif  /* NPYMATH_FPMATH_H_ */

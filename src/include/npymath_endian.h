#ifndef NPYMATH_ENDIAN_H_
#define NPYMATH_ENDIAN_H_

/*
 * NPYMATH_BYTE_ORDER is set to the same value as BYTE_ORDER set by glibc in
 * endian.h
 */

#if defined(NPYMATH_HAVE_ENDIAN_H) || defined(NPYMATH_HAVE_SYS_ENDIAN_H)
    /* Use endian.h if available */

    #if defined(NPYMATH_HAVE_ENDIAN_H)
    #include <endian.h>
    #elif defined(NPYMATH_HAVE_SYS_ENDIAN_H)
    #include <sys/endian.h>
    #endif

    #if defined(BYTE_ORDER) && defined(BIG_ENDIAN) && defined(LITTLE_ENDIAN)
        #define NPYMATH_BYTE_ORDER    BYTE_ORDER
        #define NPYMATH_LITTLE_ENDIAN LITTLE_ENDIAN
        #define NPYMATH_BIG_ENDIAN    BIG_ENDIAN
    #elif defined(_BYTE_ORDER) && defined(_BIG_ENDIAN) && defined(_LITTLE_ENDIAN)
        #define NPYMATH_BYTE_ORDER    _BYTE_ORDER
        #define NPYMATH_LITTLE_ENDIAN _LITTLE_ENDIAN
        #define NPYMATH_BIG_ENDIAN    _BIG_ENDIAN
    #elif defined(__BYTE_ORDER) && defined(__BIG_ENDIAN) && defined(__LITTLE_ENDIAN)
        #define NPYMATH_BYTE_ORDER    __BYTE_ORDER
        #define NPYMATH_LITTLE_ENDIAN __LITTLE_ENDIAN
        #define NPYMATH_BIG_ENDIAN    __BIG_ENDIAN
    #endif
#endif

#ifndef NPYMATH_BYTE_ORDER
    /* Set endianness info using target CPU */
    #include "npymath_cpu.h"

    #define NPYMATH_LITTLE_ENDIAN 1234
    #define NPYMATH_BIG_ENDIAN 4321

    #if defined(NPYMATH_CPU_X86)                  \
            || defined(NPYMATH_CPU_AMD64)         \
            || defined(NPYMATH_CPU_IA64)          \
            || defined(NPYMATH_CPU_ALPHA)         \
            || defined(NPYMATH_CPU_ARMEL)         \
            || defined(NPYMATH_CPU_ARMEL_AARCH32) \
            || defined(NPYMATH_CPU_ARMEL_AARCH64) \
            || defined(NPYMATH_CPU_SH_LE)         \
            || defined(NPYMATH_CPU_MIPSEL)        \
            || defined(NPYMATH_CPU_PPC64LE)       \
            || defined(NPYMATH_CPU_ARCEL)         \
            || defined(NPYMATH_CPU_RISCV64)       \
            || defined(NPYMATH_CPU_LOONGARCH)     \
            || defined(NPYMATH_CPU_WASM)
        #define NPYMATH_BYTE_ORDER NPYMATH_LITTLE_ENDIAN

    #elif defined(NPYMATH_CPU_PPC)                \
            || defined(NPYMATH_CPU_SPARC)         \
            || defined(NPYMATH_CPU_S390)          \
            || defined(NPYMATH_CPU_HPPA)          \
            || defined(NPYMATH_CPU_PPC64)         \
            || defined(NPYMATH_CPU_ARMEB)         \
            || defined(NPYMATH_CPU_ARMEB_AARCH32) \
            || defined(NPYMATH_CPU_ARMEB_AARCH64) \
            || defined(NPYMATH_CPU_SH_BE)         \
            || defined(NPYMATH_CPU_MIPSEB)        \
            || defined(NPYMATH_CPU_OR1K)          \
            || defined(NPYMATH_CPU_M68K)          \
            || defined(NPYMATH_CPU_ARCEB)
        #define NPYMATH_BYTE_ORDER NPYMATH_BIG_ENDIAN

    #else
        #error Unknown CPU: can not set endianness
    #endif

#endif

#endif  /* NPYMATH_ENDIAN_H_ */

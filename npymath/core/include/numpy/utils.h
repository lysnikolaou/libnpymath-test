#ifndef NPYMATH_UTILS_H_
#define NPYMATH_UTILS_H_

#ifndef __COMP_NPYMATH_UNUSED
    #if defined(__GNUC__)
        #define __COMP_NPYMATH_UNUSED __attribute__ ((__unused__))
    #elif defined(__ICC)
        #define __COMP_NPYMATH_UNUSED __attribute__ ((__unused__))
    #elif defined(__clang__)
        #define __COMP_NPYMATH_UNUSED __attribute__ ((unused))
    #else
        #define __COMP_NPYMATH_UNUSED
    #endif
#endif

/* Use this to tag a variable as not used. It will remove unused variable
 * warning on support platforms (see __COM_NPYMATH_UNUSED) and mangle the variable
 * to avoid accidental use */
#define NPYMATH_UNUSED(x) __NPYMATH_UNUSED_TAGGED ## x __COMP_NPYMATH_UNUSED

#define NPYMATH_CAT__(a, b) a ## b
#define NPYMATH_CAT_(a, b) NPYMATH_CAT__(a, b)
#define NPYMATH_CAT(a, b) NPYMATH_CAT_(a, b)

#endif  /* NPYMATH_UTILS_H_ */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "numpy/halffloat.h"

/*
 * This chooses between 'ties to even' and 'ties away from zero'.
 */
#define NPY_HALF_ROUND_TIES_TO_EVEN 1
/*
 * If these are 1, the conversions try to trigger underflow,
 * overflow, and invalid exceptions in the FP system when needed.
 */
#define NPY_HALF_GENERATE_OVERFLOW 1
#define NPY_HALF_GENERATE_UNDERFLOW 1
#define NPY_HALF_GENERATE_INVALID 1

/*
 ********************************************************************
 *                   HALF-PRECISION ROUTINES                        *
 ********************************************************************
 */

float npy_half_to_float(npymath_half h)
{
#if defined(NPYMATH_HAVE_FP16)
    float ret;
    _mm_store_ss(&ret, _mm_cvtph_ps(_mm_cvtsi32_si128(bits_)));
    return ret;
#elif defined(NPYMATH_HAVE_VSX3) && defined(vec_extract_fp_from_shorth)
    return vec_extract(vec_extract_fp_from_shorth(vec_splats(bits_)), 0);
#elif defined(NPYMATH_HAVE_VSX3) && defined(NPYMATH_HAVE_VSX_ASM)
    __vector float vf32;
    __asm__ __volatile__("xvcvhpsp %x0,%x1"
                            : "=wa"(vf32)
                            : "wa"(vec_splats(bits_.u)));
    return vec_extract(vf32, 0);
#else
    union { float ret; npymath_uint32 retbits; } conv;
    conv.retbits = npy_halfbits_to_floatbits(h);
    return conv.ret;
#endif
}

double npy_half_to_double(npymath_half h)
{
#if defined(NPYMATH_HAVE_AVX512FP16)
    double ret;
    _mm_store_sd(&ret, _mm_cvtph_pd(_mm_castsi128_ph(_mm_cvtsi32_si128(bits_))));
    return ret;
#elif defined(NPYMATH_HAVE_VSX3) && defined(NPYMATH_HAVE_VSX_ASM)
    __vector float vf64;
    __asm__ __volatile__("xvcvhpdp %x0,%x1"
                            : "=wa"(vf32)
                            : "wa"(vec_splats(bits_)));
    return vec_extract(vf64, 0);
#else
    union { double ret; npymath_uint64 retbits; } conv;
    conv.retbits = npy_halfbits_to_doublebits(h);
    return conv.ret;
#endif
}

npymath_half npy_float_to_half(float f)
{
#if defined(NPYMATH_HAVE_FP16)
    __m128 mf = _mm_load_ss(&f);
    return _mm_cvts i128_si32(_mm_cvtps_ph(mf, _MM_FROUND_TO_NEAREST_INT));
#elif defined(NPYMATH_HAVE_VSX3) && defined(NPYMATH_HAVE_VSX_ASM)
    __vector float vf32 = vec_splats(f);
    __vector unsigned short vf16;
    __asm__ __volatile__ ("xvcvsphp %x0,%x1" : "=wa" (vf16) : "wa" (vf32));
    return vec_extract(vf16, 0);
#else
    union { float f; npymath_uint32 fbits; } conv;
    conv.f = f;
    return npy_floatbits_to_halfbits(conv.fbits);
#endif
    
}

npymath_half npy_double_to_half(double d)
{
#if defined(NPYMATH_HAVE_AVX512FP16)
    __m128d md = _mm_load_sd(&f);
    return _mm_cvtsi128_si32(_mm_castph_si128(_mm_cvtpd_ph(md)));
#elif defined(NPYMATH_HAVE_VSX3) && defined(NPYMATH_HAVE_VSX_ASM)
    __vector double vf64 = vec_splats(f);
    __vector unsigned short vf16;
    __asm__ __volatile__ ("xvcvdphp %x0,%x1" : "=wa" (vf16) : "wa" (vf64));
    return vec_extract(vf16, 0);
#else
    union { double d; npymath_uint64 dbits; } conv;
    conv.d = d;
    return npy_doublebits_to_halfbits(conv.dbits);
#endif
}

int npy_half_iszero(npymath_half h)
{
    return (h&0x7fff) == 0;
}

int npy_half_isnan(npymath_half h)
{
    return ((h&0x7c00u) == 0x7c00u) && ((h&0x03ffu) != 0x0000u);
}

int npy_half_isinf(npymath_half h)
{
    return ((h&0x7fffu) == 0x7c00u);
}

int npy_half_isfinite(npymath_half h)
{
    return ((h&0x7c00u) != 0x7c00u);
}

int npy_half_signbit(npymath_half h)
{
    return (h&0x8000u) != 0;
}

npymath_half npy_half_spacing(npymath_half h)
{
    npymath_half ret;
    npymath_uint16 h_exp = h&0x7c00u;
    npymath_uint16 h_sig = h&0x03ffu;
    if (h_exp == 0x7c00u) {
#if NPY_HALF_GENERATE_INVALID
        npy_set_floatstatus_invalid();
#endif
        ret = NPY_HALF_NAN;
    } else if (h == 0x7bffu) {
#if NPY_HALF_GENERATE_OVERFLOW
        npy_set_floatstatus_overflow();
#endif
        ret = NPY_HALF_PINF;
    } else if ((h&0x8000u) && h_sig == 0) { /* Negative boundary case */
        if (h_exp > 0x2c00u) { /* If result is normalized */
            ret = h_exp - 0x2c00u;
        } else if(h_exp > 0x0400u) { /* The result is a subnormal, but not the smallest */
            ret = 1 << ((h_exp >> 10) - 2);
        } else {
            ret = 0x0001u; /* Smallest subnormal half */
        }
    } else if (h_exp > 0x2800u) { /* If result is still normalized */
        ret = h_exp - 0x2800u;
    } else if (h_exp > 0x0400u) { /* The result is a subnormal, but not the smallest */
        ret = 1 << ((h_exp >> 10) - 1);
    } else {
        ret = 0x0001u;
    }

    return ret;
}

npymath_half npy_half_copysign(npymath_half x, npymath_half y)
{
    return (x&0x7fffu) | (y&0x8000u);
}

npymath_half npy_half_nextafter(npymath_half x, npymath_half y)
{
    npymath_half ret;

    if (npy_half_isnan(x) || npy_half_isnan(y)) {
        ret = NPY_HALF_NAN;
    } else if (npy_half_eq_nonan(x, y)) {
        ret = x;
    } else if (npy_half_iszero(x)) {
        ret = (y&0x8000u) + 1; /* Smallest subnormal half */
    } else if (!(x&0x8000u)) { /* x > 0 */
        if ((npymath_int16)x > (npymath_int16)y) { /* x > y */
            ret = x-1;
        } else {
            ret = x+1;
        }
    } else {
        if (!(y&0x8000u) || (x&0x7fffu) > (y&0x7fffu)) { /* x < y */
            ret = x-1;
        } else {
            ret = x+1;
        }
    }
#if NPY_HALF_GENERATE_OVERFLOW
    if (npy_half_isinf(ret) && npy_half_isfinite(x)) {
        npy_set_floatstatus_overflow();
    }
#endif

    return ret;
}

int npy_half_eq_nonan(npymath_half h1, npymath_half h2)
{
    return (h1 == h2 || ((h1 | h2) & 0x7fff) == 0);
}

int npy_half_eq(npymath_half h1, npymath_half h2)
{
    /*
     * The equality cases are as follows:
     *   - If either value is NaN, never equal.
     *   - If the values are equal, equal.
     *   - If the values are both signed zeros, equal.
     */
    return (!npy_half_isnan(h1) && !npy_half_isnan(h2)) &&
           (h1 == h2 || ((h1 | h2) & 0x7fff) == 0);
}

int npy_half_ne(npymath_half h1, npymath_half h2)
{
    return !npy_half_eq(h1, h2);
}

int npy_half_lt_nonan(npymath_half h1, npymath_half h2)
{
    int sign_h1 = (h1 & 0x8000u) == 0x8000u;
    int sign_h2 = (h2 & 0x8000u) == 0x8000u;
    // if both `h1` and `h2` have same sign
    //   Test if `h1` > `h2` when `h1` has the sign
    //        or `h1` < `h2` when is not.
    //   And make sure they are not equal to each other
    //       in case of both are equal to +-0
    // else
    //   Test if  `h1` has the sign.
    //        and `h1` != -0.0 and `h2` != 0.0
    return (sign_h1 == sign_h2) ? (sign_h1 ^ (h1 < h2)) && (h1 != h2)
                                : sign_h1 && ((h1 | h2) != 0x8000u);
}

int npy_half_lt(npymath_half h1, npymath_half h2)
{
    return (!npy_half_isnan(h1) && !npy_half_isnan(h2)) && npy_half_lt_nonan(h1, h2);
}

int npy_half_gt(npymath_half h1, npymath_half h2)
{
    return npy_half_lt(h2, h1);
}

int npy_half_le_nonan(npymath_half h1, npymath_half h2)
{
    int sign_h1 = (h1 & 0x8000u) == 0x8000u;
    int sign_h2 = (h2 & 0x8000u) == 0x8000u;
    // if both `h1` and `h2` have same sign
    //   Test if `h1` > `h2` when `h1` has the sign
    //        or `h1` < `h2` when is not.
    //        or a == b (needed even if we used <= above instead
    //                   since testing +-0 still required)
    // else
    //   Test if `h1` has the sign
    //        or `h1` and `h2` equal to +-0.0
    return (sign_h1 == sign_h2) ? (sign_h1 ^ (h1 < h2)) || (h1 == h2)
                                : sign_h1 || ((h1 | h2) == 0x8000u);
}

int npy_half_le(npymath_half h1, npymath_half h2)
{
    return (!npy_half_isnan(h1) && !npy_half_isnan(h2)) && npy_half_le_nonan(h1, h2);
}

int npy_half_ge(npymath_half h1, npymath_half h2)
{
    return npy_half_le(h2, h1);
}

npymath_half npy_half_divmod(npymath_half h1, npymath_half h2, npymath_half *modulus)
{
    float fh1 = npy_half_to_float(h1);
    float fh2 = npy_half_to_float(h2);
    float div, mod;

    div = npy_divmodf(fh1, fh2, &mod);
    *modulus = npy_float_to_half(mod);
    return npy_float_to_half(div);
}



/*
 ********************************************************************
 *                     BIT-LEVEL CONVERSIONS                        *
 ********************************************************************
 */

npymath_uint16 npy_floatbits_to_halfbits(npymath_uint32 f)
{
    npymath_uint32 f_exp, f_sig;
    npymath_uint16 h_sgn, h_exp, h_sig;

    h_sgn = (npymath_uint16) ((f&0x80000000u) >> 16);
    f_exp = (f&0x7f800000u);

    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (f_exp >= 0x47800000u) {
        if (f_exp == 0x7f800000u) {
            /* Inf or NaN */
            f_sig = (f&0x007fffffu);
            if (f_sig != 0) {
                /* NaN - propagate the flag in the significand... */
                npymath_uint16 ret = (npymath_uint16) (0x7c00u + (f_sig >> 13));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return (npymath_uint16) (h_sgn + 0x7c00u);
            }
        } else {
            /* overflow to signed inf */
#if NPY_HALF_GENERATE_OVERFLOW
            npy_set_floatstatus_overflow();
#endif
            return (npymath_uint16) (h_sgn + 0x7c00u);
        }
    }

    /* Exponent underflow converts to a subnormal half or signed zero */
    if (f_exp <= 0x38000000u) {
        /*
         * Signed zeros, subnormal floats, and floats with small
         * exponents all convert to signed zero half-floats.
         */
        if (f_exp < 0x33000000u) {
#if NPY_HALF_GENERATE_UNDERFLOW
            /* If f != 0, it underflowed to 0 */
            if ((f&0x7fffffff) != 0) {
                npy_set_floatstatus_underflow();
            }
#endif
            return h_sgn;
        }
        /* Make the subnormal significand */
        f_exp >>= 23;
        f_sig = (0x00800000u + (f&0x007fffffu));
#if NPY_HALF_GENERATE_UNDERFLOW
        /* If it's not exactly represented, it underflowed */
        if ((f_sig&(((npymath_uint32)1 << (126 - f_exp)) - 1)) != 0) {
            npy_set_floatstatus_underflow();
        }
#endif
        /*
         * Usually the significand is shifted by 13. For subnormals an
         * additional shift needs to occur. This shift is one for the largest
         * exponent giving a subnormal `f_exp = 0x38000000 >> 23 = 112`, which
         * offsets the new first bit. At most the shift can be 1+10 bits.
         */
        f_sig >>= (113 - f_exp);
        /* Handle rounding by adding 1 to the bit beyond half precision */
#if NPY_HALF_ROUND_TIES_TO_EVEN
        /*
         * If the last bit in the half significand is 0 (already even), and
         * the remaining bit pattern is 1000...0, then we do not add one
         * to the bit after the half significand. However, the (113 - f_exp)
         * shift can lose up to 11 bits, so the || checks them in the original.
         * In all other cases, we can just add one.
         */
        if (((f_sig&0x00003fffu) != 0x00001000u) || (f&0x000007ffu)) {
            f_sig += 0x00001000u;
        }
#else
        f_sig += 0x00001000u;
#endif
        h_sig = (npymath_uint16) (f_sig >> 13);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_sig will be zero.
         * This is the correct result.
         */
        return (npymath_uint16) (h_sgn + h_sig);
    }

    /* Regular case with no overflow or underflow */
    h_exp = (npymath_uint16) ((f_exp - 0x38000000u) >> 13);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    f_sig = (f&0x007fffffu);
#if NPY_HALF_ROUND_TIES_TO_EVEN
    /*
     * If the last bit in the half significand is 0 (already even), and
     * the remaining bit pattern is 1000...0, then we do not add one
     * to the bit after the half significand.  In all other cases, we do.
     */
    if ((f_sig&0x00003fffu) != 0x00001000u) {
        f_sig += 0x00001000u;
    }
#else
    f_sig += 0x00001000u;
#endif
    h_sig = (npymath_uint16) (f_sig >> 13);
    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_sig will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
#if NPY_HALF_GENERATE_OVERFLOW
    h_sig += h_exp;
    if (h_sig == 0x7c00u) {
        npy_set_floatstatus_overflow();
    }
    return h_sgn + h_sig;
#else
    return h_sgn + h_exp + h_sig;
#endif
}

npymath_uint16 npy_doublebits_to_halfbits(npymath_uint64 d)
{
    npymath_uint64 d_exp, d_sig;
    npymath_uint16 h_sgn, h_exp, h_sig;

    h_sgn = (d&0x8000000000000000ULL) >> 48;
    d_exp = (d&0x7ff0000000000000ULL);

    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (d_exp >= 0x40f0000000000000ULL) {
        if (d_exp == 0x7ff0000000000000ULL) {
            /* Inf or NaN */
            d_sig = (d&0x000fffffffffffffULL);
            if (d_sig != 0) {
                /* NaN - propagate the flag in the significand... */
                npymath_uint16 ret = (npymath_uint16) (0x7c00u + (d_sig >> 42));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return h_sgn + 0x7c00u;
            }
        } else {
            /* overflow to signed inf */
#if NPY_HALF_GENERATE_OVERFLOW
            npy_set_floatstatus_overflow();
#endif
            return h_sgn + 0x7c00u;
        }
    }

    /* Exponent underflow converts to subnormal half or signed zero */
    if (d_exp <= 0x3f00000000000000ULL) {
        /*
         * Signed zeros, subnormal floats, and floats with small
         * exponents all convert to signed zero half-floats.
         */
        if (d_exp < 0x3e60000000000000ULL) {
#if NPY_HALF_GENERATE_UNDERFLOW
            /* If d != 0, it underflowed to 0 */
            if ((d&0x7fffffffffffffffULL) != 0) {
                npy_set_floatstatus_underflow();
            }
#endif
            return h_sgn;
        }
        /* Make the subnormal significand */
        d_exp >>= 52;
        d_sig = (0x0010000000000000ULL + (d&0x000fffffffffffffULL));
#if NPY_HALF_GENERATE_UNDERFLOW
        /* If it's not exactly represented, it underflowed */
        if ((d_sig&(((npymath_uint64)1 << (1051 - d_exp)) - 1)) != 0) {
            npy_set_floatstatus_underflow();
        }
#endif
        /*
         * Unlike floats, doubles have enough room to shift left to align
         * the subnormal significand leading to no loss of the last bits.
         * The smallest possible exponent giving a subnormal is:
         * `d_exp = 0x3e60000000000000 >> 52 = 998`. All larger subnormals are
         * shifted with respect to it. This adds a shift of 10+1 bits the final
         * right shift when comparing it to the one in the normal branch.
         */
        assert(d_exp - 998 >= 0);
        d_sig <<= (d_exp - 998);
        /* Handle rounding by adding 1 to the bit beyond half precision */
#if NPY_HALF_ROUND_TIES_TO_EVEN
        /*
         * If the last bit in the half significand is 0 (already even), and
         * the remaining bit pattern is 1000...0, then we do not add one
         * to the bit after the half significand.  In all other cases, we do.
         */
        if ((d_sig&0x003fffffffffffffULL) != 0x0010000000000000ULL) {
            d_sig += 0x0010000000000000ULL;
        }
#else
        d_sig += 0x0010000000000000ULL;
#endif
        h_sig = (npymath_uint16) (d_sig >> 53);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_sig will be zero.
         * This is the correct result.
         */
        return h_sgn + h_sig;
    }

    /* Regular case with no overflow or underflow */
    h_exp = (npymath_uint16) ((d_exp - 0x3f00000000000000ULL) >> 42);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    d_sig = (d&0x000fffffffffffffULL);
#if NPY_HALF_ROUND_TIES_TO_EVEN
    /*
     * If the last bit in the half significand is 0 (already even), and
     * the remaining bit pattern is 1000...0, then we do not add one
     * to the bit after the half significand.  In all other cases, we do.
     */
    if ((d_sig&0x000007ffffffffffULL) != 0x0000020000000000ULL) {
        d_sig += 0x0000020000000000ULL;
    }
#else
    d_sig += 0x0000020000000000ULL;
#endif
    h_sig = (npymath_uint16) (d_sig >> 42);

    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_sig will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
#if NPY_HALF_GENERATE_OVERFLOW
    h_sig += h_exp;
    if (h_sig == 0x7c00u) {
        npy_set_floatstatus_overflow();
    }
    return h_sgn + h_sig;
#else
    return h_sgn + h_exp + h_sig;
#endif
}

npymath_uint32 npy_halfbits_to_floatbits(npymath_uint16 h)
{
    npymath_uint16 h_exp, h_sig;
    npymath_uint32 f_sgn, f_exp, f_sig;

    h_exp = (h&0x7c00u);
    f_sgn = ((npymath_uint32)h&0x8000u) << 16;
    switch (h_exp) {
        case 0x0000u: /* 0 or subnormal */
            h_sig = (h&0x03ffu);
            /* Signed zero */
            if (h_sig == 0) {
                return f_sgn;
            }
            /* Subnormal */
            h_sig <<= 1;
            while ((h_sig&0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            f_exp = ((npymath_uint32)(127 - 15 - h_exp)) << 23;
            f_sig = ((npymath_uint32)(h_sig&0x03ffu)) << 13;
            return f_sgn + f_exp + f_sig;
        case 0x7c00u: /* inf or NaN */
            /* All-ones exponent and a copy of the significand */
            return f_sgn + 0x7f800000u + (((npymath_uint32)(h&0x03ffu)) << 13);
        default: /* normalized */
            /* Just need to adjust the exponent and shift */
            return f_sgn + (((npymath_uint32)(h&0x7fffu) + 0x1c000u) << 13);
    }
}

npymath_uint64 npy_halfbits_to_doublebits(npymath_uint16 h)
{
    npymath_uint16 h_exp, h_sig;
    npymath_uint64 d_sgn, d_exp, d_sig;

    h_exp = (h&0x7c00u);
    d_sgn = ((npymath_uint64)h&0x8000u) << 48;
    switch (h_exp) {
        case 0x0000u: /* 0 or subnormal */
            h_sig = (h&0x03ffu);
            /* Signed zero */
            if (h_sig == 0) {
                return d_sgn;
            }
            /* Subnormal */
            h_sig <<= 1;
            while ((h_sig&0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            d_exp = ((npymath_uint64)(1023 - 15 - h_exp)) << 52;
            d_sig = ((npymath_uint64)(h_sig&0x03ffu)) << 42;
            return d_sgn + d_exp + d_sig;
        case 0x7c00u: /* inf or NaN */
            /* All-ones exponent and a copy of the significand */
            return d_sgn + 0x7ff0000000000000ULL +
                                (((npymath_uint64)(h&0x03ffu)) << 42);
        default: /* normalized */
            /* Just need to adjust the exponent and shift */
            return d_sgn + (((npymath_uint64)(h&0x7fffu) + 0xfc000u) << 42);
    }
}

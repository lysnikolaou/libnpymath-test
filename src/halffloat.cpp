#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/*
 * If these are 1, the conversions try to trigger underflow,
 * overflow, and invalid exceptions in the FP system when needed.
 */
#define NPY_HALF_GENERATE_OVERFLOW 1
#define NPY_HALF_GENERATE_INVALID 1

#include "numpy/halffloat.h"

#include "common.hpp"
/*
 ********************************************************************
 *                   HALF-PRECISION ROUTINES                        *
 ********************************************************************
 */
using namespace npymath;

float npy_half_to_float(npymath_half h)
{
    return static_cast<float>(Half::FromBits(h));
}

double npy_half_to_double(npymath_half h)
{
    return static_cast<double>(Half::FromBits(h));
}

npymath_half npy_float_to_half(float f)
{
    return Half(f).Bits();
}

npymath_half npy_double_to_half(double d)
{
    return Half(d).Bits();
}

int npy_half_iszero(npymath_half h)
{
    return (h&0x7fff) == 0;
}

int npy_half_isnan(npymath_half h)
{
    return Half::FromBits(h).IsNaN();
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
    return Half::FromBits(h1).Equal(Half::FromBits(h2));
}

int npy_half_eq(npymath_half h1, npymath_half h2)
{
    return Half::FromBits(h1) == Half::FromBits(h2);
}

int npy_half_ne(npymath_half h1, npymath_half h2)
{
    return Half::FromBits(h1) != Half::FromBits(h2);
}

int npy_half_lt_nonan(npymath_half h1, npymath_half h2)
{
    return Half::FromBits(h1).Less(Half::FromBits(h2));
}

int npy_half_lt(npymath_half h1, npymath_half h2)
{
    return Half::FromBits(h1) < Half::FromBits(h2);
}

int npy_half_gt(npymath_half h1, npymath_half h2)
{
    return npy_half_lt(h2, h1);
}

int npy_half_le_nonan(npymath_half h1, npymath_half h2)
{
    return Half::FromBits(h1).LessEqual(Half::FromBits(h2));
}

int npy_half_le(npymath_half h1, npymath_half h2)
{
    return Half::FromBits(h1) <= Half::FromBits(h2);
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
    if constexpr (Half::kNativeConversion<float>) {
        return BitCast<uint16_t>(Half(BitCast<float>(f)));
    }
    else {
        return half_private::FromFloatBits(f);
    }
}

npymath_uint16 npy_doublebits_to_halfbits(npymath_uint64 d)
{
    if constexpr (Half::kNativeConversion<double>) {
        return BitCast<uint16_t>(Half(BitCast<double>(d)));
    }
    else {
        return half_private::FromDoubleBits(d);
    }
}

npymath_uint32 npy_halfbits_to_floatbits(npymath_uint16 h)
{
    if constexpr (Half::kNativeConversion<float>) {
        return BitCast<uint32_t>(static_cast<float>(Half::FromBits(h)));
    }
    else {
        return half_private::ToFloatBits(h);
    }
}

npymath_uint64 npy_halfbits_to_doublebits(npymath_uint16 h)
{
    if constexpr (Half::kNativeConversion<double>) {
        return BitCast<uint64_t>(static_cast<double>(Half::FromBits(h)));
    }
    else {
        return half_private::ToDoubleBits(h);
    }
}


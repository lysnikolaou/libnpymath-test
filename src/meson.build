# We need -lm for all C code (assuming it uses math functions, which is safe to
# assume for NumPy). For C++ it isn't needed, because libstdc++/libc++ is
# guaranteed to depend on it.
m_dep = cc.find_library('m', required : false)
mlib_linkflag = ''
if m_dep.found()
  mlib_linkflag = '-lm'
  add_project_link_arguments(mlib_linkflag, language : 'c')
endif

# Platform detection
is_windows = host_machine.system() == 'windows'
is_mingw = is_windows and cc.get_id() == 'gcc'

if is_mingw
  # For mingw-w64, link statically against the UCRT.
  gcc_link_args = ['-lucrt', '-static']
  add_project_link_arguments(gcc_link_args, language: ['c', 'cpp'])
  # Force gcc to float64 long doubles for compatibility with MSVC
  # builds, for C only.
  add_project_arguments('-mlong-double-64', language: 'c')
  # Make fprintf("%zd") work (see https://github.com/rgommers/scipy/issues/118)
  add_project_arguments('-D__USE_MINGW_ANSI_STDIO=1', language: ['c', 'cpp'])
endif

src_file_cli = find_program('../_build_utils/process_src_template.py')
src_file = generator(src_file_cli,
  arguments : ['@INPUT@', '--outfile', '@OUTPUT@'],
  output : '@BASENAME@'
)

cdata = configuration_data()

# Check sizes of types. Note, some of these landed in config.h before, but were
# unused. So clean that up and only define the NPYMATH_SIZEOF flavors rather than
# the SIZEOF ones
types_to_check = [
  ['NPYMATH_SIZEOF_SHORT', 'short'],
  ['NPYMATH_SIZEOF_INT', 'int'],
  ['NPYMATH_SIZEOF_LONG', 'long'],
  ['NPYMATH_SIZEOF_LONGLONG', 'long long'],
  ['NPYMATH_SIZEOF_FLOAT', 'float'],
  ['NPYMATH_SIZEOF_DOUBLE', 'double'],
  ['NPYMATH_SIZEOF_LONGDOUBLE', 'long double'],
]
foreach symbol_type: types_to_check
  cdata.set(symbol_type[0], cc.sizeof(symbol_type[1]))
endforeach

# Check for complex support
if not cc.has_header('complex.h')
  error('"complex.h" header not found')
endif

if cc.get_argument_syntax() == 'msvc'
  complex_types_to_check = [
    ['NPYMATH_SIZEOF_COMPLEX_FLOAT', '_Fcomplex'],
    ['NPYMATH_SIZEOF_COMPLEX_DOUBLE', '_Dcomplex'],
    ['NPYMATH_SIZEOF_COMPLEX_LONGDOUBLE', '_Lcomplex'],
  ]
else
  complex_types_to_check = [
    ['NPYMATH_SIZEOF_COMPLEX_FLOAT', 'complex float'],
    ['NPYMATH_SIZEOF_COMPLEX_DOUBLE', 'complex double'],
    ['NPYMATH_SIZEOF_COMPLEX_LONGDOUBLE', 'complex long double'],
  ]
endif
foreach symbol_type: complex_types_to_check
  if not cc.has_type(symbol_type[1], prefix: '#include <complex.h>')
    t = symbol_type[1]
    error(f'"complex.h" header does not include complex type @t@')
  endif
  cdata.set(symbol_type[0], cc.sizeof(symbol_type[1], prefix: '#include <complex.h>'))
endforeach

# Mandatory functions: if not found, fail the build
# Some of these can still be blocklisted if the C99 implementation
# is buggy, see numpy/_core/src/common/npy_config.h
mandatory_math_funcs = [
  'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'fabs',
  'floor', 'ceil', 'sqrt', 'log10', 'log', 'exp', 'asin',
  'acos', 'atan', 'fmod', 'modf', 'frexp', 'ldexp',
  'expm1', 'log1p', 'acosh', 'asinh', 'atanh',
  'rint', 'trunc', 'exp2', 'copysign', 'nextafter', 'cbrt',
  'log2', 'pow', 'hypot', 'atan2',
]
foreach func: mandatory_math_funcs
  if not cc.has_function(func, prefix: '#include <math.h>', dependencies: m_dep)
    error(f'Function `@func@` not found')
  endif
endforeach

mandatory_complex_math_funcs = [
  'csin', 'csinh', 'ccos', 'ccosh', 'ctan', 'ctanh', 'creal', 'cimag', 'conj'
]
foreach func: mandatory_complex_math_funcs
  if not cc.has_function(func, prefix: '#include <complex.h>', dependencies: m_dep)
    error(f'Function `@func@` not found')
  endif
endforeach

c99_complex_funcs = [
  'cabs', 'cacos', 'cacosh', 'carg', 'casin', 'casinh', 'catan',
  'catanh', 'cexp', 'clog', 'cpow', 'csqrt',
  # The long double variants (like csinl)  should be mandatory on C11,
  # but are missing in FreeBSD. Issue gh-22850
  'csin', 'csinh', 'ccos', 'ccosh', 'ctan', 'ctanh',
]
foreach func: c99_complex_funcs
  func_single = func + 'f'
  func_longdouble = func + 'l'
  if cc.has_function(func, prefix: '#include <complex.h>', dependencies: m_dep)
    cdata.set10('NPYMATH_HAVE_' + func.to_upper(), true)
  endif
  if cc.has_function(func_single, prefix: '#include <complex.h>', dependencies: m_dep)
    cdata.set10('NPYMATH_HAVE_' + func_single.to_upper(), true)
  endif
  if cc.has_function(func_longdouble, prefix: '#include <complex.h>', dependencies: m_dep)
    cdata.set10('NPYMATH_HAVE_' + func_longdouble.to_upper(), true)
  endif
endforeach

# We require C99 so these should always be found at build time. But for
# libnpymath as a C99 compat layer, these may still be relevant.
c99_macros = ['isfinite', 'isinf', 'isnan', 'signbit']
foreach macro: c99_macros
  if cc.has_function(macro, prefix: '#include <math.h>', dependencies: m_dep)
    cdata.set10('NPYMATH_HAVE_DECL_' + macro.to_upper(), true)
  endif
endforeach

if cc.has_header('features.h')
  cdata.set10('NPYMATH_HAVE_FEATURES_H', true)
endif

# Optional GCC compiler builtins and their call arguments.
# If given, a required header and definition name (HAVE_ prepended)
# Call arguments are required as the compiler will do strict signature checking
optional_intrinsics = [
  ['__builtin_isnan', '5.'],
  ['__builtin_isinf', '5.'],
  ['__builtin_isfinite', '5.'],
  ['__builtin_expect', '5, 0'],
]
foreach intrin: optional_intrinsics
  func = intrin[0]
  func_args = intrin[1]
  code = f'''
    #ifdef _MSC_VER
    #pragma function(@func@)
    #endif
    int main(void) {
      @func@(@func_args@);
      return 0;
    };
    '''
  define_name = 'NPYMATH_HAVE_' + func.to_upper()
  if cc.links(code)
    cdata.set10(define_name, true)
  endif
endforeach

# This is a port of the old python code for identifying the long double
# representation to C.  The old Python code is in this range:
# https://github.com/numpy/numpy/blob/eead09a3d02c09374942cdc787c0b5e4fe9e7472/numpy/core/setup_common.py#L264-L434
# This port is in service of solving gh-23972
# as well as https://github.com/mesonbuild/meson/issues/11068
longdouble_format = meson.get_external_property('longdouble_format', 'UNKNOWN')
if longdouble_format == 'UNKNOWN'
  longdouble_format = meson.get_compiler('c').run(
'''
#include <stdio.h>
#include <string.h>

#define repcmp(z) (memcmp((const char *)&foo.x, z, sizeof(foo.x)) == 0)

const struct {
  char before[16];
  long double x;
  char after[8];
} foo = {{'\0'}, -123456789.0, {'\0'}};

int main(void) {
  switch (sizeof(foo.x)) {
  case 8: {
    if (repcmp(
            ((const char[]){0000, 0000, 0000, 0124, 0064, 0157, 0235, 0301}))) {
      fprintf(stdout, "IEEE_DOUBLE_LE");
      return 0;
    }
    if (repcmp(
            ((const char[]){0301, 0235, 0157, 0064, 0124, 0000, 0000, 0000}))) {
      fprintf(stdout, "IEEE_DOUBLE_BE");
      return 0;
    }
    fprintf(stdout, "UNKNOWN");
    return 1;
  }
  case 12: {
    if (repcmp(((const char[]){0000, 0000, 0000, 0000, 0240, 0242, 0171, 0353,
                               0031, 0300, 0000, 0000}))) {
      fprintf(stdout, "INTEL_EXTENDED_12_BYTES_LE");
      return 0;
    }
    if (repcmp(((const char[]){0300, 0031, 0000, 0000, 0353, 0171, 0242, 0240,
                               0000, 0000, 0000, 0000}))) {
      fprintf(stdout, "MOTOROLA_EXTENDED_12_BYTES_BE");
      return 0;
    }
    fprintf(stdout, "UNKNOWN");
    return 1;
  }
  case 16: {
    if (repcmp(
            ((const char[]){0000, 0000, 0000, 0000, 0240, 0242, 0171, 0353,
                            0031, 0300, 0000, 0000, 0000, 0000, 0000, 0000}))) {
      fprintf(stdout, "INTEL_EXTENDED_16_BYTES_LE");
      return 0;
    }
    if (repcmp(
            ((const char[]){0300, 0031, 0326, 0363, 0105, 0100, 0000, 0000,
                            0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000}))) {
      fprintf(stdout, "IEEE_QUAD_BE");
      return 0;
    }
    if (repcmp(
            ((const char[]){0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000,
                            0000, 0000, 0100, 0105, 0363, 0326, 0031, 0300}))) {
      fprintf(stdout, "IEEE_QUAD_LE");
      return 0;
    }
    if (repcmp(
            ((const char[]){0000, 0000, 0000, 0124, 0064, 0157, 0235, 0301,
                            0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000}))) {
      fprintf(stdout, "IBM_DOUBLE_DOUBLE_LE");
      return 0;
    }
    if (repcmp(
            ((const char[]){0301, 0235, 0157, 0064, 0124, 0000, 0000, 0000,
                            0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000}))) {
      fprintf(stdout, "IBM_DOUBLE_DOUBLE_BE");
      return 0;
    }
    fprintf(stdout, "UNKNOWN");
    return 1;
  }
  }
}
  ''').stdout()
endif
if longdouble_format == 'UNKNOWN' or longdouble_format == 'UNDEFINED'
  error('Unknown long double format of size: ' + cc.sizeof('long double').to_string())
endif
cdata.set10('NPYMATH_HAVE_LDOUBLE_' + longdouble_format, true)

if cc.has_header('endian.h')
  cdata.set10('NPYMATH_HAVE_ENDIAN_H', true)
endif
if cc.has_header('sys/endian.h')
  cdata.set10('NPYMATH_HAVE_SYS_ENDIAN_H', true)
endif

config_h = configure_file(
  input: 'npymath_meson_config.h.in',
  output: 'npymath_meson_config.h',
  configuration: cdata,
  install: false
)

_npymathconfig_h = configure_file(
  input: 'include/_npymathconfig.h.in',
  output: '_npymathconfig.h',
  configuration: cdata,
  install: false,
)

# Build npymath static library
# ----------------------------

staticlib_cflags = [
  '-DNPYMATH_INTERNAL_BUILD',
  '-DNPYMATH_HAVE_CONFIG_H',
]
if cc.get_id() == 'msvc'
  # Disable voltbl section for vc142 to allow link using mingw-w64; see:
  # https://github.com/matthew-brett/dll_investigation/issues/1#issuecomment-1100468171
  # Needs to be added to static libraries that are shipped for reuse (i.e.,
  # libnpymath and libnpyrandom)
  if cc.has_argument('-d2VolatileMetadata-')
     staticlib_cflags +=  '-d2VolatileMetadata-'
  endif
endif

if get_option('numpy_build')
  build_flags = ['-DNUMPY_BUILD']
else
  build_flags = []
endif
staticlib_cflags += build_flags

npy_math_internal_h = custom_target(
  output: 'npy_math_internal.h',
  input: 'npy_math_internal.h.src',
  command: [src_file_cli, '@INPUT@', '-o', '@OUTPUT@'],
)

libnpymath_sources = [
  src_file.process('ieee754.c.src'),
  src_file.process('npy_math_complex.c.src'),
  npy_math_internal_h,
  'halffloat.c',
  'npy_math.c',
]
libnpymath_includes = include_directories('include', '.')

if get_option('install_lib')
  libnpymath = library(
    'npymath',
    libnpymath_sources,
    c_args: staticlib_cflags,
    include_directories: libnpymath_includes,
    dependencies: py_dep,
    install: true,
    install_dir: py.get_install_dir() / get_option('lib_dir'),
  )

  py.install_sources(
    [
      'include/numpy/npy_math.h',
      'include/numpy/halffloat.h',
      'include/npymath_common.h',
      'include/npymathconfig.h',
      _npymathconfig_h,
    ],
    subdir: get_option('headers_dir'),
  )
else
  libnpymath = library(
    'npymath',
    libnpymath_sources,
    c_args: staticlib_cflags,
    include_directories: libnpymath_includes,
    dependencies: py_dep,
  )
endif

libnpymath_dep = declare_dependency(
  compile_args: build_flags,
  sources: [npy_math_internal_h],
  include_directories: libnpymath_includes,
  link_with: libnpymath,
)

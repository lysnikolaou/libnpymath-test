project(
  'libnpymath',
  'c', 'cpp',
  version: '0.0.1',
  license: 'BSD-3',
  meson_version: '>=1.2.1',
  default_options: [
    'c_std=c99',
    'cpp_std=c++17',
  ],
)

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

# Check compiler is recent enough (see the SciPy Toolchain Roadmap for details)
if cc.get_id() == 'gcc'
  if not cc.version().version_compare('>=8.4')
    error('libnpymath requires GCC >= 8.4')
  endif
elif cc.get_id() == 'msvc'
  if not cc.version().version_compare('>=19.20')
    error('libnpymath requires at least vc142 (default with Visual Studio 2019) ' + \
          'when building with MSVC')
  endif
endif

libnpymath_install_dir = py3.get_install_dir() / 'libnpymath'
libnpymath_headers_dir = 'libnpymath'
subdir('src')

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

# We install libnpymath and libnpyrandom; ensure they're using a `.lib` rather
# than a `.a` file extension in order not to break including them in a
# distutils-based build (see gh-23981 and
# https://mesonbuild.com/FAQ.html#why-does-building-my-project-with-msvc-output-static-libraries-called-libfooa)
if is_windows and cc.get_id() == 'msvc'
  name_prefix_staticlib = ''
  name_suffix_staticlib = 'lib'
else
  name_prefix_staticlib = []
  name_suffix_staticlib = []
endif

src_file_cli = find_program('_build_utils/process_src_template.py')
src_file = generator(src_file_cli,
  arguments : ['@INPUT@', '--outfile', '@OUTPUT@'],
  output : '@BASENAME@'
)

subdir('core')

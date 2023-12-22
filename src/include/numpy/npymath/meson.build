meson_config_h = configure_file(
  input: 'meson_config.h.in',
  output: 'meson_config.h',
  configuration: cdata,
  install: false,
)

_config_h = configure_file(
  input: '_config.h.in',
  output: '_config.h',
  configuration: cdata,
  install: false,
)

install_dir = py.get_install_dir()

if get_option('install_lib')
  internal_h = custom_target(
    input: 'internal.h.src',
    output: 'internal.h',
    command: [src_file_cli, '@INPUT@', '-o', '@OUTPUT@'],
    # This installs the header in the numpy include folder
    install: true,
    install_dir: install_dir / get_option('headers_dir') / 'npymath',
  )
else
  internal_h = custom_target(
    input: 'internal.h.src',
    output: 'internal.h',
    command: [src_file_cli, '@INPUT@', '-o', '@OUTPUT@'],
  )
endif

# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

def rules(ctx):
    compile_opts = [
        '-O3', '-std=gnu++20', '-I_out',
        '-Wall', '-Wextra', '-Wpedantic', '-Wconversion', '-Wshadow', '-Werror', '-Wno-unused-parameter',
    ]
    if ctx.host.arch == 'x86_64':
        compile_opts += ['-march=x86-64-v3']

    sfpi_path = ctx.env.get('SFPI_PATH', ctx.path.expanduser('~/sfpi-7.76.0'))
    sfpi_gxx = f'{sfpi_path}/compiler/bin/riscv-tt-elf-g++'
    sfpi_objcopy = f'{sfpi_path}/compiler/bin/riscv-tt-elf-objcopy'

    cpp_files = ['main.cpp', 'gguf.cpp', 'tokenizer.cpp', 'tensor.cpp', 'model.cpp', 'requant.cpp', 'tt_backend.cpp']
    o_files = []
    for file in cpp_files:
        src_file = f'src/{file}'
        o_file = f'_out/{file.replace(".cpp", ".o")}'
        d_file = o_file.replace('.o', '.d')
        inputs = [src_file]
        if file == 'tt_backend.cpp':
            inputs += ['_out/brisc.inc']
        cmd = ['g++', *compile_opts, '-MMD', '-c', src_file, '-o', o_file]
        ctx.rule(o_file, inputs, cmd=cmd, depfile=d_file)
        o_files += [o_file]

    matvec_objects = []
    for isa in ('v3', 'v4'):
        obj = f'_out/tt_matvec_{isa}.o'
        ctx.rule(obj, 'src/tt_matvec.cpp',
                 cmd=['g++', *compile_opts, f'-march=x86-64-{isa}', '-Wno-psabi', '-MMD',
                      '-c', 'src/tt_matvec.cpp', '-o', obj], depfile=obj.replace('.o', '.d'))
        matvec_objects.append(obj)
    o_files += matvec_objects

    target = '_out/tt-lab'
    ctx.rule(target, o_files, cmd=['g++', *compile_opts, *o_files, '-o', target, '-ldl'])

    test_o_files = ['_out/tests.o', '_out/tensor.o', '_out/gguf.o', '_out/requant.o']
    ctx.rule('_out/tests.o', 'src/tests.cpp',
             cmd=['g++', *compile_opts, '-MMD', '-c', 'src/tests.cpp', '-o', '_out/tests.o'],
             depfile='_out/tests.d')

    target = '_out/test-core'
    ctx.rule(target, test_o_files, cmd=['g++', *compile_opts, *test_o_files, '-o', target])

    ctx.rule('_out/host_dispatch.o', 'tests/host_dispatch.cpp',
             cmd=['g++', *compile_opts, '-Isrc', '-MMD', '-c', 'tests/host_dispatch.cpp',
                  '-o', '_out/host_dispatch.o'], depfile='_out/host_dispatch.d')
    dispatch_objects = ['_out/host_dispatch.o', '_out/tensor.o', *matvec_objects]
    ctx.rule('_out/test-host-dispatch', dispatch_objects,
             cmd=['g++', *compile_opts, *dispatch_objects, '-o', '_out/test-host-dispatch'])

    sfpi_opts = [
        '-O2', '-std=gnu++20', '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter',
        '-fno-exceptions', '-fno-rtti', '-fno-threadsafe-statics',
        '-ffreestanding', '-fno-builtin', '-ffp-contract=fast',
        '-mcpu=tt-bh', '-march=rv32imf_zba_zbb_zicsr_zifencei', '-mabi=ilp32f',
    ]
    ctx.rule('_out/kernels_sfpu.inc', ['src/kernels.sfpu', 'tools/sfpu.py'],
             cmd=['python3', 'tools/sfpu.py', 'src/kernels.sfpu', '-o', '_out/kernels_sfpu.inc'])
    ctx.rule('_out/brisc.elf', ['src/brisc_entry.S', 'src/brisc.cpp', 'src/brisc.ld', '_out/kernels_sfpu.inc'],
             cmd=[sfpi_gxx, *sfpi_opts, '-I_out', '-MMD', '-MF', '_out/brisc.d', '-nostdlib',
                  '-Wl,-T,src/brisc.ld', '-Wl,--no-warn-rwx-segments',
                  '-o', '_out/brisc.elf', 'src/brisc_entry.S', 'src/brisc.cpp'],
             depfile='_out/brisc.d')
    ctx.rule('_out/brisc.bin', '_out/brisc.elf',
             cmd=[sfpi_objcopy, '-O', 'binary', '_out/brisc.elf', '_out/brisc.bin'])
    ctx.rule('_out/brisc.inc', '_out/brisc.bin',
             cmd=['/bin/sh', '-c',
                  "od -An -v -t x1 _out/brisc.bin | sed 's/[0-9a-f][0-9a-f]/0x&,/g' > _out/brisc.inc"])

    ctx.rule(':build', ['_out/tt-lab', '_out/test-core'])

    ctx.rule('_out/core-test.stamp', '_out/test-core',
             cmd=['/bin/sh', '-c', '_out/test-core && touch _out/core-test.stamp'], allow_output=True)
    model = ctx.env.get('GPT_OSS_TEST_MODEL', ctx.path.expanduser('~/models/gpt-oss-20b-MXFP4.gguf'))
    ctx.rule('_out/model-test.stamp', ['_out/tt-lab', 'tests/run.sh', model],
             cmd=['/bin/sh', '-c', 'tests/run.sh && touch _out/model-test.stamp'])
    ctx.rule('_out/sfpu-test.stamp', ['tools/sfpu.py', 'src/kernels.sfpu', 'src/ckernel_ops.h'],
             cmd=['/bin/sh', '-c', 'python3 tools/sfpu.py --self-test && touch _out/sfpu-test.stamp'],
             allow_output=True)
    ctx.rule('_out/router-test.stamp', 'tools/test_router_top4.py',
             cmd=['/bin/sh', '-c', 'python3 tools/test_router_top4.py && touch _out/router-test.stamp'],
             allow_output=True)
    ctx.rule('_out/requant-test.stamp', ['_out/tt-lab', 'tests/requant.py'],
             cmd=['/bin/sh', '-c', 'python3 tests/requant.py && touch _out/requant-test.stamp'],
             allow_output=True)
    ctx.rule('_out/host-dispatch-test.stamp', '_out/test-host-dispatch',
             cmd=['/bin/sh', '-c', '_out/test-host-dispatch && touch _out/host-dispatch-test.stamp'],
             allow_output=True)
    ctx.rule(':test', ['_out/core-test.stamp', '_out/sfpu-test.stamp', '_out/router-test.stamp',
                       '_out/requant-test.stamp', '_out/host-dispatch-test.stamp'])
    ctx.rule(':test-model', [':test', '_out/model-test.stamp'])

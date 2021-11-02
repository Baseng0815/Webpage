---
title:
    shedOS Toolchain and Makefiles
---

The bulk of our build system consists of Makefiles and other small tools like the
compiler, linker and assembler being invoked by those Makefiles to form a single build system.
The root Makefile defines some variables needed by all subprojects, namely

- user CFLAGS  (-O0 ...)
- make options (-j12 ...)
- target (x86_64-elf)
- directories (dependencies, system root, toolchain)

We then have a list of rules which describe how specific parts of the project are
to be built/run:

- qemu
- hard disk image
- sysroot
- toolchain
- clean

The qemu rule configures and starts our virtual machine. The hard disk image
rule creates a .hdd image. The sysroot contains all of our operating system;
subprojects will be gradually installed there by looping through a list of all
the subprojects, installing their headers and then finally installing their
executables and libraries.  At last, we have a rule to configure and build our
entire toolchain. The `clean` rules are used to *clean up* our directories,
i.e. delete previously built targets like object files.  We use an in-source
build which means our directories will be flooded which .o files and the like
once we run our Makefiles. This might be changed in the future.

The core of the build process is made up by the cross toolchain. It contains
the GNU binutils and `gcc` for compilation. We use a generic `x86_64-elf`
target which we can replace with a custom one later one once our OS becomes
more sophisticated. There is not much one needs to modify, but we need to take
extra care to disable the *red zone*. You can read more about what the red zone
is, why we need to and how we can disable it on [this osdev
article](https://wiki.osdev.org/Libgcc_without_red_zone). Basically, we just
add a new config file which adds `mno-red-zone` to the multilib options. This
creates another libgcc with red zone disabled. We can link with this instead of
the default one simply by specifying `-mno-red-zone` as a compilation option.

Step-by-step guides and additional information can be found on [this osdev
article about GCC
cross-compilation](https://wiki.osdev.org/GCC_Cross-Compiler).

We will first take a look at how the kernel is built. Its Makefile defines the

- compiler
- assembler
- linker
- archiver

, although we only use the compiler and assembler. Linking will be done through
`gcc` itself.

Each project has two top-level rules, `install-headers` and `install-exec`.
Install-headers will be run first for each project which will install all the
public headers into the `sysroot/include` directory. This is necessary because
other projects might need certain headers for their own compilation process.
The `install-exec` rule will then actually take all source files, be it C or
assembly or brainfuck, compile it to an object file and link all of this
together to a final executable. While this is pretty simple, there are some
flags needed to actually make this work properly.

The most important one is `-ffreestanding`, making sure that we target a
freestanding environment. Together with the `-nostdlib` linker flag, we
completely separate ourselves from all host libraries and make sure no unwanted
dependencies are accidentally inserted.

Because we load the kernel at the higher half later on, we need to create a
*position-independent executable*. This is done using the compiler and linker
flags `-pie -fno-pic -fpie`.

We also need to create a custom linker script. An ELF file consists of multiple
*sections*, each of which containing different data. The most important ones
are

- .text, for code
- .data, for initialized, modifiable data
- .rodata, for initialized, non-modifiable data
- .bss, for uninitialized, modifiable data

We also have a special `.stivale2hdr` section which is needed by the
bootloader. What it does and how it works will be discussed in another post.
The linker script tells the linker where to put those sections in the final
executable. It also puts it at an offset of 0xfffffff80000000 + 2 MiB, i.e. the
final 2 GiB of our 64 bit address space. This is called higher half linking.

The actual compilation process is very simple. We just invoke the corresponding
program (`gcc` for compilation and `as` for assembling) and link all of the
objects together.

As you see, setting up a build system in and of itself can present a
challenging task. I didn't have much experience in cross compilation and all
the gcc options when I started so I had to do lots of learning and
experimentation. One could use their host gcc and skip this whole process, but
this is a bad idea as you need to be sure your toolchain doesn't depend on host
libraries AND you also understand your toolchain well enough to solve problems
on your own.

**Doing things the 'easy way' and hiding complexity just leads to more problems
later on, so always try to properly learn what you are doing. This holds true
for everything, not just for OS development!**.

---

We are now able to compile our whole kernel using just a single `make` command.
Great! But we still can't run it and also can't do much as we have no way of
printing to the screen or doing any other usable form of I/O.  We need a
*bootloader* to load our kernel and give it some information. How we can
accomplish this will be presented in [part 2](/html/shedOS_booting.html) of this
series.

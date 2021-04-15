# My Projects

This list is by no means exhaustive or complete, but should give a rough
outline about how I spend my programming time and efforts. If you are interested
in projects summaries instead of detailed description, check out the corresponding github
repository. Most of them (should) include a `README.md` which gives a nice overview
of the project and how to build it.

## shedOS
![shedOS](../res/shedOS.jpg "shedOS")

shedOS is an x86_64 operating system I am currently developing, mainly for educational purposes.
Working in kernel space without being able to rely on things we take for granted when working in userspace
like memory allocation, process management, multitasking and more is an experience that really deepens
one's knowledge of how exactly operating systems manage resources and their internal workings.

The OS is written in C as it is the *lingua franca* of low level system programming. It basically translates
one to one to assembly and has minimal dependencies so it allows for easy development of freestanding
applications using options and custom toolchains. We basically create a freestanding 64-bit position-independent
elf executable containing all the kernel code, linked at the higher half. This will then
be copied to a partition containing an echfs file system on a .hdd file where we will also install limine, the
bootloader. Limine directly puts the kernel at the higher half, sets up long mode, necessary paging and jumps
to the entry point. This disk image can be attached to your virtual machine of choice (I use QEMU) and limine
will boot the kernel.

All of those steps, including the cross toolchain itself, are defined as Makefile rules for easy compilation
and execution. You can find the github repository [here](https://github.com/Baseng0815/shedOS) and detailed
descriptions below.

1. [Toolchain and Makefiles](shedOS_toolchain.html)
1. [Booting](shedOS_booting.html)

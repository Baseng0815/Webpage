---
title:
    shedOS Booting
---

We managed to create our kernel executable in [part 1](shedOS_toolchain.html). But what can we do with it?
We obviously can't just run it in our user environment because there is no environment without an OS!
That's why we have to somehow get our kernel loaded into RAM and jump to its entry point. But I just said
there is no environment, so there are no fopen() or malloc() functions. Well yes, but actuall no. See, every
operating system uses some of the functions provided by the *firmware* to allocate memory and load itself.
But because the OS is not loaded yet, we need to rely on another piece of software to load it, called
the *bootloader*.

I want to present two ways of loader an operating system which I have both used before. The first
one uses a custom bootloader, a *UEFI* application. UEFI, the Unified Extensible Firmware Interface,
is a standardized interface to the system hardware that exposes not just memory allocation or file loading
functions, but even advanced capabilities like printing to the screen! You might know BIOS, the Basic I/O System.
This is the predecessor of UEFI which lacks lots of features like support for large partitions, support for more
than four partitions, power and system management and also boots slower.

In addition to just calling the kernel entry point, the kernel needs some information passed to it to properly
do its job. This information shouldn't depend on the bootloader so the kernel can be booted by anything, but in
reality I've been too lazy to properly separate those stages. **A task for future me**.

The kernel expects

- A memory map so we know which memory we can write to
- A graphical framebuffer so we can put pixels on the screen

## custom UEFI bootloader

There are multiple ways to create a UEFI application. One can rely on the EDK2 toolchain, which is huge and bloated,
or use [gnu-efi](https://wiki.osdev.org/GNU-EFI) which is more lightweight and user-friendly. UEFI applications do
not use the elf format, they are PE executables, a format specified by Microsoft. This means we need a cross-compiler
targeting `x86_64-w64-mingw32` or any other equivalent target. While you could build a new gcc toolchain, LLVM clang
can also be used as it is a cross compiler by nature. [This article](https://wiki.osdev.org/UEFI_App_Bare_Bones#Under_LLVM.2Fclang)
describes how to use the efi subsystem to compile a UEFI application using clang.

The UEFI application will be loaded and executed on boot by the firmware. Its job is to locate the partition where
the OS is located, allocate some memory on the disk, load the kernel file, extract the elf headers and jump to its
entry point. Before jumping, one should call `ExitBootServices` to signal that we have fully taken control. We also
need to grab the memory map and a framebuffer.  To summarize:

1. create a UEFI bootloader application using a cross compiler
2. create the kernel elf
3. put everything onto a bootable medium using [xorriso](https://wiki.osdev.org/UEFI_App_Bare_Bones#Creating_the_FAT_image)
4. choose the bootable medium as a primary boot device in the UEFI interface
5. allocate memory and load kernel elf
6. grab memory map and framebuffer
7. parse elf and grab entry point
8. exit boot services and call kernel entry with the memory map and framebuffer

Creating a UEFI application was a good learning experience, but I encountered some problems down the line and people
recommended me to use `limine`.

## limine
[limine](https://github.com/limine-bootloader/limine) is a bootloader supporting multiple boot procotols and file systems.
I chose to use the [stivale2](https://github.com/stivale/stivale/blob/master/STIVALE2.md) protocol. This allows me to assume
the kernel receives certain data and is able to use certain functions without directly relying on things like UEFI. Limine
automatically loads the kernel, puts the CPU into long mode, sets up paging and maps the kernel into the higher half.

A [higher half kernel](https://wiki.osdev.org/Higher_Half_Kernel) means that the kernel is mapped into the last 2GiB of
the address space. Physically, it is still located in some low memory of course as no one has petabytes of memory (at least not yet).
Instead, the last 2GiB are paged to its physical location to allow the kernel to work as if it was loaded high. For a 64-bit
address space, this means the kernel starts at 0xffffffff80000000 + 2MiB. We add 2MiB because the kernel physically starts at the 2MiB
mark as it's better to leave the first two megs untouched.

After loading and mapping the kernel, we need to exchange some information. This information exchange is defined by the stivale2
specification. Do you remember the `.stivale2hdr` section of our elf file? This section contains a `stivale2_hdr` struct allowing us
to define where our stack is and where our linked list of tags start. The basic idea is that you give limine a linked list of
`stivale2_header_tag`, each of which requests a certain feature or changes a specific setting. The kernel then receives data back
from limine in the form of a linked list of `stivale2_struct_tag`.

We receive our memory map through `stivale2_struct_tag_memmap` and the framebuffer through `stivale2_struct_tag_framebuffer`.

---

Our kernel is now running, having full control of the computer hardware. We have a memory map we can use to manage the system memory
and a graphical framebuffer into which we can shove pixels. Getting this up and running takes lots of effort, but is also a great
learning experience. We can now start to really get our hands dirty with tables, paging, memory management, interrupts and all the
other nice things.

I chose to first create a small library that allows me to print formatted strings a la `printf`. This is great to get some colored
output and make your OS look like it has more features than it really has. It is also a great motivational boost to actually be
able to see the text you've written earlier and shoved through your toolchain and into the VM.

In [Part 3 (TODO)](shedOS_paging.html) we will discuss how memory and paging work, how to write a crude physical
page frame bitmap allocator and how to create page tables.

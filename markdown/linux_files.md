---
title:
    Files in Linux
---

## The file system

Almost all major operating and file systems arrange data in a hierarchical,
tree-like directory structure. This works by having a root directory which can
contain an arbitrary number of recursively nested subdirectories and files.
Although this approach is commonly used, details on implementation and usage
philosophy differ. We will specifically look at the standard directories making
up a common Linux file system and explore the philosophy behind the *everything
is a file*-idiom using on-hand examples. As Linux is heavily based on Unix
philosophy, most of what I present here does not actually come from Linux, but
rather is a derivation of already existing concepts stemming from much older
systems.

To start off, let's take a look at how the Arch Linux directory structure looks.

```
# an example file system layout
/
├── bin     -> usr/bin
├── boot
├── dev
├── etc
├── home
├── lib     -> usr/lib
├── lib64   -> usr/lib
├── opt
├── proc
├── sbin    -> usr/bin
├── tmp
├── usr
│   ├── bin
│   ├── include
│   ├── lib
│   ├── lib32
│   ├── lib64   -> lib
│   ├── local
│   ├── sbin    -> bin
│   ├── share
└── var
```

That's only a selection of directories, but already a lot to take in. We'll
start off by noticing that lots of directories are actually just symbolic links
(i.e. aliases which refer to the same underlying file) which can be ignored.
These symlinks don't exist on most distros - they are nonstandard and the
separation of them actually serves a purpose - but we will ignore that for now.
A detailed description of what each directory is used for can be found in the
[Filesystem Hierarchy
Standard](https://refspecs.linuxfoundation.org/FHS_3.0/fhs-3.0.pdf) published
by the Linux Standard Base working group.

- `/boot`: boot-relevant files (bootloader cfg, kernel, initrd), commonly
resides on another partition
- `/dev`: file representation of hardware devices (more on that later)
- `/etc`: config files (*edit to configure*)
- `/home`: user home directories
- `/opt`: mostly software that is distributed in bundles (*Windows-style*)
- `/proc`: file representation of processes
- `/tmp`: temporary files
- `/usr`: readonly Unix system resources
- `/usr/bin`: binary executables
- `/usr/include`: C headers
- `/usr/lib`: static and shared libraries as well as kernel modules
- `/usr/local`: libs, incs and bins that are managed manually
(i.e. not by package manager)
- `/usr/share`: shared data like keyboard layouts or application data
- `/var`: variable per-application data that changes over time

If you contrast this layout to the one Windows 10 uses, one thing that comes to
mind is how a Linux distro bundles files of the same type, not from the same
source. This means a C library called *bestlib* would have its include headers
stored in `/usr/include`, while its shared library file would end up in
`/usr/lib`. Windows would probably have a folder somewhere called *bestlib*
containing the include headers as well as the shared library file. The former is
particularly useful if you do a lot of development in C/C++ as there are
standard locations where most libraries and headers reside. I remember some
projects where I had to specify over 15 additional include and library
directories using Visual Studio in 2016 which was a real PITA and made the
whole process unnecessarily difficult.

Another thing you might have noticed is the lack of a C: or D: drive letter.
Windows generally represents each physical file system with a different root.
This root is known as the drive letter and is prefixed to every path so you get
things like `C:/Users/shed` for the users' home directory. Linux would
represent the same location as `/home/shed`. But how come Linux does not need
a drive letter if there are multiple drives connected?

## Mounting

We have seen that Linux uses only a single root. The process of binding other
file systems and disks to some location in the directory tree is known as
*mounting*. The `mount` command takes two main parameters: the device to mount
and the mount point. As we will see later on, every disk has a file
representation in `/dev`. Mounting allows one to seamlessly include many
different storage devices into a single file system. The way it works for me is
that I have my home directory on a separate partition and that I use a network
attached storage (NAS) for all of my non-local files.

```
# my layout
/
├── home (on /dev/sda2)
│   ├ NAS (on //192.168.2.110/NAS)
│   │   ├─ Uni (on //192.168.2.110/NAS)
│   │   ...
│   ├ notes (on /dev/sda2)
│   ...
├── boot (on /dev/sda1)
├── dev (on /dev/sda1)
├── etc (on /dev/sda1)
...
```

How does the system memorize where to mount what? Mounting rules are defined in
the **f**ile **s**ystem **tab**le `/etc/fstab`. This files tells the kernel
what to mount (disk UUID, network IP...), where to mount (/boot, /home...) and
how to mount (credentials, read/write, uid/gid...). Add any devices you want to
auto-mount on boot to this file.

```
# Mounting a disk with a given UUID as the root
# <file system>                             <dir>   <type>      <options>
UUID=ecae701e-aafe-4ceb-86e9-ecbda8752e4f   /       ext4        rw,relatime 0 1
```

I don't need to care whether a file sits on my computer or on somebody else's
computer (a.k.a. the *cloud*) or on a USB stick or even on a magnetic tape
drive - the kernel presents a clean directory tree and has drivers for a
variety of devices that handle the read and write operations. Lots of mounted
file systems together with their files are not even real disks, they exist
purely in memory, are created by the kernel and have abundant use cases.

## *Everything is a file* - but is it really?

Imagine wanting to read data from a webcam. How would you do that? A
non-Unix-aware programmer might look for a library that enumerates all webcams
and exposes API functionality akin to `readWebcamData(cam, buf, byteCount)`.
You might want to read mouse input next and try to do the same, but instead
look for a `readMouseData(mouse, x, y)` function. Do you notice the problem?
We would need a wrapper for every kind of device there is which is time
consuming if not nearly impossible. This is where files come in to the rescue.

Of course, not everything is a file. A webcam is not a file, it's a solid
object. A mouse also. In my opinion, it's better to say that *everything is
represented as a file*. This is the defining feature of Unix-based operating
systems and is therefore important to understand. The basic idea is that we
define a common API on files, let's say `read` and `write`, and use this API to
read from webcams, mice, keyboards etc.

Do you remember the `/dev/` directory? It does not really exist - only in
memory and is created by the kernel. As are all the files in there. Let's take
a look at a specific file, let's say `/dev/video0`. This file represents a
webcam. Additional webcams might be represented as `/dev/video1` etc. The
kernel detects these devices and exposes them through these files. A programmer
can then read data using `read("/dev/video0")` to read webcam data. He can also
do `read("/dev/mouse")` to read mouse data. Very easy.

Other examples of file-exposed devices you often see are the framebuffer,
terminals, hardware timers, random number generators, stderr/stdin/stdout and
useful process information (take a look at `/proc/self` for example), but there
are many many more.

## Taking it a step further - redirecting output

Now that we have a theoretical base on how files work, let's have some fun with
I/O redirection. Processes in Linux have a standard output which we can
redirect to other files. We can do interprocess communication by redirecting
stdout of our current process into a file which represents some form of other
process. Let's fire up two terminals and type `tty` on both. `tty` stands for
*teletype* and allows you to find out the file which represents the current
terminal stdin/stdout. We get the following two files.

- `/dev/pts/1`
- `/dev/pts/2`

We can do the funny and redirect the output of `/dev/pts/1` to `/dev/pts/2`.
Go to pts1 and type `echo 'Hello World' > /dev/pts/2`. You will see the text
appear on pts2. Boom, interprocess communication. We can do the same using the
standard input too. Try typing `cat /dev/pts/2` on pts1. `cat` tries to read
from pts2, but pts2 is also trying to read! If you type something into pts1,
you'll notice a few characters going to pts1 and a few going to pts2. You have
created a race condition which might break your terminal. This is also the
reason why the kernel has to go to great length to ensure locked, well-defined
device access because if there are simultaneous read/writes, races can occur
and lead to data loss or security vulnerabilities.

## A practical note

I wanted to turn off the CPU fan on my ThinkPad the other day and did not
immediately find a good program, so I started looking around in the `/proc`
directory which aside from processes contains various virtual files created by
drivers. I found the `/proc/acpi/ibm/fan` file which represents the fan.
Exactly what I needed!  Now I can simply type `echo 'level 0' >
/proc/acpi/ibm/fan` to turn the fan off and `echo 'level auto' >
/proc/acpi/ibm/fan` to turn the fan to auto mode again. By doing this, I'm
directly interacting with the driver and controlling the hardware through the
file read/write API. I do the same for my keyboard backlight and have written
some scripts to simplify the process. There's no need for external programs
when you have device files and I/O redirection - you can build complex things
with simple building blocks, a core principle of the [Unix Philosophy
(TODO)](./todo.html).

Files in Linux are really powerful and every user ought to learn how to use them
properly. You need them mostly when mounting, but they also allow all kinds of
ways to directly interact with the kernel and your machine on a low level.

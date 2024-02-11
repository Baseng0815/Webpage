---
title:
    Hooking shared library symbols
---

System libraries expose common functionality like opening files, allocating
memory and comparing strings to programs through clearly defined interfaces to
reduce repetition and the amount of code developers need to write. They are
really nothing more than binary files with some metadata and lots of functions.
These functions in itself are only addresses which are made accessible by an
associated symbol table mapping human-readable symbols to them. Programs can
then load these libraries and through the magic of linking, symbols are
resolved and their corresponding functions can be executed.

Libraries need to be mapped to the address space of the program using them
obviously. There are two approaches to this:

1. dynamic linking: shared library calls from programs are dynamically resolved
   by the program loader which needs to locate the appropriate library file on
   disk, load it into the address space of the caller and then resolve the
   symbol to an address
2. static linking: statically linked libraries are copy-and-pasted into the
   final executable, therefore circumventing the need for dynamic loading but
   with the tradeoff of a larger size

In the case of Linux systems, libraries can usually be found in `/usr/lib/` and
`/usr/local/lib/`. By running `ldd <file>`, we can see which shared libraries a
program requires to run. A simple program like cat might require only a few
libraries while a more complicated program like ssh will most certainly require
more:

```{.numberLines}
❯ ldd /usr/bin/cat
        linux-vdso.so.1 (0x00007ffd19be1000)
        libc.so.6 => /usr/lib/libc.so.6 (0x0000754deb858000)
        /lib64/ld-linux-x86-64.so.2 => /usr/lib64/ld-linux-x86-64.so.2 (0x0000754deba83000)
❯ ldd /usr/bin/ssh
        linux-vdso.so.1 (0x00007fff8fb57000)
        libcrypto.so.3 => /usr/lib/libcrypto.so.3 (0x00007a08f8600000)
        libldns.so.3 => /usr/lib/libldns.so.3 (0x00007a08f8bdb000)
        libgssapi_krb5.so.2 => /usr/lib/libgssapi_krb5.so.2 (0x00007a08f8b87000)
        libz.so.1 => /usr/lib/libz.so.1 (0x00007a08f8b6d000)
        libc.so.6 => /usr/lib/libc.so.6 (0x00007a08f841e000)
        libssl.so.3 => /usr/lib/libssl.so.3 (0x00007a08f833e000)
        libkrb5.so.3 => /usr/lib/libkrb5.so.3 (0x00007a08f8266000)
        libk5crypto.so.3 => /usr/lib/libk5crypto.so.3 (0x00007a08f8b3d000)
        libcom_err.so.2 => /usr/lib/libcom_err.so.2 (0x00007a08f8b37000)
        libkrb5support.so.0 => /usr/lib/libkrb5support.so.0 (0x00007a08f8258000)
        libkeyutils.so.1 => /usr/lib/libkeyutils.so.1 (0x00007a08f8251000)
        libresolv.so.2 => /usr/lib/libresolv.so.2 (0x00007a08f8240000)
        /lib64/ld-linux-x86-64.so.2 => /usr/lib64/ld-linux-x86-64.so.2 (0x00007a08f8d59000)
```

Perhaps the most important library of all is `libc.so`: the C runtime library.
It contains not only useful functions for string manipulation, memory
allocation etc., but also wraps most system calls which is the lowest level of
abstraction attainable for common userspace applications and allow direct
interfacing with the kernel. Two important system calls are `read(...)` and
`write(...)`: most program I/O goes through these two system calls. They are so
important that amongst hundreds of others, they occupy slots 0 and 1!

The dynamic linker for Linux (`ld-linux-x86-64.so.2`) can do many neat things
by setting certain environment variables. We'll first take a look at
`LD_DEBUG`: by running any program with `LD_DEBUG=help`, we are presented a
list of all valid values which can help enormously when debugging:

```
❯ LD_DEBUG=help cat
Valid options for the LD_DEBUG environment variable are:

  libs        display library search paths
  reloc       display relocation processing
  files       display progress for input file
  symbols     display symbol table processing
  bindings    display information about symbol binding
  versions    display version dependencies
  scopes      display scope information
  all         all previous options combined
  statistics  display relocation statistics
  unused      determined unused DSOs
  help        display this help message and exit

...
```

Listing locations the loader searches in for libraries for example can be done
using `LD_DEBUG=libs`:

```
❯ LD_DEBUG=libs python3
     63173:     find library=libpython3.11.so.1.0 [0]; searching
     63173:      search path=/usr/local/lib/glibc-hwcaps/x86-64-v3:/usr/local/lib/glibc-hwcaps/x86-64-v2:/usr/local/lib            (LD_LIBRARY_PATH)
     63173:       trying file=/usr/local/lib/glibc-hwcaps/x86-64-v3/libpython3.11.so.1.0
     63173:       trying file=/usr/local/lib/glibc-hwcaps/x86-64-v2/libpython3.11.so.1.0
     63173:       trying file=/usr/local/lib/libpython3.11.so.1.0
     63173:      search cache=/etc/ld.so.cache
     63173:       trying file=/usr/lib/libpython3.11.so.1.0
     63173:
     63173:     find library=libc.so.6 [0]; searching
     63173:      search path=/usr/local/lib             (LD_LIBRARY_PATH)
     63173:       trying file=/usr/local/lib/libc.so.6
     63173:      search cache=/etc/ld.so.cache
     63173:       trying file=/usr/lib/libc.so.6
     63173:
     63173:     find library=libm.so.6 [0]; searching
     63173:      search path=/usr/local/lib             (LD_LIBRARY_PATH)
     63173:       trying file=/usr/local/lib/libm.so.6
     63173:      search cache=/etc/ld.so.cache
     63173:       trying file=/usr/lib/libm.so.6
...
```

There are many more options to explore - if you are curious just take a look at
`man 8 ld.so`. One thing I find especially interesting is the ability to hook
into certain library functions by implementing them on our own and then
injecting them through a custom shared library. Let's see how this works in
practice.

It is not hard to imagine a situation in which we are curious about the I/O
behavior of our system. This can be done by replacing the aforementioned
`read(...)` and `write(...)` functions with our own implementation.
Reimplementing longer library functions sounds like a nightmare, but it's
really not - we can simply add our code and then call into the original one
which we lazily retrieve and store using `dlsym(...)`. Without further ado,
here is a simple program hooking into `read(...)` and `write(...)` logging the
amount of data and the data itself into the file given by the envvar
`IOTRACE_FILE`.

```{.c .numberLines}
#include <dlfcn.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static ssize_t (*read_real)(int fd, void *buf, size_t count) = NULL;
static ssize_t (*write_real)(int fd, const void *buf, size_t count) = NULL;

static size_t total_bytes_read = 0;
static size_t total_bytes_written = 0;
static FILE *filp = NULL;

static void ensure_valid_filp(void)
{
        if (!filp) {
                const char *file = getenv("IOTRACE_FILE");
                filp = fopen(file, "w");
                if (!filp) {
                        fprintf(stderr, "TRACE: couldn't open trace file\n");
                        exit(1);
                }
        }
}

static void print_bytes(uint8_t *bytes, size_t count)
{
        if (count == 0)
                return;

        fprintf(filp, "DATA: ");
        for (size_t i = 0; i < count; i++) {
                uint8_t c = bytes[i];
                if (c == '\0')
                        break;

                fprintf(filp, "%c", c);
        }

        fprintf(filp, "\n");
}

ssize_t read(int fd, void *buf, size_t count)
{
        if (!read_real) {
                read_real = dlsym(RTLD_NEXT, "read");
        }

        ensure_valid_filp();

        total_bytes_read += count;

        ssize_t result = read_real(fd, buf, count);

        fprintf(filp, "TRACE: read: %lu bytes from fd %d, got %ld\n", count, fd, result);
        print_bytes((uint8_t*)buf, result);
        fprintf(filp, "TRACE: total read so far: %lu bytes\n", total_bytes_read);

        return result;
}

ssize_t write(int fd, const void *buf, size_t count)
{
        if (!write_real) {
                write_real = dlsym(RTLD_NEXT, "write");
        }

        ensure_valid_filp();

        total_bytes_written += count;

        ssize_t result = write_real(fd, buf, count);

        fprintf(filp, "TRACE: write: %lu bytes to fd %d, got %ld\n", count, fd, result);
        print_bytes((uint8_t*)buf, result);
        fprintf(filp, "TRACE: total written so far: %lu bytes\n", total_bytes_written);

        return result;
}
```

Notice how we store a handle to the original functions using `dlsym(RTLD_NEXT,
"read")` and `dlsym(RTLD_NEXT, "write")`. This check needs to be done every
time since there is no clearly defined entry to initialize stuff in.

We save this file as `iotrace.c` and compile it into a shared library using gcc
with debugging symbols. It is worth mentioning that shared libraries need to be
relocatable - otherwise the whole concept wouldn't work since they are
dynamically loaded without a fixed base address.

```
❯ gcc -ggdb3 -shared -fPIC iotrace.c -o iotrace.so
```

We can then instruct the loader to load our custom library before all others
using the `LD_PRELOAD=$PWD/iotrace.so` envvar. To see whether it has picked up
our library correctly, we also set `LD_TRACE_LOADED_OBJECTS=1` to see a trace
of the loaded libraries (this is actually what `ldd` uses under the hood).

```
❯ LD_TRACE_LOADED_OBJECTS=1 LD_PRELOAD=/home/bastian/Misc/test/iotrace /usr/bin/cat
        linux-vdso.so.1 (0x00007ffcbeff8000)
        /home/bastian/Misc/test/iotrace (0x000071e40ae75000)
        libc.so.6 => /usr/lib/libc.so.6 (0x000071e40ac56000)
        /lib64/ld-linux-x86-64.so.2 (0x000071e40ae7c000)
```

Looks great, now it's time to put it to the test: let's execute the same
command without the `LD_TRACE_LOADED_OBJECTS=1` part and with
`IOTRACE_FILE=trace_log`:

```
❯ IOTRACE_FILE=trace_log LD_PRELOAD=$PWD/iotrace.so /usr/bin/cat /usr/share/applications/* >/dev/null
❯ head -n25 trace_log
TRACE: read: 131072 bytes from fd 3, got 249
DATA: [Desktop Entry]
Name=Android File Transfer (MTP)
Comment=Transfer files from/to MTP devices
Exec=android-file-transfer
Icon=android-file-transfer
StartupNotify=false
Terminal=false
Type=Application
Categories=Utility;System;FileTools;Filesystem;Qt;

TRACE: total read so far: 131072 bytes
TRACE: write: 249 bytes to fd 1, got 249
DATA: [Desktop Entry]
Name=Android File Transfer (MTP)
Comment=Transfer files from/to MTP devices
Exec=android-file-transfer
Icon=android-file-transfer
StartupNotify=false
Terminal=false
Type=Application
Categories=Utility;System;FileTools;Filesystem;Qt;

TRACE: total written so far: 249 bytes
TRACE: read: 131072 bytes from fd 3, got 0
```

Looks great - just through this small program we can learn that `cat` tries to
read in blocks of 131072 (0x20000) bytes at once. We also see lots of writes to
fd 1 which of course is just stdout.

While this specific example can only be called instructional, it is easy to see
how this method can be extended and adapted to more meaningful problems and
it's also fun to learn more about the system you're working with. One can never
have enough tools under one's belt!

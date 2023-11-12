---
title:
    Projects
---

This list is by no means exhaustive or complete, but should give a rough
outline about how I spend my programming time and efforts. If you are
interested in projects summaries instead of detailed description, check out the
corresponding github repository. Most of them (should) include a `README.md`
which gives a nice overview of the project and how to build it.

## [ascii-render](https://github.com/Baseng0815/ascii-render/)
![](/res/ascii-render.jpg "ascii-render")

ascii-render is a small Rust program that takes a GIF and token file and
renders the GIF in real-time using the tokens specified in the token file. It
is simple and the result for one example video that has been done a million
times already can be seen [here](https://www.youtube.com/watch?v=ZaW37nEcPQM).

## [ray tracer](https://github.com/Baseng0815/raytracer)
![](/res/raytracer.jpg "raytracer")

raytracer is, as you might have guessed already, a simple ray tracer written in
Rust. It is a follow-along implementation project of the great "Ray Tracing In
One Weekend" series.

## [vex](https://github.com/Baseng0815/vex)
![](/res/vex.png "vex")

vex is a simple hex editor with vim-like bindings for easy navigation. It
currently supports browsing a file, editing individual bytes and saving the
file with plenty of movement commands already implemented. It is very
lightweight, only using a TUI based on ncurses. Althought it is already usable,
I plan to add a way to insert new bytes or delete existing ones as well as more
advanced ergonomics like visual (block) selection.

## [shedOS](https://github.com/Baseng0815/shedOS)
![](/res/shedOS.jpg "shedOS")

shedOS is an x86_64 operating system I am currently developing, mainly for
educational purposes.  Working in kernel space without being able to rely on
things we take for granted when working in userspace like memory allocation,
process management, multitasking and more is an experience that really deepens
one's knowledge of how exactly operating systems manage resources and their
internal workings.

The OS is written in C as it is the *lingua franca* of low level system
programming. It basically translates one to one to assembly and has minimal
dependencies so it allows for easy development of freestanding applications
using options and custom toolchains. We basically create a freestanding 64-bit
position-independent elf executable containing all the kernel code, linked at
the higher half. This will then be copied to a partition containing an echfs
file system on a .hdd file where we will also install limine, the bootloader.
Limine directly puts the kernel at the higher half, sets up long mode,
necessary paging and jumps to the entry point. This disk image can be attached
to your virtual machine of choice (I use QEMU) and limine will boot the kernel.

All of those steps, including the cross toolchain itself, are defined as
Makefile rules for easy compilation and execution. You can find the github
repository [here](https://github.com/Baseng0815/shedOS) and detailed
descriptions below.

1. [Toolchain and Makefiles](./shedOS_toolchain.html)
2. [Booting](./shedOS_booting.html)
2. [Paging (TODO)](./todo.html)

## [voxel game](https://github.com/Baseng0815/VoxelGame)

![](/res/voxelgame.jpg "Voxel game")

Voxel game is the prototype of a 3D voxel renderer/\*craft clone. It currently
supports procedural terrain generation and texturing with biomes and caves,
first-person movement and block breaking/placing. A simple UI is provided as
well as a working inventory and clouds, including a skybox. Simple diffuse
lighting is implemented as well. The project currently lies dormant, but we
might pick it up again some time later. Because it's quite large, it has a lot
of dependencies like Freetype, Assimp and libnoise. You can find a full list as
well as build instructions in the repository.

## [xanim](https://github.com/Baseng0815/xanim)

xanim is a simple animated wallpaper manager for X11. It works by creating an
SDL2 render context from the root window (aka the 'background') and then
blitting a series of images to it. The images are obtained from a video file
which is read and parsed using OpenCV. You thus need Xlib, SDL2 and OpenCV to
be able to successfully compile the program.

I have measured the performance on my laptop (16GB RAM, i9 10510H) with a 1080p
video and after loading it consumes on average < 1% of CPU time.  RAM could be
a problem tho as every image is loaded without compression and stored in RAM
instead of VRAM. I have yet to figure out how to store the images on the GPU to
solve this problem but w/e, just don't use videos larger than a few Gigs and
you should be fine.

This method of rendering unfortunately doesn't work too well when using a
composite manager such as picom or compton. They draw to an external buffer and
then blur all the windows together to achieve their result. If you know of a
method to make it work in this case, please let me know!

## [climate data](https://github.com/Baseng0815/Climate)

I use my Raspberry Pi 4 for a lot of things, one of them being recording
climate data. The [DHT 22](https://www.adafruit.com/product/385) is a low-cost,
low-power humidity and temperature sensor.

![](/res/dht22_wiring.gif "DHT22 wiring")

It is easy to set up and can be read through the adafruit DHT python library
based on which I have written a [simple script](
https://github.com/Baseng0815/Climate). The Pi 4 also runs a mongo database
which stores the humidity, temperature and date.  A new document is inserted
every 5 minutes.  Plans for the future include an express.js-based API for
accessing this data and filtering for maximum and minimum temperature/humidity.

## [This site](https://github.com/Baseng0815/Webpage/)

I prefer simple and static layouts to dynamic ones requiring client-sided
JavaScript. This is why the great majority of pages are written in simple
Markdown and then fed through [pandoc](https://pandoc.org/) to generate HTML
documents. Although Markdown is not a clear standard and lots of dialects
exist, pandoc takes almost everything as input and produces a valid output
document. For instance, if I need to add a table or grid of images to my
documents, I can revert to writing plain HTML in the Markdown file and pandoc
will happily accept this.

To make compilation of files easier, I make use of a single Makefile that

- compiles `.md` files in the `markdown` directory to `.html` files and copies
  them to the `html` directory
- resizes files in the `res` directory down to an appropriate size so I can put
  lots of images on a single page without taking up too much bandwidth

In addition, resources embedded in newer pages link to the full-sized resource
which I do not store in the repository for obvious reasons (a single image can
be well over 20MiB, now imagine dozens or even hundreds of them on a single
page...).

I chose not to include navigation headers and footers for now since I assume
people casually browsing this site start out from the main page and can
navigate backwards using their browser on their own and people who land on a.
specific site are not always interested in anything else (if they are they can
still navigate to the main page on their own). I might add this in the future
though.

### Subdomains

Various services are also offered on different subdomains:

- [FTP directory](https://ftp.bengel.xyz): a 50GiB-sized directory for (right
  now read-only) file access
- [ytdlp](https://ytdlp.bengel.xyz): a PHP-powered ytdlp frontend for
  bullshit-less media downloading on the go
- [climate data live-view](https://climate.bengel.xyz): a Spring-powered
  frontend for accessing live climate data obtained from my climate sensor
  (might not be up-to-date since I power off the Raspberry Pi regularly)

## [discord Bot](https://github.com/Baseng0815/HelmtraegerBot)

Me and my friends use Discord for communicating. Discord offers lots of
features out of the box like free servers, video chat and stream capabilities
as well as sophisticated user, role and permission management. But sometimes
there are features we want or need which are not provided per-default.
Fortunately, discord offers an API to allow developers to create a *bot user*
which can read and send messages, do administrative tasks and even play audio.
One can use `discord.js`, a Node.js javascript binding allowing to easily
create such a bot. The repository for our bot can be found
[here](https://github.com/Baseng0815/HelmtraegerBot). It currently allows you
to

- automatically download and send video embeds of multiple sources like twitter
using [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- show and play back audio files like a soundboard,
- look up the current weather for a given region using the [openweathermap
api](https://openweathermap.org/api)
- look up a player's CS:GO faceit stats using the [faceit
api](https://developers.faceit.com/),
- look up anime using the [anilist
api](https://anilist.gitbook.io/anilist-apiv2-docs/),
- [uwuify](https://www.urbandictionary.com/define.php?term=uwuify) a message,
- send a random
  [copypasta](https://www.urbandictionary.com/define.php?term=copypasta)

The great thing is that there are lots of Node.js bindings for APIs so one can
quickly implement features without needing to mess around for hours. I will
probably add more features in the future as I see fit, maybe even something
like a chess engine or a word guessing game. The possibilities are endless!

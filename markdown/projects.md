---
title:
    Projects
---

This list is by no means exhaustive or complete, but should give a rough
outline about how I spend my programming time and efforts. If you are
interested in projects summaries instead of detailed description, check out the
corresponding github repository. Most of them (should) include a `README.md`
which gives a nice overview of the project and how to build it.

## shedOS
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

1. [Toolchain and Makefiles](/html/shedOS_toolchain.html)
2. [Booting](/html/shedOS_booting.html)

## Discord Bot

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

## Climate data

I use my Raspberry Pi 4 for lots of things, one of them being recording climate
data. The [DHT 22](https://www.adafruit.com/product/385) is a low-cost,
low-power humidity and temperature sensor.

![](/res/dht22_wiring.gif "DHT22 wiring")

It is easy to set up and can be read through the adafruit DHT python
library based on which I have written a [simple script](
https://github.com/Baseng0815/Climate). The Pi 4 also runs a mongo database
which stores the humidity, temperature and date.  A new document is inserted
every 5 minutes.  Plans for the future include an express.js-based API for
accessing this data and filtering for maximum and minimum temperature/humidity.

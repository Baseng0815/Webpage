---
title:
    How vim saves my life
---

This is a small collection of (n)vim features (or plugins) saving me from
horrendously boring and repetitive work. It's mostly a combination of \*nix
programs and macros, but (n)vim provides plenty of useful shortcuts per
default.

## Creating a list of source code

I was transferring a project from cmake to meson the other day and needed to
find a way to manually list all source files as I was using globbing before.
Fortunately, this is quite simple as vim provides a feature to execute a
command and paste its output directly in the editor using `.!` in command mode.
I can then select the files using `vip` (*visual select inner paragraph*) and
combine `norm` with `ysiW'A,` (surround inner word with ' - see
[vim-surround](https://github.com/tpope/vim-surround)) to put quotes around
each file and commata at the end. Adding brackets can be done with `ysip[`
(surround inner paragraph with [). Finally, we compact the array using `gqip`
(*format lines inner paragraph* - this respects the file type and actual
content by the way so nothing actually breaks).

<video autoplay muted loop width="100%" height="100%">
    <source src="../res/vim_source_file_listing.webm" type="video/webm">
</video>

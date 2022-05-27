---
title:
    How vim saves my life
---

This is a small collection of (n)vim features (or plugins) saving me from
horrendously boring and repetitive work. It's mostly a combination of \*nix
programs and macros, but (n)vim provides plenty of useful shortcuts per
default.

## Creating a list of source code file

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

## Easy access to language servers

More often than not people like to use a language server when programming for
things like auto-completion of expressions and quickly looking up
documentation. While lots of IDEs might support their respective languages
quite well (intellij java completion for example is great), it's generally hard
to add additional language support. Fortunately, editors like vim and visual
studio code provide an excellent way to include language servers. Adding
autocompletion features for a new language is made simple by the use of the
[language server
protocol](https://en.wikipedia.org/wiki/Language_Server_Protocol), or LSP for
short. The basic idea is that semantic completion and analysis features are
provided by a specific language server like
[ccls](https://github.com/MaskRay/ccls) speaking the LSP. The editor then only
needs to speak the LSP as well to gain near infinite completion power.

While neovim has LSP support built in, I like to use
[coc.nvim](https://github.com/neoclide/coc.nvim) which is a nodejs extension
host. Using coc, you can install a myriad of glue plugins that allow the editor
to communicate with the language servers. Coc provides a simple way to install
plugins which, in combination with a package manager and sensible defaults,
makes it excellent in quickly setting up completion for your new esolang, given
a language server has been implemented by a third party.

As an example, I've begun to learn [clojure](https://clojure.org/) a while ago,
a LISP-related, JVM-based functional programming language. There might be a
clojure IDE, or there might be none - I didn't looked it up. The only things I
had to do to properly set up my editor were executing two commands:

1. `yay -S clojure-lsp`
2. `:CocInstall coc-clojure`

The first command was used to install the language server and the second one
to install the glue plugin. It automatically activates when editing clojure
files and works wonders. Supported features include not only autocompletion,
but things like jumping to definitions, finding references, renaming symbols
and even inserting snippets.

<video autoplay muted loop width="100%" height="100%">
    <source src="../res/vim_clojure_autocomplete.webm" type="video/webm">
</video>

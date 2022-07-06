---
title:
    No, you don't need more monitors
---

I often see people talking about the need for three or more monitors because
they assume more monitors => more efficient work, but this is *bullshit*. Let
me elaborate.

The real time consumer in programming work is of course the thought necessary
to solve the problem at hand, but we'll leave that out for now and instead
focus on actual, hard workflow.

Browsing through documentation and datasheets is one example which, depending
on the kind of programming you do, might take up a large amount of time. Having
to constantly switch between the text editor and a document viewer is no fun
and takes a considerable amount of work which is why people almost always use
two monitors. The first one holds mostly active windows you interact with
frequently like your text editor or a real-time chat program. The second
monitor holds passive windows like the document viewer or a web browser with
important information. So far, so good. If you have lots of active or lots of
passive windows though, things start to get complicated.  You might often need
to switch focus between your text editor and chat program on your first and
your document viewer and web browser on your second monitor.  On traditional
systems, this becomes a nuisance quickly as you start to spend half of your
time dragging windows around.

Most people are eternally stuck in this state. Their right hand cycles between
keyboard and mouse duty a dozen times a minute. They think the solution is more
monitors because then they can view more windows at once meaning there is less
need to switch around. This creates another problem.

I have tried working with three and even four monitors before - stacked on top,
vertical side-by-side and so on. None of these solutions was satisfying. The
problem here is that when using two monitors only, one can turn the body to
face the middle so then only minimal head movement is necessary to give full
attention to a single monitor at one time. This doesn't work with three
monitors however as the constant switching from left to right puts a lot of
strain on your neck. So, if more monitors don't work the wonders we make them
out to do, what can we do to fix the problem?

The solution is simple: start using your system, especially your window
manager, more effectively. This is explicitly aimed at non-Linux users as they
are the most probable to not know what their OS can really do. Two things you
might want to look into are virtual desktops and window snapping shortcuts.
Virtual desktops or workspaces are used to group windows and allow you to
switch between the different groups, even across monitors.  A few groups I like
to have:

- text editor with major documentation
- compile/debug terminals
- test suites
- minor documentation with web browser
- discord and spotify

, but these can of course be completely customized. Window snapping is simple
too and refers to shortcuts with which you can put a window on the left half or
right half of your monitor. Or the top half. Or the bottom left half. Whatever
you want.

That's basically it. Use your environment more efficiently instead of spending
money on things you don't need (oddly enough, you see this everywhere - trying
to compensate for missing efficiency by bluntly increasing something else;
please try to avoid this). If you want to learn more about how window managers
on \*nix systems work, check out [this introduction to the dwm window
manager(TODO)](./todo.html). And if you really want to spend lots of money
on your setup, make sure to buy *two* large, high-quality monitors or a single
ultrawide one. After all, it's the size that matters.

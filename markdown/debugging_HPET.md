---
title:
    Debugging Adventures #1: HPET IRQ
---

This is the first debugging post of a probably long ongoing series. I don't want
to introduce specific topics or dive deep into explanations. Rather, I will write
a short (or long, it depends) post about roadblocks I encounter and don't manage
to fix in a short period. This here is one of them.

I have written the bare minimum of my kernel, i.e. loading a custom GDT, implementing
a simple terminal and handling interrupts and CPU exceptions. Because this was just
a learning experience, I wanted to tidy up some of my code to prepare for the next stage
of development. The first thing I did was reorganize the address space so everything
is correctly mapped into the higher half, including MMIO. I additionally changed HPET/APIC
MMIO access to use array indices and replaced bit field structs with bitwise logic
which is way more predictable because the former's behavior often depends on the
compiler used.

And of course it didn't work. The kernel just halted at APIC initialization. Fixing
this problem did not take too long though as I only forgot to offset the MMIO registers
into the higher half. A simple macro did the trick.

```C
#define VADDR_ENSURE_HIGHER(p) (p < VADDR_HIGHER ? p + VADDR_HIGHER : p)
lapic_addr = VADDR_ENSURE_HIGHER(madt->local_apic);
```

Having fixed the crashes, my expectations where high. Oh, how foolish to assume that
a simple rewrite without a logic change would *NOT* completely fuck everything over.
Okay, not *everything*, just the HPET timer interrupts. For some reason, the kernel
timer system does not receive any interrupts and thus cannot keep track of the time.

There are two possible points of failure to examine, the first one being the HPET
itself. I quickly noticed that I forgot to shift flags so some of them were not set
correctly, including the IRQ enable bit. Duh.

```C
enum hpet_timer_conf_cap {
    LEVEL_TRIGGERED     = 0x1,
    IRQ_ENABLE          = 0x2,
    PERIODIC_ENABLE     = 0x3,
    PERIODIC_CAPABLE    = 0x4,
    IS_64_BIT           = 0x5,
    DIRECT_ACCUM_SET    = 0x6,
    FORCE_32_BIT        = 0x8,
    ROUTING_CONF        = 0x9,
    FSB_USE             = 0xe,
    FSB_CAPABLE         = 0xf,
    ROUTING_CAP         = 0x20
};

/* this is stupid and wrong */
timer0 |= IRQ_ENABLE;
timer0 |= PERIODIC_ENABLE;
timer0 |= DIRECT_ACCUM_SET;

/* lookin' better now */
timer0 |= 1ul << IRQ_ENABLE;
timer0 |= 1ul << PERIODIC_ENABLE;
timer0 |= 1ul << DIRECT_ACCUM_SET;
```

Repeatedly reading out the first timer's comparator register now gave an increasing
value which meant the timer was properly running. But there are still no IRQs!

Having set the `IRQ_ENABLE` bit of the first timer and seeing it increase properly
as well as having `LEGACY_REPLACE` enabled, the second point of failure could be
the actual APIC code itself as I have rewritten some parts of it too. So what did I
change exactly?

```C
/* the old code, using byte offsets */
enum LAPIC_REGISTER {
    LAPIC_ID        = 0x20,
    LAPIC_VERSION   = 0x30,
    TPR             = 0x80,
    EOI             = 0xb0,
    SIVR            = 0xf0
};

/* the new code, using array indices (same as above, but divided by four)*/
enum lapic_register {
    LAPIC_REG_ID        = 0x08,
    LAPIC_REG_VERSION   = 0x0c,
    LAPIC_REG_TPR       = 0x20,
    LAPIC_REG_EOI       = 0x2c,
    LAPIC_REG_SIVR      = 0x3c
};
```

To illustrate how this change affects the code:
```C
/* register access before */
uint32_t bsp_lapic_id = *(uint32_t*)(lapic_addr + LAPIC_ID);

/* register access after */
uint32_t bsp_lapic_id = lapic_mmio_regs[LAPIC_REG_ID];
```

This changes nothing about how the access works, it just makes it look a little
bit nicer. As you can see, there are no real changes, but it still doesn't work
as intended. The next step is rerolling everything back to an earlier stage where
everything worked and carefully replacing specific parts of the code one by one.

After I had done this, it still didn't work, but I noticed something interesting
when staring at the terminal output. The IDT was stored at address `0x10b40000` which
is clearly not in the higher half! I actually forgot to offset the pmm pages because
this is a job for the virtual memory manager which I've not yet written. By
simply offsetting things manually for now, everything works like a charm again.  👍

---
title:
    Weird things in programming
---

I have encountered some weird behavior of programs or programming languages over
the past couple of years and want to collect them here.

# C

```{.c .numberLines}
#include <stdio.h>

void pass_by_value(int a)
{ }

void pass_by_magic(void)
{
        int magic;
        printf("%d\n", magic);
}

int main(void)
{
        int number = 69;
        pass_by_value(number);
        pass_by_magic();
        return 0;
}
```

This code produces 69 on my machine when compiled with gcc 11.1.0 without any
optimizations. Why? The first argument is passed in register rdi, but `int a` in
line 3 is allocated on the stack so the top of the stack will be set to 69. `int magic`
in line 9 is not initialized, but refers to the top of the stack which still contains
69 as we didn't overwrite it with anything.

```{.c .numberLines}
#include <stdio.h>

int main(void)
{
        int test[100];
        for (size_t i = 0; i < 100; i++) {
                test[i] = i * i;
        }

        for (size_t i = 0; i < 100; i++) {
                printf("%d ", i[test]);
        }

        printf("\n");

        return 0;
}
```

This program prints all numbers from 0 to 99. Doesn't seem unusual, does it? Well,
take a closer look at line 9. We would expect `test[i]`, but not `i[test]`! Why
does it work? Array indexing in C is syntactic sugar for adding an offset to the
pointer and deferencing the resulting address. So `test[i]` is equivalent to `*(test + i)`
which in turn is equivalent to `*(i + test)` because of commutativity of the addition,
finally giving us `i[test]`.

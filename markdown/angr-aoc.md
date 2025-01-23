---
title:
    'Problem solving with angr: a short example'
---

## The basics

[angr](https://github.com/angr/angr is) is a powerful binary analysis tool with
a wide range of applications in the security and reverse engineering domain,
but it can also be used for other kinds of problems. To put it concisely, it
can read program executable files and convert them into logical formulas while
representing unknown values like memory cells, registers or user input from
`stdin` as variables which can then be constrained and solved for.

This allows for lots of interesting applications like solving for a set of
function parameters that produces certain program behavior. Consider the
following snippet of C code:

```{.c .numberLines}
#include <stdio.h>

int derive_passcode(int seed) {
        int a = seed >> 0  & 0xffff;
        int b = seed >> 16 & 0xffff;
        for (int i = 0; i < 100; i++) {
                a *= 0x19461999;
                b *= 0x9af10bbf;
        }

        return a ^ b;
}

int check_access(int passcode) {
        return passcode == derive_passcode(0x12de456c);
}

int main(int argc, char *argv[])
{
        int passcode;
        scanf("%d", &passcode);
        return !check_access(passcode);
}
```

In this case, `derive_passcode` is straightforward but might be way more complex in reality.
It could for example derive valid license keys for a program you wish to use. Your
task now is to find a valid passcode that passes the check. How do we do this without
searching the whole 2^32-large space of possibilities?

Angr can help us here: we load the program, convert it to a formula and then
use [z3](https://github.com/Z3Prover/z3) as a backend to solve this formula. A
great advantage is that angr works with machine code so we don't even need any
debug symbols or source code for this. Let's take a look at the revelant machine
code:

```{.asm}
    ...
0000000000001149 <derive_passcode>:
    1149:       55                      push   %rbp
    114a:       48 89 e5                mov    %rsp,%rbp
    114d:       89 7d ec                mov    %edi,-0x14(%rbp)
    1150:       8b 45 ec                mov    -0x14(%rbp),%eax
    1153:       0f b7 c0                movzwl %ax,%eax
    1156:       89 45 f4                mov    %eax,-0xc(%rbp)
    1159:       8b 45 ec                mov    -0x14(%rbp),%eax
    115c:       c1 e8 10                shr    $0x10,%eax
    115f:       89 45 f8                mov    %eax,-0x8(%rbp)
    1162:       c7 45 fc 00 00 00 00    movl   $0x0,-0x4(%rbp)
    1169:       eb 1c                   jmp    1187 <derive_passcode+0x3e>
    116b:       8b 45 f4                mov    -0xc(%rbp),%eax
    116e:       69 c0 99 19 46 19       imul   $0x19461999,%eax,%eax
    1174:       89 45 f4                mov    %eax,-0xc(%rbp)
    1177:       8b 45 f8                mov    -0x8(%rbp),%eax
    117a:       69 c0 bf 0b f1 9a       imul   $0x9af10bbf,%eax,%eax
    1180:       89 45 f8                mov    %eax,-0x8(%rbp)
    1183:       83 45 fc 01             addl   $0x1,-0x4(%rbp)
    1187:       83 7d fc 63             cmpl   $0x63,-0x4(%rbp)
    118b:       7e de                   jle    116b <derive_passcode+0x22>
    118d:       8b 45 f4                mov    -0xc(%rbp),%eax
    1190:       33 45 f8                xor    -0x8(%rbp),%eax
    1193:       5d                      pop    %rbp
    1194:       c3                      ret

0000000000001195 <check_access>:
    1195:       55                      push   %rbp
    1196:       48 89 e5                mov    %rsp,%rbp
    1199:       48 83 ec 08             sub    $0x8,%rsp
    119d:       89 7d fc                mov    %edi,-0x4(%rbp)
    11a0:       bf 6c 45 de 12          mov    $0x12de456c,%edi
    11a5:       e8 9f ff ff ff          call   1149 <derive_passcode>
    11aa:       39 45 fc                cmp    %eax,-0x4(%rbp)
    11ad:       0f 94 c0                sete   %al
    11b0:       0f b6 c0                movzbl %al,%eax
    11b3:       c9                      leave
    11b4:       c3                      ret
    ...
```

According to the System V ABI, the input to `check_access` is stored in register `edi` and
the output in register `eax`. The problem now is as follows:

<center>
**What do we need to write to `edi` to get a 1 in `eax`?**
</center>

A possible solution will be presented without going too deep into the API of
angr - there is a plethora of documentation and practice material out there.

```{.python .numberLines}
#!/usr/bin/env python3

import angr

# load the program
project = angr.Project('./a.out')

# get the address of the `check_access` function and construct a state ready to execute it
check_access_addr = project.loader.find_symbol('check_access').rebased_addr
check_access_state = project.factory.blank_state(addr=check_access_addr)

# create symbolic variable (this is what we want to solve for)
passcode = check_access_state.regs.edi

# now we need to simulate the program up to the point of return so we can constrain the result
simgr = project.factory.simulation_manager(check_access_state)
simgr.explore(find=0x4011B4)

# constrain the result and solve for the passcode
result_state = simgr.found[0]
result = result_state.regs.eax
result_state.add_constraints(result == 1)
concretized_passcode = result_state.solver.eval(passcode)

print(concretized_passcode)
```

This spits out `703611698` which is the correct value we were looking for.

## Advent of Code

The goal of [Day 17](https://adventofcode.com/2024/day/17) of the aoc 2024
challenges was to find a fixpoint input that encodes a program outputting itself.
Brute-forcing was not possible since the input space is unconstrained and over
two hours of searching yielded nothing. I then tried the good ol' pen-and-paper
approach and figured out some patterns in the program which probably allows you
to solve it, but I didn't want to solve all of this by hand. That's why I
decided to implement the interpreter in a small C program and do the same
procedure as above to find an input that generates the appropriate output. This
is the C program:

```{.c .numberLines}
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

const uint64_t program[] = { 2,4, 1,3, 7,5, 0,3, 1,5, 4,4, 5,5, 3,0 };
const size_t program_len = sizeof(program) / sizeof(program[0]);

uint64_t combo(uint64_t operand, uint64_t regs[3]) {
        switch (operand) {
                case 0:
                case 1:
                case 2:
                case 3: return operand;
                case 4: return regs[0];
                case 5: return regs[1];
                case 6: return regs[2];
        }

        exit(1);
}

void test(uint64_t reg_a) {
        uint64_t regs[] = { reg_a, 0, 0 };
        size_t ip = 0;
        size_t out_index = 0;
        while (ip < program_len) {
                uint64_t instr = program[ip];
                uint64_t operand = program[ip + 1];

                switch (instr) {
                        case 0: regs[0] >>= combo(operand, regs); break;
                        case 1: regs[1] ^= operand; break;
                        case 2: regs[1] = combo(operand, regs) % 8; break;
                        case 3: if (regs[0] != 0) { ip = operand; continue; }; break;
                        case 4: regs[1] ^= regs[2]; break;
                        case 5: if (program[out_index++] != combo(operand, regs) % 8) { return; }; break;
                        case 6: regs[1] = regs[0] >> combo(operand, regs); break;
                        case 7: regs[2] = regs[0] >> combo(operand, regs); break;
                }


                ip += 2;
        }

        if (out_index == program_len) {
                printf("Success!\n");
        }
}

int main(int argc, char *argv[])
{
        test(234971750570964);

        return EXIT_SUCCESS;
}
```

This made it fairly easy to constrain the program to reach
`printf("Success!\n")` and solving for the input `reg_a`. Since there might be
multiple solutions and the challenge specifically asks for the smallest one, we
solve using the `min` function.

```{.python .numberLines}
#!/usr/bin/env python3

import angr
import claripy

proj = angr.Project('a.out')
reg_a = claripy.BVS('reg_a', 64)

func_addr = proj.loader.find_symbol('test').rebased_addr
print(hex(func_addr))

state = proj.factory.blank_state(addr=func_addr)
state.regs.rdi = reg_a

simgr = proj.factory.simulation_manager(state)
simgr.explore(find=0x4013EC)

print(simgr.found)
if simgr.found:
    reg_a_value = simgr.found[0].solver.min(reg_a)
    print(reg_a_value)
```

## Summary

Constraint solving is an extremely powerful technique that has many
applications. You can not just use it for reverse engineering, but also for
encoding problems in programs instead of directly using formulas which I find
to be more ergonomic. A deep understanding of SMT solvers and all the different
options is not necessary, but might be helpful. Definitely a tool that should
be in every computer scientists' toolbox!

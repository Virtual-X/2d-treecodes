#
# Makefile
# Part of MRAG/2d-treecode-potential
#
# Created and authored by Diego Rossinelli on 2015-09-25.
# Copyright 2015. All rights reserved.
#
# Users are NOT authorized
# to employ the present software for their own publications
# before getting a written permission from the author of this file.
#

CC = gcc -std=c99
backend ?= sse
real ?= double
treecode-potential-order ?= 12
treecode-force-order ?= 24

UPWARDKERNELS_POTENTIAL = upward-kernels-order$(treecode-potential-order)
UPWARDKERNELS_FORCE=upward-kernels-order$(treecode-force-order)

REFERENCE_UPWARDKERNELS_POTENTIAL = reference-upward-kernels-order$(treecode-potential-order)
REFERENCE_UPWARDKERNELS_FORCE=reference-upward-kernels-order$(treecode-force-order)

OBJS =  potential-kernels.o reference-potential-kernels.o force-kernels.o force-kernels-tiled.o downward-kernels.o \
	$(UPWARDKERNELS_POTENTIAL).o  $(REFERENCE_UPWARDKERNELS_POTENTIAL).o

ifneq "$(treecode-potential-order)" "$(treecode-force-order)"
	OBJS += $(UPWARDKERNELS_FORCE).o $(REFERENCE_UPWARDKERNELS_FORCE).o
endif

M4FLAGS = -D realtype=$(real)

FORCEKERNELSM4 = force-kernels-tiled-V4.m4
ifeq ($(real),double)
FORCEKERNELSM4 = force-kernels-tiled-$(backend).m4
endif

KERNELSFLAGS = -std=c99 -O4 -DNDEBUG -ftree-vectorize -Wno-psabi \
	 -march=native -mtune=native -fassociative-math -ffast-math

kernels: $(OBJS)

%.o: %.c
	$(CC) $(KERNELSFLAGS) -c $^

%.o: %.ispc
	ispc  $^ -O3 --arch=x86-64 --cpu=core-avx2  --target=avx2-i64x4 --math-lib=fast --opt=disable-assertions --opt=fast-math  -o $@
#	ispc  $^ -O2 --arch=x86-64 --target=avx2-i64x4 --emit-asm -o $@.s

potential-kernels.ispc: potential-kernels.m4 unroll.m4 Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-potential-order) $<  > $@

reference-potential-kernels.c: potential-kernels-sse.m4 potential-kernels.h unroll.m4 Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-potential-order) $<  > $@

force-kernels.c: force-kernels-sse.m4 force-kernels.h Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-force-order) $<  > force-kernels.c

force-kernels-tiled.c: force-kernels-tiled-sse.m4 force-kernels.h Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-force-order) $<  > force-kernels-tiled.c

downward-kernels.c:  downward-kernels-sse.m4 downward-kernels.h Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-force-order) $<  > downward-kernels.c

$(UPWARDKERNELS_POTENTIAL).ispc: upward-kernels.m4 upward-kernels.h unroll.m4 Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-potential-order) $< > $@

$(UPWARDKERNELS_FORCE).ispc: upward-kernels.m4 upward-kernels.h unroll.m4  Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-force-order) $< > $@

$(REFERENCE_UPWARDKERNELS_POTENTIAL).c: reference-upward-kernels.m4 upward-kernels.h unroll.m4 Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-potential-order) $< > $@

$(REFERENCE_UPWARDKERNELS_FORCE).c: reference-upward-kernels.m4 upward-kernels.h unroll.m4  Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-force-order) $< > $@




clean:
	rm -f *.o *kernels*.c *kernels*.ispc

.PHONY = clean

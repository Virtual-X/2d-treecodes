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

NVCC ?= nvcc
CC = gcc -std=c99
backend ?= sse
real ?= double
treecode-potential-order ?= 12
treecode-force-order ?= 24

UPWARDKERNELS_POTENTIAL = upward-kernels-order$(treecode-potential-order)
UPWARDKERNELS_POTENTIAL_REF = upward-kernels-order$(treecode-potential-order)-ref

UPWARDKERNELS_FORCE=upward-kernels-order$(treecode-force-order)

OBJS =  potential-kernels.o force-kernels.o force-kernels-tiled.o downward-kernels.o $(UPWARDKERNELS_POTENTIAL).o $(UPWARDKERNELS_POTENTIAL_REF).o

ifneq "$(treecode-potential-order)" "$(treecode-force-order)"
	OBJS += $(UPWARDKERNELS_FORCE).o 
endif

M4FLAGS = -D realtype=$(real)

FORCEKERNELSM4 = force-kernels-tiled-V4.m4
ifeq ($(real),double)
FORCEKERNELSM4 = force-kernels-tiled-$(backend).m4
endif

KERNELSFLAGS = -std=c99 -O4 -DNDEBUG -ftree-vectorize -Wno-psabi \
	 -march=native -mtune=native -fassociative-math -ffast-math

kernels: $(OBJS)

$(UPWARDKERNELS_POTENTIAL).o : $(UPWARDKERNELS_POTENTIAL).cu
	$(NVCC) -O3 --use_fast_math -arch=sm_35 -dc -c $^

%.o: %.c
	$(CC) $(KERNELSFLAGS) -c $^
#module load intel ; icpc -march=native -mtune=native -Ofast -fno-alias -fno-rtti -fno-exceptions -c $^ ; module unload intel;

potential-kernels.c: potential-kernels-sse.m4 potential-kernels.h unroll.m4 Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-potential-order) $<  > potential-kernels.c

force-kernels.c: force-kernels-sse.m4 force-kernels.h Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-force-order) $<  > force-kernels.c

force-kernels-tiled.c: force-kernels-tiled-sse.m4 force-kernels.h Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-force-order) $<  > force-kernels-tiled.c

downward-kernels.c:  downward-kernels-sse.m4 downward-kernels.h Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-force-order) $<  > downward-kernels.c

$(UPWARDKERNELS_POTENTIAL).cu: upward-kernels-cuda.m4 upward-kernels.h unroll.m4 Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-potential-order) $< > $@

$(UPWARDKERNELS_POTENTIAL_REF).c: upward-kernels-sse.m4 upward-kernels.h unroll.m4 Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-potential-order) $< > $@

$(UPWARDKERNELS_FORCE).c: upward-kernels-sse.m4 upward-kernels.h unroll.m4  Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-force-order) $< > $(UPWARDKERNELS_FORCE).c

clean:
	rm -f *.o *kernels*.c *kernels*.cu

.PHONY = clean

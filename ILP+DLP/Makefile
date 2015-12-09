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
backend ?= avx
real ?= double
treecode-potential-order ?= 12
treecode-force-order ?= 24

UPWARDKERNELS_POTENTIAL = upward-kernels-order$(treecode-potential-order)
UPWARDKERNELS_FORCE=upward-kernels-order$(treecode-force-order)

OBJS =  potential-kernels.o force-kernels.o force-kernels-tiled.o downward-kernels.o $(UPWARDKERNELS_POTENTIAL).o

ifneq "$(treecode-potential-order)" "$(treecode-force-order)"
	OBJS += $(UPWARDKERNELS_FORCE).o 
endif

M4FLAGS = -D realtype=$(real)

FORCEKERNELSM4 = force-kernels-tiled-V4.m4
ifeq ($(real),double)
FORCEKERNELSM4 = force-kernels-tiled-$(backend).m4
endif

KERNELSFLAGS = -O4 -DNDEBUG -ftree-vectorize -Wno-psabi \
	 -march=native -mtune=native -fassociative-math -ffast-math

kernels: $(OBJS)

potential-kernels.o: potential-kernels.c
	$(CC) $(KERNELSFLAGS) -c $^

force-kernels.o: force-kernels.c
	$(CC) $(KERNELSFLAGS) -c $^

force-kernels-tiled.o: force-kernels-tiled.c
	module load intel ; icpc -march=native -mtune=native -Ofast -fno-alias -fno-rtti -fno-exceptions -c $^ ; module unload intel
	#$(CC) $(KERNELSFLAGS) -c $^

downward-kernels.o: downward-kernels.c 
	#module load intel ; icpc -march=native -mtune=native -Ofast -fno-alias -fno-rtti -fno-exceptions -c $^ ; module unload intel
	$(CC) $(KERNELSFLAGS) -c $^

$(UPWARDKERNELS_POTENTIAL).o: $(UPWARDKERNELS_POTENTIAL).c 
	$(CC) $(KERNELSFLAGS) -c $^

$(UPWARDKERNELS_FORCE).o: $(UPWARDKERNELS_FORCE).c 
	$(CC) $(KERNELSFLAGS) -c $^

potential-kernels.c: potential-kernels.m4 potential-kernels.h unroll.m4 Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-potential-order) potential-kernels.m4  > potential-kernels.c

force-kernels.c: force-kernels.m4 force-kernels.h Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-force-order) force-kernels.m4  > force-kernels.c

force-kernels-tiled.c: $(FORCEKERNELSM4) force-kernels.h Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-force-order) $(FORCEKERNELSM4) | indent > force-kernels-tiled.c

downward-kernels.c:  downward-kernels.m4 downward-kernels.h Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-force-order) downward-kernels.m4 | indent > downward-kernels.c

$(UPWARDKERNELS_POTENTIAL).c: upward-kernels.m4 upward-kernels.h unroll.m4 Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-potential-order) upward-kernels.m4  > $(UPWARDKERNELS_POTENTIAL).c

$(UPWARDKERNELS_FORCE).c: upward-kernels.m4 upward-kernels.h unroll.m4  Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-force-order) upward-kernels.m4  > $(UPWARDKERNELS_FORCE).c

clean:
	rm -f *.o *kernels*.c

.PHONY = clean

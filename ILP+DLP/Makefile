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

CC = gcc
ICPC = /cluster/apps/intel/composer_xe_2015.0.090/composer_xe_2015.0.090/bin/intel64/icpc

real ?= double
treecode-potential-order ?= 12
treecode-force-order ?= 24

UPWARDKERNELS_POTENTIAL = upward-kernels-order$(treecode-potential-order)
UPWARDKERNELS_FORCE=upward-kernels-order$(treecode-force-order)

OBJS =  potential-kernels.o force-kernels.o force-kernels-tiled.o $(UPWARDKERNELS_POTENTIAL).o

ifneq "$(treecode-potential-order)" "$(treecode-force-order)"
	OBJS += $(UPWARDKERNELS_FORCE).o 
endif

M4FLAGS = -D realtype=$(real)
AVXSUPPORT = $(shell cat /proc/cpuinfo | egrep avx)
ifeq ($(real),double)
ifneq ($(AVXSUPPORT),"") 
#M4FLAGS += -D TUNED4AVXDP=1
endif
endif

KERNELSFLAGS = -O4 -DNDEBUG  -ftree-vectorize \
	-std=c99 -march=native -mtune=native -fassociative-math -ffast-math

define INSTRUCTIONCOUNT
	@echo "INSTRUCTION COUNT $2/$1:" \
	$(shell OPT=$$(objdump -S $2 |  \
		egrep -n -i "(END_$1|<$1>)" | tr "\n" " " | \
		awk 'BEGIN{ FS=":"} {printf( "%d,%dp", $$1+1,$$3-1);}') ; \
		objdump -S $2 | sed -n $${OPT} | cut -d $$'\t' -f3 | sed '/^$$/d' | wc -l );
endef

kernels: $(OBJS)

potential-kernels.o: potential-kernels.c
	$(CC) $(KERNELSFLAGS) -c $^

force-kernels.o: force-kernels.c
	$(CC) $(KERNELSFLAGS) -c $^

force-kernels-tiled.o: force-kernels-tiled.c
	module load intel ; icpc -march=native -mtune=native -Ofast -fno-alias -fno-rtti -fno-exceptions -c $^ ; module unload intel

$(UPWARDKERNELS_POTENTIAL).o: $(UPWARDKERNELS_POTENTIAL).c 
	$(CC) $(KERNELSFLAGS) -c $^

$(UPWARDKERNELS_FORCE).o: $(UPWARDKERNELS_FORCE).c 
	$(CC) $(KERNELSFLAGS) -c $^

potential-kernels.c: potential-kernels.m4 potential-kernels.h unroll.m4 Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-potential-order) potential-kernels.m4  > potential-kernels.c

force-kernels.c: force-kernels.m4 force-kernels.h Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-force-order) force-kernels.m4  > force-kernels.c

force-kernels-tiled.c: force-kernels-tiled-sse.m4 force-kernels.h Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-force-order) force-kernels-tiled-sse.m4 | indent > force-kernels-tiled.c

$(UPWARDKERNELS_POTENTIAL).c: upward-kernels.m4 upward-kernels.h unroll.m4 Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-potential-order) upward-kernels.m4  > $(UPWARDKERNELS_POTENTIAL).c

$(UPWARDKERNELS_FORCE).c: upward-kernels.m4 upward-kernels.h unroll.m4  Makefile
	m4 $(M4FLAGS) -D ORDER=$(treecode-force-order) upward-kernels.m4  > $(UPWARDKERNELS_FORCE).c

clean:
	rm -f *.o *kernels*.c

.PHONY = clean

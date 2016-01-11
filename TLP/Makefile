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
real ?= double

treecode-potential-order ?= 12
treecode-force-order ?= 24
mrag-blocksize ?= 32

UPWARDKERNELS_POTENTIAL = upward-kernels-order$(treecode-potential-order)
UPWARDKERNELS_FORCE=upward-kernels-order$(treecode-force-order)


OBJS = treecode-potential.o treecode-force.o upward.o

#config ?=release

TLPFLAGS = -I../ILP+DLP/ -std=c++11 -DREAL=$(real) -DBLOCKSIZE=$(mrag-blocksize)
NVCCFLAGS := $(TLPFLAGS) -arch=sm_30 -Xcompiler -fopenmp
TLPFLAGS += -march=native -fopenmp


ifeq "$(config)" "release"
	TLPFLAGS += -O3 -DNDEBUG
	NVCCFLAGS += -O3 -DNDEBUG
else
	TLPFLAGS += -g -G
endif

ifeq "$(gprof)" "1"
	TLPFLAGS += -pg
endif

drivers: $(OBJS)

treecode-potential.o: treecode-potential.cu treecode.h Makefile
	$(NVCC) $(NVCCFLAGS) -DORDER=$(treecode-potential-order) -c $<

treecode-force.o: treecode-force.cu treecode.h Makefile
	$(NVCC) $(NVCCFLAGS) \
	$(shell ../ILP+DLP/instruction-count.sh  FORCE_E2P_TILED ../ILP+DLP/force-kernels-tiled.o) \
	-DORDER=$(treecode-force-order) -c $<

upward.o: upward.cu upward.h Makefile
	$(NVCC) $(NVCCFLAGS) -c $<

clean:
	rm -f *.o 

.PHONY = clean

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

real ?= double
order ?= 12
mrag-blocksize ?= 32
#config ?=release

TLPFLAGS = -I../ILP+DLP/ -std=c++11 -DREAL=$(real) -DBLOCKSIZE=$(mrag-blocksize) -DORDER=$(order) -DLMAX=15 -O3
NVCCFLAGS := $(TLPFLAGS) -arch=sm_30 -Xcompiler -fopenmp -Xcompiler '-fPIC' -Xcompiler '-fvisibility=hidden'
TLPFLAGS += -march=native -fopenmp -fvisibility=hidden

ifeq "$(config)" "release"
	TLPFLAGS +=  -DNDEBUG
	NVCCFLAGS += -DNDEBUG
else
#TLPFLAGS += -g -G
endif

ifeq "$(profiling)" "1"
	TLPFLAGS += -pg
	NVCCFLAGS += -lineinfo
endif

potential: treecode-potential.o order$(order)-upward.o sort-sources.o

force: treecode-force.o order$(order)-upward.o

treecode-potential.o: treecode-potential.cu treecode-potential.h Makefile
	nvcc $(NVCCFLAGS) -dc -c $<

treecode-force.o: treecode-force.cu treecode-force.h Makefile
	nvcc $(NVCCFLAGS) \
	$(shell ../ILP+DLP/instruction-count.sh  FORCE_E2P_TILED ../ILP+DLP/order$(order)-force-kernels-tiled.o) \
	-c $<

order$(order)-upward.o: upward.cu upward.h Makefile
	nvcc $(NVCCFLAGS) -dc -c $< -o $@

sort-sources.o: sort-sources.cu sort-sources.h Makefile
	nvcc $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f *.o 

.PHONY = clean potential force

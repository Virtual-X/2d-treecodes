#
# TLP/Makefile
# Part of 2d-treecodes
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
#profiling = 1

TLPFLAGS = -I../ILP+DLP/ -std=c++11 -DREAL=$(real) -DBLOCKSIZE=$(mrag-blocksize) -DORDER=$(order) -DLMAX=15 -O3
NVCCFLAGS := $(TLPFLAGS) -arch=compute_35 -code=sm_35 -Xcompiler -fopenmp -Xcompiler '-fPIC' -Xcompiler '-fvisibility=hidden' -Xcompiler -O4  -Xcompiler  -fassociative-math  -Xcompiler  -ffast-math
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

potential: treecode-potential.o order$(order)-upward.o sort-sources.o driver.o

force: treecode-force.o order$(order)-upward.o sort-sources.o

treecode-potential.o: treecode-potential.cu treecode-potential.h Makefile
	nvcc $(NVCCFLAGS)  -dc -c $<

driver.o: driver.cpp
	g++ $(TLPFLAGS) -fPIC -c $<

treecode-force.o: treecode-force.cu treecode-force.h Makefile
	nvcc $(NVCCFLAGS) -dc -c $<

order$(order)-upward.o: upward.cu upward.h Makefile
	nvcc $(NVCCFLAGS) -dc -c $< -o $@

sort-sources.o: sort-sources.cu sort-sources.h Makefile
	nvcc $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f *.o 

.PHONY = clean potential force

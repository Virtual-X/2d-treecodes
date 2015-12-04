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

CXX ?= g++
real ?= double

treecode-potential-order ?= 12
treecode-force-order ?= 24
mrag-blocksize ?= 32

UPWARDKERNELS_POTENTIAL = upward-kernels-order$(treecode-potential-order)
UPWARDKERNELS_FORCE=upward-kernels-order$(treecode-force-order)

OBJS = treecode-potential.o treecode-force.o upward.o

config ?= release

TLPFLAGS = -I../ILP+DLP/ -std=c++11 -march=native -fopenmp -DREAL=$(real) -DBLOCKSIZE=$(mrag-blocksize)

ifeq "$(config)" "release"
	TLPFLAGS += -O3 -DNDEBUG 
else
	TLPFLAGS += -g
endif

ifeq "$(gprof)" "1"
	CXXFLAGS += -pg
	TLPFLAGS += -pg
endif

drivers: $(OBJS)

treecode-potential.o: treecode-potential.cpp treecode.h Makefile
	$(CXX) $(TLPFLAGS) -DORDER=$(treecode-potential-order) -c $<

treecode-force.o: treecode-force.cpp treecode.h Makefile
	$(CXX) $(TLPFLAGS) -DORDER=$(treecode-force-order) -c $<

upward.o: upward.cpp upward.h Makefile
	$(CXX) $(TLPFLAGS) -c $<

clean:
	rm -f *.o 

.PHONY = clean

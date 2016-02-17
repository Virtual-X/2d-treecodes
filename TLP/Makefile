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
CXX ?= g++

CXXFLAGS = -I../ILP+DLP/ -std=c++11 -fPIC
CXXFLAGS += -DREAL=$(real) -DBLOCKSIZE=$(mrag-blocksize) -DORDER=$(order) -DLMAX=15
CXXFLAGS += -O3 -march=native -fopenmp -fvisibility=hidden  

ifeq "$(config)" "release"
	TLPFLAGS +=  -DNDEBUG
else
#TLPFLAGS += -g -G
endif

ifeq "$(profiling)" "1"
	TLPFLAGS += -pg
endif

potential: treecode-potential.o order$(order)-upward.o sort-sources.o

force: treecode-force.o order$(order)-upward.o sort-sources.o

treecode-potential.o: treecode-potential.cpp treecode-potential.h Makefile
	$(CXX) $(CXXFLAGS) -c $<

treecode-force.o: treecode-force.cpp treecode-force.h Makefile
	$(CXX) $(CXXFLAGS) -c $<

order$(order)-upward.o: upward.cpp upward.h Makefile
	$(CXX) $(CXXFLAGS) -c $< -o $@

sort-sources.o: sort-sources.cpp sort-sources.h Makefile
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o 

.PHONY = clean potential force

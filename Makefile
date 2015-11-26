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
CC = gcc

real ?= double
treecode-potential-order ?= 12
treecode-force-order ?= 24

UPWARDKERNELS_POTENTIAL = upward-kernels-order$(treecode-potential-order)
UPWARDKERNELS_FORCE=upward-kernels-order$(treecode-force-order)

OBJS = treecode-potential.o treecode-force.o upward.o potential-kernels.o force-kernels.o $(UPWARDKERNELS_POTENTIAL).o

ifneq "$(treecode-potential-order)" "$(treecode-force-order)"
	OBJS += $(UPWARDKERNELS_FORCE).o 
endif

config ?= release

CXXFLAGS = -std=c++11 -g -D_GLIBCXX_DEBUG -fopenmp -DREAL=$(real)  -DORDER=$(treecode-potential-order)
TLPFLAGS = -std=c++11 -march=native -fopenmp -DREAL=$(real) 

ifeq "$(config)" "release"
	TLPFLAGS += -O3 -DNDEBUG 
else
	TLPFLAGS += $(CXXFLAGS)
endif

KERNELSFLAGS =  -O4 -DNDEBUG  -ftree-vectorize \
	-std=c99 -march=native -mtune=native -fassociative-math -ffast-math \
	-ftree-vectorizer-verbose=0

ifeq "$(gprof)" "1"
	CXXFLAGS += -pg
	TLPFLAGS += -pg
endif

test: main.cpp treecode.a
	$(CXX) $(CXXFLAGS) $^ -g -o test

treecode.a: $(OBJS) treecode.h
	ar rcs treecode.a $(OBJS)

treecode-potential.o: treecode-potential.cpp treecode.h Makefile
	$(CXX) $(TLPFLAGS) -DORDER=$(treecode-potential-order) -c $<

treecode-force.o: treecode-force.cpp treecode.h Makefile
	$(CXX) $(TLPFLAGS) -DORDER=$(treecode-force-order) -c $<

upward.o: upward.cpp upward.h Makefile
	$(CXX) $(TLPFLAGS) -c $<

potential-kernels.o: potential-kernels.c
	$(CC) $(KERNELSFLAGS) -c $^

potential-kernels.c: potential-kernels.m4 potential-kernels.h unroll.m4 Makefile
	m4 -D ORDER=$(treecode-potential-order) -D realtype=$(real) potential-kernels.m4 | indent > potential-kernels.c

force-kernels.o: force-kernels.c
	$(CC) $(KERNELSFLAGS) -c $^

force-kernels.c: force-kernels.m4 force-kernels.h Makefile
	m4 -D ORDER=$(treecode-force-order) -D realtype=$(real) force-kernels.m4 | indent > force-kernels.c

$(UPWARDKERNELS_POTENTIAL).o: $(UPWARDKERNELS_POTENTIAL).c 
	$(CC) $(KERNELSFLAGS) -c $^

$(UPWARDKERNELS_FORCE).o: $(UPWARDKERNELS_FORCE).c 
	$(CC) $(KERNELSFLAGS) -c $^

$(UPWARDKERNELS_POTENTIAL).c: upward-kernels.m4 upward-kernels.h unroll.m4 Makefile
	m4 -D ORDER=$(treecode-potential-order) -D realtype=$(real) upward-kernels.m4 | indent > $(UPWARDKERNELS_POTENTIAL).c

$(UPWARDKERNELS_FORCE).c: upward-kernels.m4 upward-kernels.h unroll.m4  Makefile
	m4 -D ORDER=$(treecode-force-order) -D realtype=$(real) upward-kernels.m4 | indent > $(UPWARDKERNELS_FORCE).c

clean:
	rm -f test *.o *.a potential-kernels*.c upward-kernels*.c

.PHONY = clean

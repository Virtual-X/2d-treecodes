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

OBJS = upward.o treecode-potential.o potential-kernels.o force-kernels.o $(UPWARDKERNELS_POTENTIAL).o

ifneq "$(treecode-potential-order)" "$(treecode-force-order)"
	OBJS += $(UPWARDKERNELS_FORCE).o
endif

config ?= release

CXXFLAGS = -std=c++11 -g -D_GLIBCXX_DEBUG -fopenmp -DREAL=$(real)  -DORDER=$(treecode-potential-order)
TREECODEFLAGS =  -std=c++11 -march=native -fopenmp -DREAL=$(real)  -DORDER=$(treecode-potential-order)

ifeq "$(config)" "release"
	TREECODEFLAGS += -O3 -DNDEBUG -ftree-vectorize
else
	TREECODEFLAGS += $(CXXFLAGS)
endif

KERNELSFLAGS =  -O4 -DNDEBUG  -ftree-vectorize \
	-std=c99 -march=native -mtune=native -fassociative-math -ffast-math \
	-ftree-vectorizer-verbose=0

ifeq "$(gprof)" "1"
	CXXFLAGS += -pg
	TREECODEFLAGS += -pg
endif

test: main.cpp $(OBJS) treecode.h
	echo OBJS ARE $(OBJS)
	$(CXX) $(CXXFLAGS) main.cpp $(OBJS) -g -o test
	ar rcs treecode.a $(OBJS)

treecode-potential.o: treecode-potential.cpp treecode.h Makefile
	$(CXX) $(TREECODEFLAGS) -c $<

upward.o: upward.cpp upward.h Makefile
	echo OBJS ARE $(OBJS)
	$(CXX) $(TREECODEFLAGS) -c $<

potential-kernels.o: potential-kernels.c
	$(CC) $(KERNELSFLAGS) -c $^

potential-kernels.c: potential-kernels.mc potential-kernels.h Makefile
	m4 -D ORDER=$(treecode-potential-order) -D realtype=$(real) potential-kernels.mc | indent > potential-kernels.c

force-kernels.o: force-kernels.c
	$(CC) $(KERNELSFLAGS) -c $^

force-kernels.c: force-kernels.mc force-kernels.h Makefile
	m4 -D ORDER=$(treecode-force-order) -D realtype=$(real) force-kernels.mc | indent > force-kernels.c

$(UPWARDKERNELS_POTENTIAL).o: $(UPWARDKERNELS_POTENTIAL).c 
	$(CC) $(KERNELSFLAGS) -c $^

$(UPWARDKERNELS_FORCE).o: $(UPWARDKERNELS_FORCE).c 
	$(CC) $(KERNELSFLAGS) -c $^

$(UPWARDKERNELS_POTENTIAL).c: upward-kernels.mc upward-kernels.h Makefile
	m4 -D ORDER=$(treecode-potential-order) -D realtype=$(real) upward-kernels.mc | indent > $(UPWARDKERNELS_POTENTIAL).c

$(UPWARDKERNELS_FORCE).c: upward-kernels.mc upward-kernels.h Makefile
	m4 -D ORDER=$(treecode-force-order) -D realtype=$(real) upward-kernels.mc | indent > $(UPWARDKERNELS_FORCE).c

clean:
	rm -f test *.o *.a potential-kernels*.c upward-kernels*.c

.PHONY = clean

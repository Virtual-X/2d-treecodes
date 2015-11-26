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

UPWARDKERNELS=upward-kernels-order$(treecode-potential-order)

config ?= release

CXXFLAGS = -std=c++11 -g -D_GLIBCXX_DEBUG -fopenmp -DREAL=$(real)
TREECODEFLAGS = -DORDER=$(treecode-potential-order) -DREAL=$(real) -std=c++11 -march=native -fopenmp

ifeq "$(config)" "release"
	TREECODEFLAGS += -O3 -DNDEBUG -ftree-vectorize
else
	TREECODEFLAGS += $(CXXFLAGS)
endif

KERNELSFLAGS = -DORDER=$(treecode-potential-order) -DREAL=$(real) -O4 -DNDEBUG  -ftree-vectorize \
	-std=c99 -march=native -mtune=native -fassociative-math -ffast-math \
	-ftree-vectorizer-verbose=0

ifeq "$(gprof)" "1"
	CXXFLAGS += -pg
	TREECODEFLAGS += -pg
endif

test: main.cpp treecode.o potential-kernels.o $(UPWARDKERNELS).o treecode.h
	$(CXX) $(CXXFLAGS)  main.cpp treecode.o potential-kernels.o $(UPWARDKERNELS).o -g -o test
	ar rcs treecode.a treecode.o potential-kernels.o $(UPWARDKERNELS).o 

treecode.o: treecode.cpp treecode.h Makefile
	$(CXX) $(TREECODEFLAGS) -c $<

potential-kernels.o: potential-kernels.c
	$(CC) $(KERNELSFLAGS) -c $^

potential-kernels.c: potential-kernels.mc potential-kernels.h Makefile
	m4 -D ORDER=$(treecode-potential-order) potential-kernels.mc | indent > potential-kernels.c

$(UPWARDKERNELS).o: $(UPWARDKERNELS).c 
	$(CC) $(KERNELSFLAGS) -c $^

$(UPWARDKERNELS).c: upward-kernels.mc upward-kernels.h Makefile
	m4 -D ORDER=$(treecode-potential-order) upward-kernels.mc | indent > $(UPWARDKERNELS).c

clean:
	rm -f test *.o *.a potential-kernels*.c upward-kernels*.c

.PHONY = clean

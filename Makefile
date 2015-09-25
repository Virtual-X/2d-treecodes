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

treecode-potential-order ?= 12

CXXFLAGS = -std=c++11 -g -D_GLIBCXX_DEBUG -fopenmp 

TREECODEFLAGS = -DORDER=$(treecode-potential-order) -std=c++11 -march=native

ifeq "$(config)" "release"
	TREECODEFLAGS += -O3 -DNDEBUG -ftree-vectorize
else
	TREECODEFLAGS += $(CXXFLAGS)
endif

KERNELSFLAGS = -DORDER=$(treecode-potential-order) -O4 -DNDEBUG  -ftree-vectorize \
	-std=c99 -march=native -mtune=native -fassociative-math -ffast-math \
	-ftree-vectorizer-verbose=0 

ifeq "$(gprof)" "1"
	CXXFLAGS += -pg
	TREECODEFLAGS += -pg
endif

test: main.cpp treecode.o treecode-kernels.o treecode.h
	$(CXX) $(CXXFLAGS)  main.cpp treecode.o treecode-kernels.o -g -o test
	ar rcs treecode.a treecode.o treecode-kernels.o

treecode.o: treecode.cpp treecode.h Makefile
	$(CXX) $(TREECODEFLAGS) -c $<

treecode-kernels.o: treecode-kernels.c treecode.h Makefile
	$(CC) $(KERNELSFLAGS) -c $<

clean:
	rm -f test *.o *.a

.PHONY = clean

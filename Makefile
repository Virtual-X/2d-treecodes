CXX ?= g++
CC = gcc

treecode-potential-order ?= 12

CXXFLAGS = -std=c++11 -g -D_GLIBCXX_DEBUG -fopenmp

TREECODEFLAGS = -DORDER=$(treecode-potential-order) -std=c++11

ifeq "$(config)" "release"
	TREECODEFLAGS += -O3 -DNDEBUG -ftree-vectorize
else
	TREECODEFLAGS += $(CXXFLAGS)
endif

ifeq "$(gprof)" "1"
	CXXFLAGS += -pg
	TREECODEFLAGS += -pg
endif

test: main.cpp treecode.o treecode-kernels.o treecode.h
	$(CXX) $(CXXFLAGS)  main.cpp treecode.o treecode-kernels.o -g -o test
	ar rcs treecode.a treecode.o

treecode.o: treecode.cpp treecode.h Makefile
	$(CXX) $(TREECODEFLAGS) -c $<

treecode-kernels.o: treecode-kernels.c treecode.h Makefile
	$(CC) -O4 -DNDEBUG -ftree-vectorize -std=c99 -march=native -mtune=native -fassociative-math -ffast-math -ftree-vectorizer-verbose=1 -c $<

clean:
	rm -f test *.o

.PHONY = clean

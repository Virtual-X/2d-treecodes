CXX ?= g++

fmm-order-pressure ?= 12

CXXFLAGS = -std=c++11 -g -D_GLIBCXX_DEBUG -fopenmp

TREECODEFLAGS = -DORDER=$(fmm-order-pressure) -std=c++11

ifeq "$(config)" "release"
	TREECODEFLAGS += -O3 -DNDEBUG -ftree-vectorize
else
	TREECODEFLAGS += $(CXXFLAGS)
endif

ifeq "$(gprof)" "1"
	CXXFLAGS += -pg
	TREECODEFLAGS += -pg
endif

test: main.cpp treecode.o treecode.h
	$(CXX) $(CXXFLAGS)  main.cpp treecode.o -g -o test

treecode.o: treecode.cpp treecode.h Makefile
	$(CXX) $(TREECODEFLAGS) -c $<

clean:
	rm -f test *.o

.PHONY = clean

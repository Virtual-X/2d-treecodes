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

potential-order=12
force-order=24
mrag-blocksize=32

TESTOPT = real=double mrag-blocksize=32

NVCCFLAGS = -std=c++11 -Xcompiler -fopenmp \
	-Drealtype=double \
	-DPOTENTIAL_ORDER=$(potential-order) \
	-DFORCE_ORDER=$(force-order) \
	-DBLOCKSIZE=$(mrag-blocksize) -lcudart

ifeq "$(gprof)" "1"
	TLPFLAGS += -pg
endif

test: main.cpp libraries
	nvcc $(NVCCFLAGS) -g $< libtreecode-potential.so libtreecode-force.so  -o test 

libraries:
	make -f libraries.Makefile order=$(potential-order) libtreecode-potential.so
	make -f libraries.Makefile order=$(force-order) mrag-blocksize=32 libtreecode-force.so

clean:
	rm -f test
	make -f libraries.Makefile clean

.PHONY = clean

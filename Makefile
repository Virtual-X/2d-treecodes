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
config ?= release

CXXFLAGS = -std=c++11  -fopenmp -Drealtype=$(real)  -DORDER=$(treecode-potential-order) -DBLOCKSIZE=$(mrag-blocksize)

ifeq "$(gprof)" "1"
	CXXFLAGS += -pg
	TLPFLAGS += -pg
endif

define INSTRUCTIONCOUNT
	@echo "INSTRUCTION COUNT $2/$1:" \
	$(shell OPT=$$(objdump -S ILP+DLP/$2 |  \
		egrep -n -i "(END_$1|<$1>)" | tr "\n" " " | \
		awk 'BEGIN{ FS=":"} {printf( "%d,%dp", $$1+1,$$3-1);}') ; \
		objdump -S $2 | sed -n $${OPT} | cut -d $$'\t' -f3 | sed '/^$$/d' | wc -l );
endef

test: main.cpp treecode.a
	$(CXX) $(CXXFLAGS) $^ -g -o test

treecode.a: $(OBJS) TLP/treecode.h kernels drivers header
	ar rcs treecode.a TLP/*.o ILP+DLP/*.o
	$(call INSTRUCTIONCOUNT,FORCE_E2P,force-kernels.o)
	$(call INSTRUCTIONCOUNT,FORCE_E2P_TILED,force-kernels-tiled.o)

header:
	m4 -D realtype=$(real) TLP/treecode.h | sed '/typedef/d'  > treecode.h

kernels:
	make -C ILP+DLP kernels

drivers: 
	make -C TLP drivers

clean:
	rm -f test treecode.a treecode.h
	make -C TLP clean
	make -C ILP+DLP clean

.PHONY = clean kernels drivers header

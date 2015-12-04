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
	$(shell OPT=$$(objdump -S $2 |egrep -n -i "(END_$1|<$1>)" | tr "\n" " " |awk 'BEGIN{ FS=":"} {printf( "%d,%dp", $$1+1,$$3-1);}');RESULT=$$(objdump -S $2 | sed -n $${OPT} | cut -d $$'\t' -f3 | sed '/^$$/d' | wc -l); echo $$RESULT | tr -d " \t\n\r")
endef

test: main.cpp treecode.a
	$(CXX) $(CXXFLAGS) $^ -g -o test

treecode.a: $(OBJS) TLP/treecode.h kernels drivers header
	ar rcs treecode.a TLP/*.o ILP+DLP/*.o

header:
	m4 -D realtype=$(real) TLP/treecode.h | sed '/typedef/d'  > treecode.h

drivers: kernels
	make -C TLP drivers \
	E2P_TILED_IC=$(shell ILP+DLP/instruction-count.sh  FORCE_E2P_TILED ILP+DLP/force-kernels-tiled.o) \
	E2P_IC=$(shell ILP+DLP/instruction-count.sh  FORCE_E2P ILP+DLP/force-kernels.o)

kernels:
	make -C ILP+DLP kernels

clean:
	rm -f test treecode.a treecode.h
	make -C TLP clean
	make -C ILP+DLP clean

.PHONY = clean header drivers kernels


	#@echo "INSTRUCTION COUNT $2/$1:" 
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
CC = gcc -std=c99

real ?= double
treecode-potential-order ?= 12
treecode-force-order ?= 24
mrag-blocksize ?= 32
config ?= release
backend ?= sse

CXXFLAGS = -std=c++11  -fopenmp -Drealtype=$(real)  -DORDER=$(treecode-potential-order) -DBLOCKSIZE=$(mrag-blocksize)

ifeq "$(gprof)" "1"
	CXXFLAGS += -pg
	TLPFLAGS += -pg
endif

test: main.cpp libtreecode.a
	$(CXX) $(CXXFLAGS) $^   -g -o test

libtreecode.a: $(OBJS) TLP/treecode.h kernels drivers header svml
	ar rcs libtreecode.a TLP/*.o ILP+DLP/*.o \
	svml/svml_d_atan22_iface_la.o \
	svml/svml_d_log2_iface_la.o \
	svml/svml_d_feature_flag_.o \
	svml/cpu_feature_disp.o \
	svml/svml_d_atan22_core_exla.o \
	svml/svml_d_atan22_core_h9la.o \
	svml/svml_d_log2_core_h9la.o \
	svml/svml_d_log2_core_exla.o \
	svml/svml_d_log2_core_e7la.o

svml: 
	$(shell mkdir svml ; cd svml; \
	ar -x /cluster/apps/intel/composer_xe_2015.0.090/composer_xe_2015.0.090/compiler/lib/intel64/libsvml.a; \
	ar -x /cluster/apps/intel/composer_xe_2015.0.090/composer_xe_2015.0.090/compiler/lib/intel64/libirc.a; \
	cd ..)

header:
	m4 -D realtype=$(real) TLP/treecode.h | sed '/typedef/d'  > treecode.h

drivers: kernels
	make -C TLP drivers CXX="$(CXX)" \
	E2P_TILED_IC=$(shell ILP+DLP/instruction-count.sh  FORCE_E2P_TILED ILP+DLP/force-kernels-tiled.o) \
	E2P_IC=$(shell ILP+DLP/instruction-count.sh  FORCE_E2P ILP+DLP/force-kernels.o)

kernels:
	make -C ILP+DLP kernels backend="$(backend)" CC="$(CC)"

clean:
	rm -f test libtreecode.a treecode.h
	make -C TLP clean
	make -C ILP+DLP clean

.PHONY = clean header drivers kernels

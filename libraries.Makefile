#
# libraries.Makefile
# Part of MRAG/2d-treecodes
#
# Created and authored by Diego Rossinelli on 2015-09-25.
# Copyright 2015. All rights reserved.
#
# Users are NOT authorized
# to employ the present software for their own publications
# before getting a written permission from the author of this file.
#

real ?= double
order ?= 12
mrag-blocksize ?= 32

OBJS = drivers/order$(order)-upward.o drivers/sort-sources.o

ifeq "$(MAKECMDGOALS)" "libtreecode-force.so"
	OBJS += drivers/treecode-force.o
	TARGET=force
else
	OBJS += drivers/treecode-potential.o
	TARGET=potential
endif

OBJS += $(wildcard kernels/order$(order)-*.o)

libtreecode-potential.so: drivers/treecode-potential.h alldrivers
	m4 -D realtype=$(real) drivers/treecode-potential.h | \
	sed '/typedef/d' | sed '/attribute/d' > treecode-potential.h
	g++ -shared -o $@ $(OBJS)

libtreecode-force.so: drivers/treecode-force.h alldrivers
	m4 -D realtype=$(real) drivers/treecode-force.h | \
	sed '/typedef/d' | sed '/attribute/d' > treecode-force.h
	g++ -shared -o $@ $(OBJS)

alldrivers: allkernels
	make -C drivers $(TARGET)

allkernels:
	make -C kernels $(TARGET)

clean:
	rm -f test *.so treecode-potential.h treecode-force.h
	make -C drivers clean
	make -C kernels clean


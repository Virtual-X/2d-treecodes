#
# libraries.Makefile
# Part of MRAG/2d-treecode-potential
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

OBJS = TLP/order$(order)-upward.o

ifeq "$(MAKECMDGOALS)" "libtreecode-force.so"
	OBJS += TLP/treecode-force.o
	TARGET=force
else
	OBJS += TLP/treecode-potential.o
	TARGET=potential
endif

OBJS += $(wildcard ILP+DLP/order$(order)-*.o)

libtreecode-potential.so: TLP/treecode-potential.h drivers
	m4 -D realtype=$(real) TLP/treecode-potential.h | sed '/typedef/d' > treecode-potential.h
	nvcc -arch=sm_30 --resource-usage -Xcompiler '-fPIC' -dlink $(OBJS) -o linkpot.o
	g++ -shared -o $@ $(OBJS) linkpot.o TLP/sort-sources.o -L/usr/local/cuda/lib64 -lcudart

libtreecode-force.so: TLP/treecode-force.h drivers
	m4 -D realtype=$(real) TLP/treecode-force.h | sed '/typedef/d' > treecode-force.h
	nvcc -arch=sm_30 -Xcompiler '-fPIC' -dlink $(OBJS) -o linkfor.o
	g++ -shared -o $@ $(OBJS) TLP/sort-sources.o linkfor.o -L/usr/local/cuda/lib64 -lcudart

drivers: kernels
	make -C TLP $(TARGET)

kernels:
	make -C ILP+DLP order=$(order) $(TARGET)

clean:
	rm -f test *.so treecode-potential.h treecode-force.h
	make -C TLP clean
	make -C ILP+DLP clean

.PHONY = clean drivers kernels

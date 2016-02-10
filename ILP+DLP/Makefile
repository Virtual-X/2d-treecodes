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

NVCC ?= nvcc
CC = gcc -std=c99
backend ?= sse
real ?= double
order ?= 12

OBJS =order$(order)-upward-kernels.cuda.o 

ifneq "$(MAKECMDGOALS)" "force"
	OBJS += order$(order)-potential-kernels.cuda.o
else
	OBJS += order$(order)-force-kernels.cuda.o 
endif

potential force: $(OBJS)

%.cuda.o: %.cu Makefile
	nvcc -Xcompiler '-fPIC' -Xcompiler '-fvisibility=hidden'  -maxrregcount=32  -O4 -arch=compute_35 -code=sm_35 -use_fast_math -dc -c $< -o $@

order$(order)-%.cu: %.m4 Makefile
	m4 -D realtype=$(real) -D ORDER=$(order) $< > $@

clean:
	rm -f *.o *kernels*.c *kernels*.cu

.PRECIOUS: order$(order)-%.cu order$(order)-%.o

.PHONY = clean potential force


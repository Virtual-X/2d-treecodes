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

OBJS = 	order$(order)-upward-kernels.cuda.o \
	order$(order)-upward-kernels-reference.o 

ifneq "$(MAKECMDGOALS)" "force"
	OBJS += order$(order)-potential-kernels.cuda.o \
		order$(order)-reference-potential-kernels.o 
else
	OBJS += order$(order)-force-kernels.o \
		order$(order)-force-kernels-tiled.o \
		order$(order)-downward-kernels.o 
endif

potential force: $(OBJS)

%.cuda.o: %.cu Makefile
	nvcc  -Xcompiler '-fPIC' -Xcompiler '-fvisibility=hidden' -O4 -arch=sm_30 -use_fast_math -dc -c $< -o $@

%.o: %.c Makefile
	gcc -fPIC -fvisibility=hidden -std=c99 -O4 -DNDEBUG -ftree-vectorize \
	 -march=native -mtune=native -fassociative-math -ffast-math -c $<

order$(order)-%.c order$(order)-%.cu: %.m4 Makefile
	m4 -D realtype=$(real) -D ORDER=$(order) $< > $@

clean:
	rm -f *.o *kernels*.c *kernels*.cu

.PRECIOUS: order$(order)-%.c order$(order)-%.cu order$(order)-%.o order$(order)-%.o

.PHONY = clean potential force


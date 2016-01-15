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

real ?= double
order ?= 12

OBJS = 	upward-kernels.cu.o \
		upward-kernels-reference.o \
		potential-kernels.o \
		force-kernels.o \
		force-kernels-tiled.o \
		downward-kernels.o 

kernels: $(OBJS)

%.cu.o: %.cu Makefile
	nvcc -O4 -arch=sm_30 -use_fast_math -dc -c $<

%.o: %.c Makefile
	gcc -std=c99 -O4 -DNDEBUG -ftree-vectorize \
	 -march=native -mtune=native -fassociative-math -ffast-math -c $<

%.c %.cu: %.m4 Makefile
	m4 -D realtype=$(real) -D ORDER=$(order) $< > $@

clean:
	rm -f *.o *kernels*.c *kernels*.cu

.PRECIOUS: %.c %.cu

.PHONY = clean

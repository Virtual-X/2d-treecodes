#
# Makefile
# Part of 2d-treecodes
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

OBJS = order$(order)-upward-kernels.o upward-common.o

ifneq "$(MAKECMDGOALS)" "force"
	OBJS += order$(order)-potential-kernels.o
else
	OBJS += order$(order)-force-kernels.o 
endif

potential force: $(OBJS)

%.o: %.ispc Makefile
	ispc  $< -Drealtype=$(real) -O3 --arch=x86-64 --math-lib=fast \
	--opt=disable-assertions --opt=fast-math \
	--cpu=core-avx2 --target=avx2-i64x4 \
	--wno-perf --pic -o $@

order$(order)-%.ispc: %.m4 Makefile
	m4 -D realtype=$(real) -D ORDER=$(order) $< > $@

clean:
	rm -f *.o *kernels*.c *kernels*.ispc

.PRECIOUS: order$(order)-%.ispc order$(order)-%.o

.PHONY = clean potential force

/*
 *  sort-sources.h
 *  Part of 2d-treecodes
 *
 *  Created and authored by Diego Rossinelli on 2015-11-25.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

extern "C"
__attribute__ ((visibility ("hidden")))
void sort_sources(const realtype * const xsrc,
		  const realtype * const ysrc,
		  const realtype * const vsrc,
		  const int nsrc,
		  int * const keysorted,
		  realtype * const xsorted,
		  realtype * const ysorted,
		  realtype * const vsorted,
		  realtype * const output_xmin,
		  realtype * const output_ymin,
		  realtype * const output_extent);

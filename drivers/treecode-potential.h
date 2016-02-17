/*
 *  treecode-potential.h
 *  Part of 2d-treecodes
 *
 *  Created and authored by Diego Rossinelli on 2015-09-25.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

typedef REAL realtype;

#ifdef __cplusplus
extern "C"
__attribute__ ((visibility ("default")))
#endif
void treecode_potential(const realtype theta,
			const realtype * const xsources,
			const realtype * const ysources,
			const realtype * const sourcevalues,
			const int nsources,
			const realtype * const xtargets,
			const realtype * const ytargets,
			const int ntargets,
			realtype * const targetvalues);

/*
 *  treecode-kernels.c
 *  Part of MRAG/2d-treecode-potential
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
{
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

    void treecode_force(const realtype theta,
			const realtype * const xsources,
			const realtype * const ysources,
			const realtype * const sourcevalues,
			const int nsources,
			const realtype * const xtargets,
			const realtype * const ytargets,
			const int ntargets,
			realtype * const xresult,
			realtype * const yresult);

    void treecode_force_mrag(const realtype theta,
			     const realtype * const xsources,
			     const realtype * const ysources,
			     const realtype * const sourcevalues,
			     const int nsources,
			     const realtype * const x0s,
			     const realtype * const y0s,
			     const realtype * const hs,
			     const int nblocks,
			     realtype * const xresults,
			     realtype * const yresults);

#ifdef __cplusplus
}
#endif

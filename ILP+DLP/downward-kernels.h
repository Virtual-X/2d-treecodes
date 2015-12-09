/*
 *  downward-kernels.h
 *  Part of MRAG/2d-treecode-potential
 *
 *  Created and authored by Diego Rossinelli on 2015-11-25.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

    void downward_e2l(const realtype * x0s,
		      const realtype * y0s,
		      const realtype * masses,
		      const realtype ** __restrict__ const vrexpansions,
		      const realtype ** __restrict__ const viexpansions,
		      const int nexpansions,
		      realtype * __restrict__ const rlocal,
		      realtype * __restrict__ const ilocal);

    void downward_l2p_tiled( const realtype rx,
			     const realtype ry,
			     const realtype h,
			     const realtype * __restrict__ const rlocal,
			     const realtype * __restrict__ const ilocal,
			     realtype * const xresult,
			     realtype * const yresult, const int stride);

#ifdef __cplusplus
}
#endif


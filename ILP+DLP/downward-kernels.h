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

    void downward_e2l(const realtype x0,
		      const realtype y0,
		      const realtype h,
		      const realtype mass,
		      const realtype * __restrict__ const rexpansions,
		      const realtype * __restrict__ const iexpansions,
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


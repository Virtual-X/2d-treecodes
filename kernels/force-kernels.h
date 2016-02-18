/*
 *  force-kernels.h
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

typedef REAL realtype;

extern "C"
{
    __attribute__ ((visibility ("hidden")))
    void force_p2p_8x8(const realtype * __restrict__ const xsources,
		       const realtype * __restrict__ const ysources,
		       const realtype * __restrict__ const sources,
		       const int nsources,
		       const realtype xt,
		       const realtype yt,
		       const realtype h,
		       realtype * const xresult,
		       realtype * const yresult);
    
    __attribute__ ((visibility ("hidden")))
    void force_e2p_8x8(const realtype mass,
		       const realtype rx,
		       const realtype ry,
		       const realtype h,
		       const realtype * __restrict__ const rxp,
		       const realtype * __restrict__ const ixp,
		       realtype * const xresult,
		       realtype * const yresult);
    
    __attribute__ ((visibility ("hidden")))
    void downward_e2l(const realtype * x0s,
		      const realtype * y0s,
		      const realtype * masses,
		      const realtype * __restrict__ * const vrexpansions,
		      const realtype * __restrict__ * const viexpansions,
		      const int nexpansions,
		      realtype * __restrict__ const rlocal,
		      realtype * __restrict__ const ilocal);

    __attribute__ ((visibility ("hidden")))
    void downward_l2p_8x8(const realtype rx,
			  const realtype ry,
			  const realtype h,
			  const realtype * __restrict__ const rlocal,
			  const realtype * __restrict__ const ilocal,
			  realtype * const xresult,
			  realtype * const yresult);
}


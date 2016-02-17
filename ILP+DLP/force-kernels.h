/*
 *  force-kernels.h
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

typedef REAL realtype;

#ifdef __cplusplus
extern "C"
{
#endif

    void force_p2p(const realtype * __restrict__ const xsources,
		   const realtype * __restrict__ const ysources,
		   const realtype * __restrict__ const sources,
		   const int nsources,
		   const realtype _xt,
		   const realtype _yt,
		   realtype * const xresult,
		   realtype * const yresult);

    void reference_force_p2p_tiled(const realtype * __restrict__ const xsources,
				   const realtype * __restrict__ const ysources,
				   const realtype * __restrict__ const sources,
				   const int nsources,
				   const realtype xt,
				   const realtype yt,
				   const realtype h,
				   realtype * const xresult,
				   realtype * const yresult, const int );

    
    void force_p2p_tiled(const realtype * __restrict__ const xsources,
			 const realtype * __restrict__ const ysources,
			 const realtype * __restrict__ const sources,
			 const int nsources,
			 const realtype xt,
			 const realtype yt,
			 const realtype h,
			 realtype * const xresult,
			 realtype * const yresult);

    void reference_force_p2p_tiled_mixprec(const float * __restrict__ const xsrc,
					   const float * __restrict__ const ysrc,
					   const float * __restrict__ const vsrc,
					   const int nsources,
					   const float _xt,
					   const float _yt,
					   const float h,
					   float * const xresult,
					   float * const yresult,
					   const int stride);

    void force_e2p(const realtype mass,
		   const realtype rx,
		   const realtype ry,
		   const realtype * __restrict__ const rxp,
		   const realtype * __restrict__ const ixp,
		   realtype * const xresult,
		   realtype * const yresult);

    void reference_force_e2p_tiled(const realtype mass,
				   const realtype rx,
				   const realtype ry,
				   const realtype h,
				   const realtype * __restrict__ const rxp,
				   const realtype * __restrict__ const ixp,
				   realtype * const xresult,
				   realtype * const yresult, const int stride);

    void force_e2p_tiled(const realtype mass,
			 const realtype rx,
			 const realtype ry,
			 const realtype h,
			 const realtype * __restrict__ const rxp,
			 const realtype * __restrict__ const ixp,
			 realtype * const xresult,
			 realtype * const yresult);
    
    void downward_e2l(const realtype * x0s,
		      const realtype * y0s,
		     const realtype * masses,
		     const realtype * __restrict__ * const vrexpansions,
		     const realtype * __restrict__ * const viexpansions,
		     const int nexpansions,
		      realtype * __restrict__ const rlocal,
		      realtype * __restrict__ const ilocal);

    void downward_l2p_tiled( const realtype rx,
				    const realtype ry,
			     const realtype h,
			     const realtype * __restrict__ const rlocal,
			     const realtype * __restrict__ const ilocal,
			     realtype * const xresult,
			     realtype * const yresult);

#ifdef __cplusplus
}
#endif

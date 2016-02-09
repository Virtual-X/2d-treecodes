/*
 *  potential-kernels.h
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
    realtype potential_p2p(const realtype * __restrict__ const xsources,
			   const realtype * __restrict__ const ysources,
			   const realtype * __restrict__ const sources,
			   const int nsources,
			   const realtype _xt,
			   const realtype _yt);

    realtype reference_potential_p2p(const realtype * __restrict__ const _xsrc,
  const realtype * __restrict__ const _ysrc,
  const realtype * __restrict__ const _vsrc,
  const int nsources,
  const realtype _xt,
				     const realtype _yt);
    
 
     realtype potential_e2p(
      const realtype * rzs,
      const realtype * izs,
      const realtype * masses,
      const realtype * const * rxps,
      const realtype * const * ixps,
      const int ndst);

      realtype reference_potential_e2p(const realtype mass,
			   const realtype rx,
			   const realtype ry,
			   const realtype * __restrict__ const rxp,
			   const realtype * __restrict__ const ixp);
  realtype treference_potential_e2p(
      const realtype * rzs,
      const realtype * izs,
      const realtype * masses,
      const realtype * const * rxps,
      const realtype * const * ixps,
      const int ndst)
  {
      realtype s = 0;
      for(int c = 0; c < ndst; ++c)
	  s += reference_potential_e2p(masses[c], rzs[c], izs[c], rxps[c], ixps[c]);

      return s;
  }
	
  
    
#ifdef __cplusplus
}
#endif

/*
 *  potential-kernels.m4, potential-kernels.ispc
 *  Part of 2d-treecodes
 *
 *  Created and authored by Diego Rossinelli on 2015-11-25.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

include(unroll.m4)

#define EPS (10 * __DBL_EPSILON__)
define(NACC, 4)
define(GSIZE, 4)

#include <cmath>

extern "C"  realtype potential_p2p(
    const realtype *  _xsrc,
    const realtype *  _ysrc,
    const realtype *  _vsrc,
    const int nsources,
    const realtype xt,
    const realtype yt)
   {
   const realtype eps = EPS;

   realtype LUNROLL(`i', 0, eval(NACC - 1), `
   ifelse(i,0,,`,') TMP(s,i) = 0') ;
    const int nnice = eval(4 * NACC) * (nsources / eval(4 * NACC));

      for( int i = 0; i < nnice; i += eval(4 * NACC))
      {
	const  realtype *  const xsrc = _xsrc + i;
      	const  realtype *  const ysrc = _ysrc + i;
      	const  realtype *  const vsrc = _vsrc + i;

	LUNROLL(j, 0, eval(NACC - 1), `
	for(int programIndex = 0; programIndex < 4; ++programIndex)
	{
	const realtype TMP(xr, j) = xt - xsrc[programIndex + eval(j * GSIZE)];
	const realtype TMP(yr, j) = yt - ysrc[programIndex + eval(j * GSIZE)];
	TMP(s, j) += log((float)(TMP(xr, j) * TMP(xr, j) + TMP(yr, j) * TMP(yr, j) + EPS)) * vsrc[programIndex + eval(j * GSIZE)];
	}
	')
    	}

	REDUCE(`+=', LUNROLL(`i', 0, eval(NACC - 1), `ifelse(i,0,,`,') TMP(s,i)'))

	for(int i = nnice; i < nsources; ++i)
    	{
	    const realtype xr = xt - _xsrc[i];
      	    const realtype yr = yt - _ysrc[i];

      	    s_0 += log(xr * xr + yr * yr + eps) * _vsrc[i];
    	}

	return s_0 * 0.5;
   }

extern "C"  realtype potential_e2p(
       const realtype rzs[],
       const realtype izs[],
       const realtype masses[],
       const realtype * const  rxps[],
       const realtype * const  ixps[],
       const int ndst)
  {
	realtype result = 0;

	for(int i = 0;  i < ndst ; ++i)
	{
	   const realtype rz = rzs[i];
	   const realtype iz = izs[i];
	   const realtype r2 = rz * rz + iz * iz;
	   const realtype rinvz_1 = rz / r2;
	   const realtype iinvz_1 = -iz / r2;

	   LUNROLL(j, 2, ORDER, `
     	   const realtype TMP(rinvz, j) = TMP(rinvz, eval(j - 1)) * rinvz_1 - TMP(iinvz, eval(j - 1)) * iinvz_1;
     	   const realtype TMP(iinvz, j) = TMP(rinvz, eval(j - 1)) * iinvz_1 + TMP(iinvz, eval(j - 1)) * rinvz_1;')

	   LUNROLL(j, 1, ORDER, `
	   realtype TMP(rsum, eval(j - 1)) = rxps[i][eval(j - 1)] * TMP(rinvz, j) - ixps[i][eval(j - 1)] * TMP(iinvz, j);')
 	   REDUCE(`+=', LUNROLL(i, 0, eval(ORDER - 1),`ifelse(i,0,,`,')TMP(rsum,i)'))

	   result += masses[i] * log(r2) / 2 + TMP(rsum, 0);
	 }

	 return result;
  }

/*
*  force-kernels.mc
*  Part of MRAG/2d-treecode-potential
*
*  Created and authored by Diego Rossinelli on 2015-11-25.
*  Copyright 2015. All rights reserved.
*
*  Users are NOT authorized
*  to employ the present software for their own publications
*  before getting a written permission from the author of this file.
*/

define(NACC, 4)
include(unroll.m4)

__device__ void force_p2p(const realtype * __restrict__ const xsources,
			  const realtype * __restrict__ const ysources,
			  const realtype * __restrict__ const vsources,
			  const int nsources,
			  const realtype xt,
			  const realtype yt,
			  realtype& xforce,
			  realtype& yforce)
    {		    
	const double eps = 10 * __DBL_EPSILON__;

	realtype xsum = 0, ysum = 0;
	for(int i = 0; i < nsources; ++i)
	{
	    const realtype xr = xt - xsources[i];
	    const realtype yr = yt - ysources[i];
	    const realtype factor =  vsources[i] / (xr * xr + yr * yr + eps);
	    
	    xsum += xr * factor;
	    ysum += yr * factor;
	}
	
	xforce += xsum;
	yforce += ysum;
    }
    

__device__ void force_e2p(const realtype mass,
	   			const realtype * __restrict__ const rxp,
			    	const realtype * __restrict__ const ixp,
			    	const realtype rz,
			    	const realtype iz,
			    	realtype& xforce,
			    	realtype& yforce)
    {
	const realtype r2 = rz * rz + iz * iz;

	const realtype rinvz_1 = rz / r2;
	const realtype iinvz_1 = -iz / r2;
	
	realtype rsum = mass * rinvz_1, isum = mass * iinvz_1;
	realtype rprod = rinvz_1, iprod = iinvz_1;

	for(int j = 0; j < ORDER; ++j)
	{
	    const realtype rtmp = rprod * rinvz_1 - iprod * iinvz_1;
	    const realtype itmp = rprod * iinvz_1 + iprod * rinvz_1;
	    
	    rprod = rtmp;
	    iprod = itmp;	
	    
	    rsum -= (j + 1) * (rxp[j] * rprod - ixp[j] * iprod);
	    isum -= (j + 1) * (rxp[j] * iprod + ixp[j] * rprod);
	}
	    
	xforce += rsum;
	yforce -= isum;
    }

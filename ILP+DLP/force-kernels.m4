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
divert(-1)
include(unroll.m4)
define(mysign, `ifelse(eval((-1)**($1)), -1,-,+)')
divert(0)
dnl

#define ACCESS(x) __ldg(&(x)) 

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
	    const realtype xr = xt - ACCESS(xsources[i]);
	    const realtype yr = yt - ACCESS(ysources[i]);
	    const realtype factor =  ACCESS(vsources[i]) / (xr * xr + yr * yr + eps);
	    
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
	    
	    rsum -= (j + 1) * (ACCESS(rxp[j]) * rprod - ACCESS(ixp[j]) * iprod);
	    isum -= (j + 1) * (ACCESS(rxp[j]) * iprod + ACCESS(ixp[j]) * rprod);
	}
	    
	xforce += rsum;
	yforce -= isum;
    }


__device__ void force_downward_e2l(
     const realtype x0,
     const realtype y0,
     const realtype mass,
     const realtype * const rxp,
     const realtype * const ixp,
     realtype * const rlocal,
     realtype * const ilocal)
  {
	const int tid = threadIdx.x;

	const realtype r2z0 = x0 * x0 + y0 * y0;
    	const realtype rinvz_1 = x0 / r2z0;
    	const realtype iinvz_1 = -y0 / r2z0;

	dnl
    	LUNROLL(j, 1, eval(ORDER),`
    	ifelse(j, 1, , `
      	  const realtype TMP(rinvz, j) = TMP(rinvz, eval(j - 1)) * rinvz_1 - TMP(iinvz, eval(j - 1)) * iinvz_1;
      	  const realtype TMP(iinvz, j) = TMP(rinvz, eval(j - 1)) * iinvz_1 + TMP(iinvz, eval(j - 1)) * rinvz_1;')

	  const realtype TMP(rcoeff, j) = ACCESS(rxp[eval(j - 1)]) * TMP(rinvz, j) - ACCESS(ixp[eval(j - 1)]) * TMP(iinvz, j);
      	  const realtype TMP(icoeff, j) = ACCESS(rxp[eval(j - 1)]) * TMP(iinvz, j) + ACCESS(ixp[eval(j - 1)]) * TMP(rinvz, j);
      	')

	/*not needed as long as we evaluate grad(pot)
	{
		const realtype rlogmz0 = log(r2z0) * 0.5;
    		const realtype ilogmz0 = atan2(y0, x0) - M_PI;

		realtype rpartial = mass * rlogmz0 LUNROLL(k, 1, eval(ORDER),` mysign(k) TMP(rcoeff, k)  ');
       		realtype ipartial = mass * ilogmz0 LUNROLL(k, 1, eval(ORDER),` mysign(k) TMP(icoeff, k)  ');
		
		WARPSUM(rpartial, ipartial)

		if (tid == 0)
		{
			rlocal[0] += tmp0;
			ilocal[0] += tmp1;
		}
	}
	*/ 

	LUNROLL(l, 1, eval(ORDER),`
      	{
		const realtype TMP(prefac, l) = ifelse(l,1,
		` - mass', 
		`mass * esyscmd(echo -1/eval(l) | bc --mathlib )');

		pushdef(`BINFAC', `BINOMIAL(eval(l + k - 1), eval(k - 1)).f')dnl
		const realtype TMP(rtmp, l) = TMP(prefac, l) LUNROLL(k, 1, eval(ORDER),`
       		mysign(k) ifelse(BINFAC,1.f,,`BINFAC *') TMP(rcoeff, k)');

       		const realtype TMP(itmp, l) =  LUNROLL(k, 1, eval(ORDER),`
       		mysign(k) ifelse(BINFAC,1.f,,`BINFAC *') TMP(icoeff, k)');
		popdef(`BINFAC')dnl
        		
       		realtype rpartial = TMP(rtmp, l) * TMP(rinvz, l) - TMP(itmp, l) * TMP(iinvz, l);
       		realtype ipartial = TMP(rtmp, l) * TMP(iinvz, l) + TMP(itmp, l) * TMP(rinvz, l);

		WARPSUM(rpartial, ipartial)

		if (tid == 0)
		{
			rlocal[l] += rpartial;
       			ilocal[l] += ipartial;
		}
	}')
}

__device__ void force_downward_l2p(
     const realtype rz_1,
     const realtype iz_1,
     const realtype * const rlocal,
     const realtype * const ilocal,
     realtype& xresult,
     realtype& yresult)
      {
          const realtype TMP(rresult, 1) = rlocal[1];
          const realtype TMP(iresult, 1) = ilocal[1];

          LUNROLL(l, 2, eval(ORDER),`
          const realtype TMP(rz, l) = TMP(rz, eval(l - 1)) * rz_1 - TMP(iz, eval(l - 1)) * iz_1;
          const realtype TMP(iz, l) = TMP(rz, eval(l - 1)) * iz_1 + TMP(iz, eval(l - 1)) * rz_1;

          const realtype TMP(rresult, l) = TMP(rresult, eval(l - 1)) +
          l * (rlocal[l] * TMP(rz, eval(l - 1)) - ilocal[l] * TMP(iz, eval(l - 1)));

          const realtype TMP(iresult, l) = TMP(iresult, eval(l - 1)) +
          l * (rlocal[l] * TMP(iz, eval(l - 1)) + ilocal[l] * TMP(rz, eval(l - 1)));
          ')

	  xresult += TMP(rresult, ORDER);
	  yresult -= TMP(iresult, ORDER);
      }

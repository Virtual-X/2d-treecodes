/*
 *  force-kernels.m4, force-kernels.ispc
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

export void force_p2p_8x8(
			 uniform const realtype xsources[],
			 uniform const realtype ysources[],
			 uniform const realtype vsources[],
			 uniform const int nsources,
			 uniform const realtype x0,
			 uniform const realtype y0,
			 uniform const realtype h,
			 uniform realtype xresult[],
			 uniform realtype yresult[])
{
	const double eps = 10 * __DBL_EPSILON__;

	foreach(d = 0 ... 64)
	{
		const int ix = d & 7;
		const int iy = d >> 3;

		const realtype xt = x0 + ix * h;
		const realtype yt = y0 + iy * h;

		realtype xsum = 0, ysum = 0;
		for(int s = 0; s < nsources; ++s)
		{
			const realtype xr = xt - xsources[s];
	    		const realtype yr = yt - ysources[s];
	    		const realtype factor = vsources[s] / (xr * xr + yr * yr + eps);

			xsum += xr * factor;
	    		ysum += yr * factor;
		}

		xresult[d] += xsum;
		yresult[d] += ysum;
	}
}

export void force_e2p_8x8(
			uniform const realtype mass,
			uniform const realtype x0,
			uniform const realtype y0,
			uniform const realtype h,
			uniform const realtype rxp[],
			uniform const realtype ixp[],
			uniform realtype xresult[],
			uniform realtype yresult[])
{
	foreach(d = 0 ... 64)
	{
		const int ix = d & 7;
		const int iy = d >> 3;

		const realtype rz = x0 + ix * h;
		const realtype iz = y0 + iy * h;

		const realtype r2 = rz * rz + iz * iz;

		const realtype rinvz_1 = rz / r2;
		const realtype iinvz_1 = -iz / r2;

		realtype rsum = mass * rinvz_1, isum = mass * iinvz_1;
		realtype rprod = rinvz_1, iprod = iinvz_1;

		LUNROLL(j, 0, eval(ORDER - 1),`
		{
			const realtype rtmp = rprod * rinvz_1 - iprod * iinvz_1;
	    		const realtype itmp = rprod * iinvz_1 + iprod * rinvz_1;

			rprod = rtmp;
	    		iprod = itmp;

			rsum -= (j + 1) * (rxp[j] * rprod - ixp[j] * iprod);
	    		isum -= (j + 1) * (rxp[j] * iprod + ixp[j] * rprod);
		}')

	    	xresult[d] += rsum;
		yresult[d] -= isum;
	}
}

define(mysign, `ifelse(eval((-1)**($1)), -1,-,+)')

export void downward_e2l(
	uniform const realtype x0s[],
	uniform const realtype y0s[],
	uniform const realtype masses[],
	uniform const realtype * uniform rxps[],
	uniform const realtype * uniform ixps[],
	uniform const int nexpansions,
	uniform realtype rlocal[],
	uniform realtype ilocal[])
{
	foreach(i = 0 ... nexpansions)
	{
		const realtype mass = masses[i];

		const realtype x0 = x0s[i];
		const realtype y0 = y0s[i];

		const realtype r2z0 = x0 * x0 + y0 * y0;
    		const realtype rinvz_1 = x0 / r2z0;
    		const realtype iinvz_1 = -y0 / r2z0;

		uniform const realtype * const rxp = rxps[i];
		uniform const realtype * const ixp = ixps[i];

		dnl
    		LUNROLL(j, 1, eval(ORDER),`
    		ifelse(j, 1, , `
      	  		  const realtype TMP(rinvz, j) = TMP(rinvz, eval(j - 1)) * rinvz_1 - TMP(iinvz, eval(j - 1)) * iinvz_1;
      	  		  const realtype TMP(iinvz, j) = TMP(rinvz, eval(j - 1)) * iinvz_1 + TMP(iinvz, eval(j - 1)) * rinvz_1;')

	  		  const realtype TMP(rcoeff, j) = rxp[eval(j - 1)] * TMP(rinvz, j) - ixp[eval(j - 1)] * TMP(iinvz, j);
      	  		  const realtype TMP(icoeff, j) = rxp[eval(j - 1)] * TMP(iinvz, j) + ixp[eval(j - 1)] * TMP(rinvz, j);
      		')

		LUNROLL(l, 1, eval(ORDER),`
      		{
			realtype TMP(rtmp, l) = ifelse(l,1,` - mass', `mass * esyscmd(echo -1/eval(l) | bc --mathlib )');
			realtype TMP(itmp, l) = 0;

			pushdef(`BINFAC', `BINOMIAL(eval(l + k - 1), eval(k - 1)).')dnl
			 LUNROLL(k, 1, eval(ORDER),`
			 TMP(rtmp, l) mysign(k)= ifelse(BINFAC,1.f,,`BINFAC *') TMP(rcoeff, k);
			 TMP(itmp, l) mysign(k)= ifelse(BINFAC,1.f,,`BINFAC *') TMP(icoeff, k);
			 ')
			popdef(`BINFAC')dnl

       			realtype rpartial = TMP(rtmp, l) * TMP(rinvz, l) - TMP(itmp, l) * TMP(iinvz, l);
       			realtype ipartial = TMP(rtmp, l) * TMP(iinvz, l) + TMP(itmp, l) * TMP(rinvz, l);

			rlocal[l] += reduce_add(rpartial);
			ilocal[l] += reduce_add(ipartial);
		}')
	}
}

export void downward_l2p_8x8(
       uniform const realtype x0,
       uniform const realtype y0,
       uniform const realtype h,
       uniform const realtype rlocal[],
       uniform const realtype ilocal[],
       uniform realtype xresult[],
       uniform realtype yresult[])
{
	foreach(d = 0 ... 64)
	{
		const int ix = d & 7;
		const int iy = d >> 3;

		const realtype rz_1 = x0 + ix * h;
		const realtype iz_1 = y0 + iy * h;

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

		xresult[d] += TMP(rresult, ORDER);
		yresult[d] -= TMP(iresult, ORDER);
	}
}

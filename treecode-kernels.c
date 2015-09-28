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

#include <math.h>
#include <immintrin.h>
#include "treecode.h"

#if ORDER > 20 || ORDER < 1
#error MAX ORDER supported is 20
#endif

#define N2SIZE 64
#define EPS (10 * __DBL_EPSILON__)
#define MAX(a,b) (((a)>(b))?(a):(b))

realtype treecode_p2p(const realtype * __restrict__ const _xsrc,
		      const realtype * __restrict__ const _ysrc,
		      const realtype * __restrict__ const _vsrc,
		      const int nsources,
		      const realtype xt,
		      const realtype yt)
{
    enum { NACC = 32 };

    realtype s[NACC];
    for(int i = 0; i < NACC; ++i)
	s[i] = 0;

    const int nnice = NACC * (nsources / NACC);

    for(int i = 0; i < nnice; i += NACC)
    {
	const realtype * const xsrc = _xsrc + i;
	const realtype * const ysrc = _ysrc + i;
	const realtype * const vsrc = _vsrc + i;

#pragma GCC ivdep
	for(int j = 0; j < NACC; ++j)
	{
	    const realtype xr = xt - xsrc[j];
	    const realtype yr = yt - ysrc[j];

	    s[j] += log(xr * xr + yr * yr + EPS) * vsrc[j];
	}
    }

    realtype sum = 0;

    for(int i = nnice; i < nsources; ++i)
    {
	const realtype xr = xt - _xsrc[i];
	const realtype yr = yt - _ysrc[i];

	sum += log(xr * xr + yr * yr + EPS) * _vsrc[i];
    }

    for(int i = 0; i < NACC; ++i)
	sum += s[i];

    return sum / 2;
}

realtype __attribute__((pure)) treecode_e2p(const realtype mass,
					    const realtype rz,
					    const realtype iz,
					    const realtype * __restrict__ const rxp,
					    const realtype * __restrict__ const ixp)
{
    const realtype r2 = rz * rz + iz * iz;

    const realtype rinvz = rz / r2;
    const realtype iinvz = -iz / r2;

    realtype rprod = rinvz;
    realtype iprod = iinvz;
    realtype rs = mass * log(r2) / 2;

    rs += rprod * rxp[0] - iprod * ixp[0];

    realtype rprods[ORDER], iprods[ORDER];
    for(int n = 1; n < ORDER; ++n)
    {
	const realtype rnewprod = rinvz * rprod - iinvz * iprod;
	const realtype inewprod = iinvz * rprod + rinvz * iprod;

	rprod = rnewprod;
	iprod = inewprod;

	rprods[n] = rprod;
	iprods[n] = iprod;
    }

    for(int n = 1; n < ORDER; ++n)
    	rs += rprods[n] * rxp[n] - iprods[n] * ixp[n];

    return rs;
}

void treecode_p2e(const realtype * __restrict__ const xsources,
		  const realtype * __restrict__ const ysources,
		  const realtype * __restrict__ const sources,
		  const int nsources,
		  const realtype x0,
		  const realtype y0,
		  const realtype h,
		  realtype * const mass,
		  realtype * const weight,
		  realtype * const xsum,
		  realtype * const ysum,
		  realtype * const radius,
		  realtype * __restrict__ const rexpansions,
		  realtype * __restrict__ const iexpansions)
{
    realtype m = 0, w = 0, wx = 0, wy = 0;

    for(int i = 0; i < nsources; ++i)
    {
	const realtype sv = sources[i];
	const realtype av = fabs(sources[i]);

	m += sv;
	w += av;
	wx += xsources[i] * av;
	wy += ysources[i] * av;
    }

    *mass = m;
    *weight = w;
    *xsum = wx;
    *ysum = wy;

    if (w == 0)
    {
	*weight = 1;
	*xsum = x0 + 0.5 * h;
	*ysum = y0 + 0.5 * h;
    }

    const realtype xcom = wx / w;
    const realtype ycom = wy / w;

    realtype r2 = 0;

    for(int i = 0; i < nsources; ++i)
    {
	const realtype xr = xsources[i] - xcom;
	const realtype yr = ysources[i] - ycom;

	r2 = MAX(r2, xr * xr + yr * yr);
    }

    *radius = sqrt(r2);

    for (int n = 0; n < ORDER; ++n)
	rexpansions[n] = 0;

    for (int n = 0; n < ORDER; ++n)
	iexpansions[n] = 0;

    for(int i = 0; i < nsources; ++i)
    {
	const realtype rrp = xsources[i] - xcom;
	const realtype irp = ysources[i] - ycom;

	realtype rprod = rrp;
	realtype iprod = irp;

	const realtype term = sources[i];

	rexpansions[0] -= rprod * term;
	iexpansions[0] -= iprod * term;

#pragma GCC ivdep
	for (int n = 1; n < ORDER; ++n)
	{
	    const realtype rnewprod = rprod * rrp - iprod * irp;
	    const realtype inewprod = rprod * irp + iprod * rrp;

	    rprod = rnewprod;
	    iprod = inewprod;

	    const realtype term = sources[i] / (n + 1);

	    rexpansions[n] -= rprod * term;
	    iexpansions[n] -= iprod * term;
	}
    }
}

extern const realtype binomlut[20][20];

void treecode_e2e(const V4 srcmass, const V4 rx, const V4 ry,
		  const V4 * __restrict__ const rsrcxp,
		  const V4 * __restrict__ const isrcxp,
		  realtype * __restrict__ const rdstxp,
		  realtype * __restrict__ const idstxp)
{
    V4 zero = {0, 0, 0, 0};

    V4 rresult[ORDER];
    
#pragma ivdep
    for (int i = 0; i < ORDER; ++i)
	rresult[i] = zero;

    V4 iresult[ORDER];
    
#pragma ivdep
    for (int i = 0; i < ORDER; ++i)
	iresult[i] = zero;

    for (int j = 0; j < ORDER; ++j)
    {
	V4 rsum = {0, 0, 0, 0}, isum = {0, 0, 0, 0}, rprod = {1, 1, 1, 1}, iprod = {0, 0, 0, 0};

	for (int k = j; k >= 0; --k)
	{
	    const realtype bterm = binomlut[j][k];

	    rsum += bterm * (rsrcxp[k] * rprod - isrcxp[k] * iprod);
	    isum += bterm * (isrcxp[k] * rprod + rsrcxp[k] * iprod);

	    const V4 rnewprod = rprod * rx - iprod * ry;
	    const V4 inewprod = rprod * ry + iprod * rx;

	    rprod = rnewprod;
	    iprod = inewprod;
	}

	const V4 term = srcmass / (j + 1);

	rsum -= rprod * term;
	isum -= iprod * term;

	rresult[j] += rsum;
	iresult[j] += isum;
    }

#pragma GCC ivdep
    for(int i = 0; i < ORDER; ++i)
    {
	const V4 re = rresult[i];
	rdstxp[i] = re[0] + re[1] + re[2] + re[3];
    }

#pragma GCC ivdep
    for(int i = 0; i < ORDER; ++i)
    {
	const V4 im = iresult[i];
	idstxp[i] = im[0] + im[1] + im[2] + im[3];
    }
}

const realtype binomlut[20][20] = { 1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,3,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,4,6,4,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,5,10,10,5,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,6,15,20,15,6,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,7,21,35,35,21,7,1,0,0,0,0,0,0,0,0,0,0,0,0,1,8,28,56,70,56,28,8,1,0,0,0,0,0,0,0,0,0,0,0,1,9,36,84,126,126,84,36,9,1,0,0,0,0,0,0,0,0,0,0,1,10,45,120,210,252,210,120,45,10,1,0,0,0,0,0,0,0,0,0,1,11,55,165,330,462,462,330,165,55,11,1,0,0,0,0,0,0,0,0,1,12,66,220,495,792,924,792,495,220,66,12,1,0,0,0,0,0,0,0,1,13,78,286,715,1287,1716,1716,1287,715,286,78,13,1,0,0,0,0,0,0,1,14,91,364,1001,2002,3003,3432,3003,2002,1001,364,91,14,1,0,0,0,0,0,1,15,105,455,1365,3003,5005,6435,6435,5005,3003,1365,455,105,15,1,0,0,0,0,1,16,120,560,1820,4368,8008,11440,12870,11440,8008,4368,1820,560,120,16,1,0,0,0,1,17,136,680,2380,6188,12376,19448,24310,24310,19448,12376,6188,2380,680,136,17,1,0,0,1,18,153,816,3060,8568,18564,31824,43758,48620,43758,31824,18564,8568,3060,816,153,18,1,0,1,19,171,969,3876,11628,27132,50388,75582,92378,92378,75582,50388,27132,11628,3876,969,171,19,1};


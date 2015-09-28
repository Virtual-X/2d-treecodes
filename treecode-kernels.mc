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
divert(-1)
define(`forloop',
       `pushdef(`$1', `$2')_forloop(`$1', `$2', `$3', `$4')popdef(`$1')')
define(`_forloop',
       `$4`'ifelse($1, `$3', ,
		   `define(`$1', incr($1))_forloop(`$1', `$2', `$3', `$4')')')

define(`forrloop',
       `pushdef(`$1', `$2')_forrloop(`$1', `$2', `$3', `$4')popdef(`$1')')
define(`_forrloop',
       `$4`'ifelse($1, `$3', ,
		   `define(`$1', decr($1))_forrloop(`$1', `$2', `$3', `$4')')')

define(BINOMIAL, `syscmd(python binomial.py $1 $2)')


#1 iteration variable
#2 iteration start
#3 iteration end
#4 body

define(LUNROLL, `forloop($1, $2, $3,`$4')')
define(RLUNROLL, `forrloop($1, $2, $3, `$4')')
divert(0)
#include <math.h>
#include <immintrin.h>
#include "treecode.h"

#if ORDER > 20 || ORDER < 1
#error MAX ORDER supported is 20
#endif

#define N2SIZE 64
#define EPS (10 * __DBL_EPSILON__)
#define MAX(a,b) (((a)>(b))?(a):(b))

define(NACC, 32)
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
    realtype s[NACC];
    for(int i = 0; i < NACC; ++i)
	s[i] = 0;

    const int nnice = NACC * (nsources / NACC);

    for(int i = 0; i < nnice; i += NACC)
    {
	const realtype * const xsrc = _xsrc + i;
	const realtype * const ysrc = _ysrc + i;
	const realtype * const vsrc = _vsrc + i;

	LUNROLL(j, 0, eval(NACC - 1), `
	{
	    const realtype xr = xt - xsrc[j];
	    const realtype yr = yt - ysrc[j];

	    s[j] += log(xr * xr + yr * yr + EPS) * vsrc[j];
	}')
    }

    realtype sum = 0;

    for(int i = nnice; i < nsources; ++i)
    {
	const realtype xr = xt - _xsrc[i];
	const realtype yr = yt - _ysrc[i];

	sum += log(xr * xr + yr * yr + EPS) * _vsrc[i];
    }

    LUNROLL(i, 0, eval(NACC - 1), `
	sum += s[i];')

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

    LUNROLL(n, 1, eval(ORDER - 1), `
    {
	const realtype rnewprod = rinvz * rprod - iinvz * iprod;
	const realtype inewprod = iinvz * rprod + rinvz * iprod;

	rprod = rnewprod;
	iprod = inewprod;

	rprods[n] = rprod;
	iprods[n] = iprod;
    }')

    LUNROLL(n, 1, eval(ORDER - 1),`
    	rs += rprods[n] * rxp[n] - iprods[n] * ixp[n];');

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

    const V4 zero = {0, 0, 0, 0};

    V4 rxp[ORDER], ixp[ORDER];

    {
	LUNROLL(n, 0, eval(ORDER - 1),`
	    rxp[n] = zero;')
	
	LUNROLL(n, 0, eval(ORDER - 1),`
	    ixp[n] = zero;')
    }

    for(int i = 0; i < nsources; i += 4)
    {
	V4 rrp = zero, irp = zero, srcs = zero;

	for(int c = 0; c < 4; ++c)
	    if (i + c < nsources)
		rrp[c] = xsources[i + c];

	for(int c = 0; c < 4; ++c)
	    if (i + c < nsources)
		irp[c] = ysources[i + c];
	
	for(int c = 0; c < 4; ++c)
	    if (i + c < nsources)
		srcs[c] = sources[i + c];
	
	rrp -= xcom;
	irp -= ycom;
	
	V4 rprod = rrp, iprod = irp;

	rxp[0] -= rprod * srcs;
	ixp[0] -= iprod * srcs;

	LUNROLL(n, 1, eval(ORDER - 1),`
	{
	    const V4 rnewprod = rprod * rrp - iprod * irp;
	    const V4 inewprod = rprod * irp + iprod * rrp;

	    rprod = rnewprod;
	    iprod = inewprod;

	    const V4 term = srcs / (realtype)(n + 1);

	    rxp[n] -= rprod * term;
	    ixp[n] -= iprod * term;
	}')
    }

    LUNROLL(n, 0, eval(ORDER - 1),`
	rexpansions[n] = rxp[n][0] + rxp[n][1] + rxp[n][2] + rxp[n][3];')

    LUNROLL(n, 0, eval(ORDER - 1),`
	iexpansions[n] = ixp[n][0] + ixp[n][1] + ixp[n][2] + ixp[n][3];')
}

void treecode_e2e(const V4 srcmass, const V4 rx, const V4 ry,
		  const V4 * __restrict__ const rsrcxp,
		  const V4 * __restrict__ const isrcxp,
		  realtype * __restrict__ const rdstxp,
		  realtype * __restrict__ const idstxp)
{
    const V4 zero = {0, 0, 0, 0};
    const V4 one = {1, 1, 1, 1};

    V4 rresult[ORDER];
    
    LUNROLL(i, 0, eval(ORDER - 1),`
	rresult[i] = zero;')

    V4 iresult[ORDER];
    
    LUNROLL(i, 0, eval(ORDER - 1),`
	iresult[i] = zero;')

    LUNROLL(j, 0, eval(ORDER - 1),`
    {
	V4 rsum = zero, isum = zero, rprod = one, iprod = zero;

	RLUNROLL(k, j, 0, `
	{
	    const realtype bterm = BINOMIAL(j, k);
	    
	    rsum += bterm * (rsrcxp[k] * rprod - isrcxp[k] * iprod);
	    isum += bterm * (isrcxp[k] * rprod + rsrcxp[k] * iprod);

	    const V4 rnewprod = rprod * rx - iprod * ry;
	    const V4 inewprod = rprod * ry + iprod * rx;

	    rprod = rnewprod;
	    iprod = inewprod;
	}')

	const V4 term = srcmass / (j + 1);

	rsum -= rprod * term;
	isum -= iprod * term;

	rresult[j] += rsum;
	iresult[j] += isum;
    }')

    LUNROLL(i, 0, eval(ORDER - 1), `
    {
	const V4 re = rresult[i];
	rdstxp[i] = re[0] + re[1] + re[2] + re[3];
    }')

    LUNROLL(i, 0, eval(ORDER - 1), `
    {
	const V4 im = iresult[i];
	idstxp[i] = im[0] + im[1] + im[2] + im[3];
    }')
}

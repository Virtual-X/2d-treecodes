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

#USAGE LUNROLL
#$1 iteration variable
#$2 iteration start
#$3 iteration end
#$4 body

define(LUNROLL, `forloop($1, $2, $3,`$4')')
define(RLUNROLL, `forrloop($1, $2, $3, `$4')')
define(NACC, 32)
define(`TMP', $1_$2)
divert(0)

#define EPS (10 * __DBL_EPSILON__)
#define MAX(a,b) (((a)>(b))?(a):(b))

void force_p2p(const realtype * __restrict__ const _xsrc,
		      const realtype * __restrict__ const _ysrc,
		      const realtype * __restrict__ const _vsrc,
		      const int nsources,
		      const realtype xt,
		      const realtype yt,
		      realtype * const xresult,
		      realtype * const yresult)
{
    realtype dummy LUNROLL(`i', 0, NACC, `, TMP(xs,i) = 0, TMP(ys,i) = 0') ;

    const int nnice = NACC * (nsources / NACC);

    for(int i = 0; i < nnice; i += NACC)
    {
	const realtype * __restrict__ const xsrc = _xsrc + i;
	const realtype * __restrict__ const ysrc = _ysrc + i;
	const realtype * __restrict__ const vsrc = _vsrc + i;

	LUNROLL(j, 0, eval(NACC - 1), `
	const realtype TMP(xr, j) = xt - xsrc[j];
	const realtype TMP(yr, j) = yt - ysrc[j];
	const realtype TMP(factor, j) = vsrc[j] * (((realtype)1) / (TMP(xr, j) * TMP(xr, j) + TMP(yr, j) * TMP(yr, j) + EPS)) ;')
	
	LUNROLL(j, 0, eval(NACC - 1), `
	TMP(xs, j) += TMP(xr, j) * TMP(factor, j);
	TMP(ys, j) += TMP(yr, j) * TMP(factor, j);
	')
    }

    realtype xsum = 0, ysum = 0;

    for(int i = nnice; i < nsources; ++i)
    {
	const realtype xr = xt - _xsrc[i];
	const realtype yr = yt - _ysrc[i];
	const realtype factor = _vsrc[i] / (xr * xr + yr * yr + EPS);

	xsum += xr * factor;
	ysum += yr * factor;
    }

    LUNROLL(i, 0, eval(NACC - 1), `
    xsum += TMP(xs, i);
    ysum += TMP(ys, i);')

    *xresult = xsum;
    *yresult = ysum;
}

void force_e2p(const realtype mass,
     			   const realtype rz,
			    const realtype iz,
			    const realtype * __restrict__ const rxp,
			    const realtype * __restrict__ const ixp,
			    realtype * const xresult,
			    realtype * const yresult)
{
    const realtype r2 = rz * rz + iz * iz;

    const realtype rinvz = rz / r2;
    const realtype iinvz = -iz / r2;

    realtype xs = mass * rinvz;
    realtype ys = mass * iinvz;

    const realtype rprod_0 = rinvz * rinvz - iinvz * iinvz;
    const realtype iprod_0 = 2 * rinvz * iinvz;
    
    xs -= rprod_0 * rxp[0] - iprod_0 * ixp[0];
    ys -= rprod_0 * ixp[0] + iprod_0 * rxp[0];
    
    LUNROLL(n, 1, eval(ORDER - 1), `
    	const realtype TMP(rprod, n) = rinvz * TMP(rprod, eval(n - 1)) - iinvz * TMP(iprod, eval(n - 1));
	const realtype TMP(iprod, n) = iinvz * TMP(rprod, eval(n - 1)) + rinvz * TMP(iprod, eval(n - 1));

	xs -= (n + 1) * (TMP(rprod, n) * rxp[n] - TMP(iprod, n) * ixp[n]);
	ys -= (n + 1) * (TMP(rprod, n) * ixp[n] + TMP(iprod, n) * rxp[n]);
    ')

    *xresult = xs;
    *yresult = -ys;
}


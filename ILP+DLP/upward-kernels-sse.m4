/*
*  upward-kernels.mc
*  Part of MRAG/2d-treecode-potential
*
*  Created and authored by Diego Rossinelli on 2015-11-25.
*  Copyright 2015. All rights reserved.
*
*  Users are NOT authorized
*  to employ the present software for their own publications
*  before getting a written permission from the author of this file.
*/

include(unroll.m4)

#include <pmmintrin.h>
#include <math.h>

#define EPS (10 * __DBL_EPSILON__)
#define MAX(a,b) (((a)>(b))?(a):(b))

define(P2E_KERNEL, upward_p2e_order$1)
define(E2E_KERNEL, upward_e2e_order$1)

void P2E_KERNEL(ORDER)(const realtype * __restrict__ const xsources,
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

    const __m128d xcom2 = _mm_set1_pd(xcom);
    const __m128d ycom2 = _mm_set1_pd(ycom);

    const int nnice8 = 8 * (nsources / 8);

    __m128d LUNROLL(c, 0, 3, `ifelse(c,0,,`,') 
    TMP(r22,c) = _mm_setzero_pd()');

    for(int i = 0; i < nnice8; i += 8)
    {
      LUNROLL(c, 0, 3, `
      const __m128d TMP(xr, c) = _mm_loadu_pd(xsources + i + eval(c * 2)) - xcom2;')
      LUNROLL(c, 0, 3, `
      const __m128d TMP(yr, c) = _mm_loadu_pd(ysources + i + eval(c * 2)) - ycom2;')
      LUNROLL(c, 0, 3, `
      TMP(r22, c) = _mm_max_pd(TMP(r22, c),  TMP(xr, c) * TMP(xr, c) + TMP(yr, c) * TMP(yr, c));')
    }

    LUNROLL(c, 1, 3, `
    TMP(r22,0) = _mm_max_pd(TMP(r22, 0), TMP(r22, c));');
    realtype r22a[2];
    _mm_storeu_pd(r22a, TMP(r22, 0));

    realtype r2 = MAX(r22a[0], r22a[1]);
    
    for(int i = nnice8; i < nsources; ++i)
    {
      const realtype xr = xsources[i] - xcom;
      const realtype yr = ysources[i] - ycom;

      r2 = MAX(r2, xr * xr + yr * yr);
    }

    *radius = sqrt(r2);
    
    __m128d LUNROLL(n, 0, eval(ORDER - 1),`ifelse(n,0,,`,') 
    TMP(rxp, n) = _mm_setzero_pd()') 
    LUNROLL(n, 0, eval(ORDER - 1),`, 
    TMP(ixp, n) = _mm_setzero_pd()');

    const int nnice = 2 * (nsources / 2);
    for(int i = 0; i < nnice; i += 2)
    {
      const __m128d rrp = _mm_loadu_pd(xsources + i) - xcom2;
      const __m128d irp = _mm_loadu_pd(ysources + i) - ycom2;
      const __m128d srcs = _mm_loadu_pd(sources + i);

      __m128d rprod = rrp, iprod = irp;

      TMP(rxp, 0) -= rprod * srcs;
      TMP(ixp, 0) -= iprod * srcs;

      LUNROLL(n, 1, eval(ORDER - 1),`
      const __m128d TMP(rnewprod, n) = rprod * rrp - iprod * irp;
      const __m128d TMP(inewprod, n) = rprod * irp + iprod * rrp;

      rprod = TMP(rnewprod, n);
      iprod = TMP(inewprod, n);

      const __m128d TMP(term, n) = srcs * _mm_set1_pd(esyscmd(echo 1. / eval(n+1) | bc -l));

      TMP(rxp, n) -= rprod * TMP(term, n);
      TMP(ixp, n) -= iprod * TMP(term, n);')
    }

    for(int i = nnice; i < nsources; ++i)
    {
      const __m128d rrp = _mm_set_pd(RLUNROLL(c, 1, 0, `(i + c < nsources) ? xsources[i + c] : 0 ifelse(c,0,,`,')')) - xcom2;
      const __m128d irp = _mm_set_pd(RLUNROLL(c, 1, 0, `(i + c < nsources) ? ysources[i + c] : 0 ifelse(c,0,,`,')')) - ycom2;
      const __m128d srcs = _mm_set_pd(RLUNROLL(c, 1, 0, `(i + c < nsources) ? sources[i + c] : 0 ifelse(c,0,,`,')'));

      __m128d rprod = rrp, iprod = irp;

      TMP(rxp, 0) -= rprod * srcs;
      TMP(ixp, 0) -= iprod * srcs;

      LUNROLL(n, 1, eval(ORDER - 1),`
      const __m128d TMP(rnewprod, n) = rprod * rrp - iprod * irp;
      const __m128d TMP(inewprod, n) = rprod * irp + iprod * rrp;

      rprod = TMP(rnewprod, n);
      iprod = TMP(inewprod, n);

      const __m128d TMP(term, n) = srcs / _mm_set1_pd(eval(n+1).f);

      TMP(rxp, n) -= rprod * TMP(term, n); 
      TMP(ixp, n) -= iprod * TMP(term, n);')
    }

    LUNROLL(n, 0, eval(ORDER - 1),`
    _mm_store_sd(rexpansions + n, _mm_hadd_pd(TMP(rxp, n), TMP(rxp, n)));')
    LUNROLL(n, 0, eval(ORDER - 1),`
    _mm_store_sd(iexpansions + n, _mm_hadd_pd(TMP(ixp, n), TMP(ixp, n)));')
  }

typedef realtype V4 __attribute__ ((vector_size (sizeof(realtype) * 4)));

  void E2E_KERNEL(ORDER)(const V4 srcmass, const V4 rx, const V4 ry,
    const V4 * __restrict__ const rsrcxp,
    const V4 * __restrict__ const isrcxp,
    realtype * __restrict__ const rdstxp,
    realtype * __restrict__ const idstxp)
    {
      const V4 zero = {0, 0, 0, 0};
      const V4 one = {1, 1, 1, 1};

      V4 rresult[ORDER];

      V4 dummy LUNROLL(i, 0, eval(ORDER - 1),`, TMP(rresult, i) = zero, TMP(iresult, i) = zero');

      LUNROLL(j, 0, eval(ORDER - 1),`
      {
        V4 rsum = zero, isum = zero, rprod = one, iprod = zero;

        RLUNROLL(k, j, 0, `
          {
	    pushdef(`BINVAL', BINOMIAL(j, k))
	    ifelse(BINVAL, 1,`dnl', `const V4 factor = {BINVAL, BINVAL, BINVAL, BINVAL};')
	    	    
            rsum += ifelse(BINVAL, 1,, factor *) (rsrcxp[k] * rprod - isrcxp[k] * iprod);
            isum += ifelse(BINVAL, 1,, factor *) (isrcxp[k] * rprod + rsrcxp[k] * iprod);
	    popdef(`BINVAL')

            const V4 rnewprod = rprod * rx - iprod * ry;
            const V4 inewprod = rprod * ry + iprod * rx;

            rprod = rnewprod;
            iprod = inewprod;
          }')

	  ifelse(eval(j + 1), 1, `
	  		rsum -= rprod * srcmass;
          		isum -= iprod * srcmass;',
	  		`pushdef(`INVDENOM', esyscmd(echo 1/eval(j + 1) | bc -l));
	  		const V4 invdenom = {INVDENOM, INVDENOM, INVDENOM, INVDENOM};
	  		popdef(`DENOM')
          		const V4 term = srcmass * invdenom;
	  		rsum -= rprod * term;
          		isum -= iprod * term;')

          TMP(rresult, j) += rsum;
          TMP(iresult, j) += isum;
        }')

        LUNROLL(i, 0, eval(ORDER - 1), `rdstxp[i] =  TMP(rresult, i)[0] + TMP(rresult, i)[1] + TMP(rresult, i)[2] + TMP(rresult, i)[3];')
        LUNROLL(i, 0, eval(ORDER - 1), `idstxp[i] =  TMP(iresult, i)[0] + TMP(iresult, i)[1] + TMP(iresult, i)[2] + TMP(iresult, i)[3];')
      }

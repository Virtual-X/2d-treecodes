/*
*  potential-kernels.mc
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

#include <math.h>
define(NACC, 32)
#define EPS (10 * __DBL_EPSILON__)
#define MAX(a,b) (((a)>(b))?(a):(b))

realtype potential_p2p(const realtype * __restrict__ const _xsrc,
  const realtype * __restrict__ const _ysrc,
  const realtype * __restrict__ const _vsrc,
  const int nsources,
  const realtype xt,
  const realtype yt)
  {
    realtype dummy LUNROLL(`i', 0, NACC, `, TMP(s,i) = 0') ;

    const int nnice = NACC * (nsources / NACC);

    for(int i = 0; i < nnice; i += NACC)
    {
      const realtype * __restrict__ const xsrc = _xsrc + i;
      const realtype * __restrict__ const ysrc = _ysrc + i;
      const realtype * __restrict__ const vsrc = _vsrc + i;

      LUNROLL(j, 0, eval(NACC - 1), `
      const realtype TMP(xr, j) = xt - xsrc[j];')
      LUNROLL(j, 0, eval(NACC - 1), `
      const realtype TMP(yr, j) = yt - ysrc[j];')
      LUNROLL(j, 0, eval(NACC - 1), `
      TMP(s, j) += log(TMP(xr, j) * TMP(xr, j) + TMP(yr, j) * TMP(yr, j) + EPS) * vsrc[j];')
    }

    realtype sum = 0;

    for(int i = nnice; i < nsources; ++i)
    {
      const realtype xr = xt - _xsrc[i];
      const realtype yr = yt - _ysrc[i];

      sum += log(xr * xr + yr * yr + EPS) * _vsrc[i];
    }

    LUNROLL(i, 0, eval(NACC - 1), `sum += TMP(s, i);')

    return sum / 2;
  }

divert(ifelse(TUNED4AVXDP, 1, -1, 0))
  typedef realtype V4 __attribute__ ((vector_size (sizeof(realtype) * 4)));

  realtype potential_e2p(const realtype mass,
    const realtype rz,
    const realtype iz,
    const realtype * __restrict__ const rxp,
    const realtype * __restrict__ const ixp)
    {
      const realtype r2 = rz * rz + iz * iz;

      const realtype rinvz_1 = rz / r2;
      const realtype iinvz_1 = -iz / r2;

      LUNROLL(n, 2, eval(4), `
      const realtype TMP(rinvz, n) = TMP(rinvz, eval(n - 1)) * rinvz_1 - TMP(iinvz, eval(n - 1)) * iinvz_1;
      const realtype TMP(iinvz, n) = TMP(rinvz, eval(n - 1)) * iinvz_1 + TMP(iinvz, eval(n - 1)) * rinvz_1;')

      const V4 rz4 = { TMP(rinvz, 4), TMP(rinvz, 4), TMP(rinvz, 4), TMP(rinvz, 4) };
			const V4 iz4 = { TMP(iinvz, 4), TMP(iinvz, 4), TMP(iinvz, 4), TMP(iinvz, 4) };
      const V4 rbase_0 = { TMP(rinvz, 1), TMP(rinvz, 2), TMP(rinvz, 3), TMP(rinvz, 4) };
      const V4 ibase_0 = { TMP(iinvz, 1), TMP(iinvz, 2), TMP(iinvz, 3), TMP(iinvz, 4) };

      LUNROLL(j, 0, eval(ORDER/4 - 1), `
        const V4 TMP(rbase, eval(j + 1)) = TMP(rbase, j) * rz4 - TMP(ibase, j) * iz4;
				const V4 TMP(ibase, eval(j + 1)) = TMP(rbase, j) * iz4 + TMP(ibase, j) * rz4;

        const V4 TMP(rxp4, j) = { rxp[eval(4 * j)], rxp[eval(4 * j + 1)], rxp[eval(4 * j + 2)], rxp[eval(4 * j + 3)] };
        const V4 TMP(ixp4, j) = { ixp[eval(4 * j)], ixp[eval(4 * j + 1)], ixp[eval(4 * j + 2)], ixp[eval(4 * j + 3)] };

        const V4 TMP(rsum, j) = TMP(rxp4, j) * TMP(rbase, j) - TMP(ixp4, j) * TMP(ibase, j) ifelse(j, 0, ,+ TMP(rsum, eval(j - 1)));
      ')

      return  mass * log(r2) / 2 + TMP(rsum, eval(ORDER/4 - 1))[0] + TMP(rsum, eval(ORDER/4 - 1))[1] +
      TMP(rsum, eval(ORDER/4 - 1))[2] + TMP(rsum, eval(ORDER/4 - 1))[3];
    }

divert(0)
divert(ifelse(TUNED4AVXDP, 1, 0, -1))
#include "immintrin.h"
realtype potential_e2p(const realtype mass,
  const realtype rz,
  const realtype iz,
  const realtype * __restrict__ const rxp,
  const realtype * __restrict__ const ixp)
  {
    const realtype r2 = rz * rz + iz * iz;

    const realtype rinvz_1 = rz / r2;
    const realtype iinvz_1 = -iz / r2;

    LUNROLL(n, 2, eval(4), `
    const realtype TMP(rinvz, n) = TMP(rinvz, eval(n - 1)) * rinvz_1 - TMP(iinvz, eval(n - 1)) * iinvz_1;
    const realtype TMP(iinvz, n) = TMP(rinvz, eval(n - 1)) * iinvz_1 + TMP(iinvz, eval(n - 1)) * rinvz_1;')

    const __m256d rz4 = _mm256_set1_pd(TMP(rinvz, 4));
    const __m256d iz4 = _mm256_set1_pd(TMP(iinvz, 4));
    const __m256d rbase_0 = _mm256_set_pd(TMP(rinvz, 4), TMP(rinvz, 3), TMP(rinvz, 2), TMP(rinvz, 1));
    const __m256d ibase_0 = _mm256_set_pd(TMP(iinvz, 4), TMP(iinvz, 3), TMP(iinvz, 2), TMP(iinvz, 1));

    LUNROLL(j, 0, eval(ORDER/4 - 1), `
    ifelse(j, eval(ORDER/4 - 1), , const __m256d TMP(rbase, eval(j + 1)) = TMP(rbase, j) * rz4 - TMP(ibase, j) * iz4;)
    ifelse(j, eval(ORDER/4 - 1), , const __m256d TMP(ibase, eval(j + 1)) = TMP(rbase, j) * iz4 + TMP(ibase, j) * rz4;)

    const __m256d TMP(rxp4, j) = _mm256_loadu_pd(rxp + eval(4 * j));
    const __m256d TMP(ixp4, j) = _mm256_loadu_pd(ixp + eval(4 * j));

    const __m256d TMP(rsum, j) = (TMP(rxp4, j) * TMP(rbase, j) - TMP(ixp4, j) * TMP(ibase, j)) ifelse(j, 0, ,+ TMP(rsum, eval(j - 1)));
    ')

    return  mass * log(r2) / 2 + TMP(rsum, eval(ORDER/4 - 1))[0] + TMP(rsum, eval(ORDER/4 - 1))[1] +
    TMP(rsum, eval(ORDER/4 - 1))[2] + TMP(rsum, eval(ORDER/4 - 1))[3];
  }

  divert(0)

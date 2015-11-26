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

      V4 rz4 = { TMP(rinvz, 4), TMP(rinvz, 4), TMP(rinvz, 4), TMP(rinvz, 4) };
      V4 iz4 = { TMP(iinvz, 4), TMP(iinvz, 4), TMP(iinvz, 4), TMP(iinvz, 4) };
      V4 rbase = { TMP(rinvz, 1), TMP(rinvz, 2), TMP(rinvz, 3), TMP(rinvz, 4) };
      V4 ibase = { TMP(iinvz, 1), TMP(iinvz, 2), TMP(iinvz, 3), TMP(iinvz, 4) };
      V4 rsum = {0, 0, 0, 0};

      LUNROLL(j, 0, eval(ORDER/4 - 1), `
      {
        V4 tmp0 = rbase * rz4 - ibase * iz4;
        V4 tmp1 = rbase * iz4 + ibase * rz4;

        V4 rxp4 = { rxp[eval(4 * j)], rxp[eval(4 * j + 1)], rxp[eval(4 * j + 2)], rxp[eval(4 * j + 3)] };
        V4 ixp4 = { ixp[eval(4 * j)], ixp[eval(4 * j + 1)], ixp[eval(4 * j + 2)], ixp[eval(4 * j + 3)] };

        rsum += rxp4 * rbase - ixp4 * ibase;
        //isum += rxp4 * ibase + ixp4 * rbase;

        rbase = tmp0;
        ibase = tmp1;
      }
      ')

      return  mass * log(r2) / 2 + rsum[0] + rsum[1] + rsum[2] + rsum[3];
    }

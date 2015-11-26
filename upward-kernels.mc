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
#include <math.h>
#include <immintrin.h>
#include "upward-kernels.h"

#define EPS (10 * __DBL_EPSILON__)
#define MAX(a,b) (((a)>(b))?(a):(b))

define(P2E_KERNEL, treecode_p2e_order$1)
define(E2E_KERNEL, treecode_e2e_order$1)

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

    realtype r2 = 0;

    for(int i = 0; i < nsources; ++i)
    {
      const realtype xr = xsources[i] - xcom;
      const realtype yr = ysources[i] - ycom;

      r2 = MAX(r2, xr * xr + yr * yr);
    }

    *radius = sqrt(r2);

    const V4 zero = {0, 0, 0, 0};

    V4 dummy LUNROLL(n, 0, eval(ORDER - 1),`, TMP(rxp, n) = zero') LUNROLL(n, 0, eval(ORDER - 1),`, TMP(ixp, n) = zero');

    for(int i = 0; i < nsources; i += 4)
    {
      V4 rrp = zero, irp = zero, srcs = zero;

      LUNROLL(c, 0, 3, `rrp[c] = (i + c < nsources) ? xsources[i + c] : 0;')

      LUNROLL(c, 0, 3, `irp[c] = (i + c < nsources) ? ysources[i + c] : 0;')

      LUNROLL(c, 0, 3, `srcs[c] = (i + c < nsources) ? sources[i + c] : 0;')

      rrp -= xcom;
      irp -= ycom;

      V4 rprod = rrp, iprod = irp;

      TMP(rxp, 0) -= rprod * srcs;
      TMP(ixp, 0) -= iprod * srcs;

      LUNROLL(n, 1, eval(ORDER - 1),`
      const V4 TMP(rnewprod, n) = rprod * rrp - iprod * irp;
      const V4 TMP(inewprod, n) = rprod * irp + iprod * rrp;

      rprod = TMP(rnewprod, n);
      iprod = TMP(inewprod, n);

      const V4 TMP(term, n) = srcs / (realtype)(n + 1);

      TMP(rxp, n) -= rprod * TMP(term, n);
      TMP(ixp, n) -= iprod * TMP(term, n);
      ')
    }

    LUNROLL(n, 0, eval(ORDER - 1),`rexpansions[n] = TMP(rxp, n)[0] + TMP(rxp, n)[1] + TMP(rxp, n)[2] + TMP(rxp, n)[3];')

    LUNROLL(n, 0, eval(ORDER - 1),`iexpansions[n] = TMP(ixp, n)[0] + TMP(ixp, n)[1] + TMP(ixp, n)[2] + TMP(ixp, n)[3];')
  }

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
            rsum += BINOMIAL(j, k) * (rsrcxp[k] * rprod - isrcxp[k] * iprod);
            isum += BINOMIAL(j, k) * (isrcxp[k] * rprod + rsrcxp[k] * iprod);

            const V4 rnewprod = rprod * rx - iprod * ry;
            const V4 inewprod = rprod * ry + iprod * rx;

            rprod = rnewprod;
            iprod = inewprod;
          }')

          const V4 term = srcmass / (j + 1);

          rsum -= rprod * term;
          isum -= iprod * term;

          TMP(rresult, j) += rsum;
          TMP(iresult, j) += isum;
        }')

        LUNROLL(i, 0, eval(ORDER - 1), `rdstxp[i] =  TMP(rresult, i)[0] + TMP(rresult, i)[1] + TMP(rresult, i)[2] + TMP(rresult, i)[3];')
        LUNROLL(i, 0, eval(ORDER - 1), `idstxp[i] =  TMP(iresult, i)[0] + TMP(iresult, i)[1] + TMP(iresult, i)[2] + TMP(iresult, i)[3];')
      }

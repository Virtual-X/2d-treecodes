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
include(unroll.m4)


define(mysign, `ifelse(eval((-1)**($1)), -1,-,+)')
divert(0)
dnl
typedef realtype V4 __attribute__ ((vector_size (sizeof(realtype) * 4)));
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

#ifdef __cplusplus
extern "C"
#endif
void downward_e2l(
     const realtype * x0s,
     const realtype * y0s,
     const realtype * masses,
     const realtype ** __restrict__ const vrexpansions,
     const realtype ** __restrict__ const viexpansions,
     const int nexpansions,
     realtype * __restrict__ const rlocal,
     realtype * __restrict__ const ilocal)
  {
	for(int iexpansion = 0; iexpansion < nexpansions; ++iexpansion)
  	{
		const realtype x0 = x0s[iexpansion];
		const realtype y0 = y0s[iexpansion];
		const realtype mass = masses[iexpansion];
		const realtype * __restrict__ const rxp = vrexpansions[iexpansion];
		const realtype * __restrict__ const ixp = viexpansions[iexpansion];

		const realtype r2z0 = x0 * x0 + y0 * y0;
    		const realtype rlogmz0 = log(r2z0) / 2;
    		const realtype ilogmz0 = atan2(y0, x0) - M_PI;

    		const realtype rinvz_1 = +x0 / r2z0;
    		const realtype iinvz_1 = -y0 / r2z0;

    		dnl
    		LUNROLL(j, 1, eval(ORDER),`
    		ifelse(j, 1, , `
      		  const realtype TMP(rinvz, j) = TMP(rinvz, eval(j - 1)) * rinvz_1 - TMP(iinvz, eval(j - 1)) * iinvz_1;
      		  const realtype TMP(iinvz, j) = TMP(rinvz, eval(j - 1)) * iinvz_1 + TMP(iinvz, eval(j - 1)) * rinvz_1;')

      		  const realtype TMP(rcoeff, j) = rxp[eval(j - 1)] * TMP(rinvz, j) - ixp[eval(j - 1)] * TMP(iinvz, j);
      		  const realtype TMP(icoeff, j) = rxp[eval(j - 1)] * TMP(iinvz, j) + ixp[eval(j - 1)] * TMP(rinvz, j);
      		')

		{
			rlocal[0] += mass * rlogmz0 LUNROLL(k, 1, eval(ORDER),` mysign(k) TMP(rcoeff, k)  ');
        		ilocal[0] += mass * ilogmz0 LUNROLL(k, 1, eval(ORDER),` mysign(k) TMP(icoeff, k)  ');
      		}

      		LUNROLL(l, 1, eval(ORDER),`
      		{
			const realtype TMP(rtmp, l) = LUNROLL(k, 1, eval(ORDER),`
        		mysign(k) BINOMIAL(eval(l + k - 1), eval(k - 1)) * TMP(rcoeff, k)');

        		const realtype TMP(itmp, l) = LUNROLL(k, 1, eval(ORDER),`
        		mysign(k) BINOMIAL(eval(l + k - 1), eval(k - 1)) * TMP(icoeff, k)');

        		const realtype TMP(prefac, l) = - mass / l;
        		rlocal[l] += (TMP(prefac, l) + TMP(rtmp, l)) * TMP(rinvz, l) - TMP(itmp, l) * TMP(iinvz, l);
        		ilocal[l] += (TMP(prefac, l) + TMP(rtmp, l)) * TMP(iinvz, l) + TMP(itmp, l) * TMP(rinvz, l);
      		}')
	}

      	__asm__("L_END_DOWNWARD_E2L:");
    }

#ifdef __cplusplus
extern "C"
#endif
    void downward_l2p_tiled( const realtype rxbase,
      const realtype rybase,
      const realtype h,
      const realtype * __restrict__ const rlocal,
      const realtype * __restrict__ const ilocal,
      realtype * const xresult,
      realtype * const yresult, const int stride)
      {
	LUNROLL(iy, 0, 3,`
	{
          const V4 rz_1 = { LUNROLL(ix, 0, 3, `rxbase + ix * h,')};
          const V4 iz_1 = { LUNROLL(ix, 0, 3, `rybase + iy * h,')};

          const V4 TMP(rresult, 1) = { LUNROLL(ix, 0, 3, `rlocal[1], ')};
          const V4 TMP(iresult, 1) = { LUNROLL(ix, 0, 3, `ilocal[1], ')};

          LUNROLL(l, 2, eval(ORDER),`
          const V4 TMP(rz, l) = TMP(rz, eval(l - 1)) * rz_1 - TMP(iz, eval(l - 1)) * iz_1;

          const V4 TMP(iz, l) = TMP(rz, eval(l - 1)) * iz_1 + TMP(iz, eval(l - 1)) * rz_1;

          const V4 TMP(rresult, l) = TMP(rresult, eval(l - 1)) +
          l * (rlocal[l] * TMP(rz, eval(l - 1)) - ilocal[l] * TMP(iz, eval(l - 1)));

          const V4 TMP(iresult, l) = TMP(iresult, eval(l - 1)) +
          l * (rlocal[l] * TMP(iz, eval(l - 1)) + ilocal[l] * TMP(rz, eval(l - 1)));
          ')

          LUNROLL(ix, 0, 3,`
	  xresult[ix + stride * iy] += TMP(rresult, ORDER)[ix];
          yresult[ix + stride * iy] -= TMP(iresult, ORDER)[ix];')

        }')

	__asm__("L_END_DOWNWARD_L2P_TILED:");
      }

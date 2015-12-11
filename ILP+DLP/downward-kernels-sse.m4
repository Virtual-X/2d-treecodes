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
#include <emmintrin.h>
#include <math.h>

#if defined(__INTEL_COMPILER)
inline __m128d operator+(__m128d a, __m128d b){ return _mm_add_pd(a, b); }
inline __m128d operator/(__m128d a, __m128d b){ return _mm_div_pd(a, b); }
inline __m128d operator*(__m128d a, __m128d b){ return _mm_mul_pd(a, b); }
inline __m128d operator-(__m128d a, __m128d b){ return _mm_sub_pd(a, b); }
inline __m128d operator += (__m128d& a, __m128d b){ return a = _mm_add_pd(a, b); }
inline __m128d operator -= (__m128d& a, __m128d b){ return a = _mm_sub_pd(a, b); }
#endif

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
     const int ne,
     realtype * __restrict__ const rlocal,
     realtype * __restrict__ const ilocal)
  {
	for(int ie = 0; ie < ne; ie += 2)
  	{
		const __m128d x0 = _mm_loadu_pd(x0s + ie);
		const __m128d y0 = _mm_loadu_pd(y0s + ie);
		const __m128d mass = _mm_loadu_pd(masses + ie);

		LUNROLL(j, 0, 1, `const realtype * __restrict__ const TMP(rxp, j) = vrexpansions[ie + j];')
		LUNROLL(j, 0, 1, `const realtype * __restrict__ const TMP(ixp, j) = viexpansions[ie + j];')

		const __m128d r2z0 = x0 * x0 + y0 * y0;
    		//not needed as long as we evaluate grad(pot)
		//const __m128d rlogmz0 = _mm_log_pd(r2z0) * _mm_set1_pd(0.5);
    		//const __m128d ilogmz0 = _mm_atan2_pd(y0, x0) - _mm_set1_pd(M_PI);

    		const __m128d rinvz_1 = x0 / r2z0;
    		const __m128d iinvz_1 = _mm_setzero_pd() - y0 / r2z0;

    		dnl
    		LUNROLL(j, 1, eval(ORDER),`
    		ifelse(j, 1, , `
      		  const __m128d TMP(rinvz, j) = TMP(rinvz, eval(j - 1)) * rinvz_1 - TMP(iinvz, eval(j - 1)) * iinvz_1;
      		  const __m128d TMP(iinvz, j) = TMP(rinvz, eval(j - 1)) * iinvz_1 + TMP(iinvz, eval(j - 1)) * rinvz_1;')

		  const __m128d TMP(curr_rxp, j) = _mm_set_pd(RLUNROLL(c, 1, 0, 
		  	`(ie + c >= ne) ? 0 : TMP(rxp, c)[eval(j - 1)]ifelse(c,0,,`,') '));
		  const __m128d TMP(curr_ixp, j) = _mm_set_pd(RLUNROLL(c, 1, 0, 
		  	`(ie + c >= ne) ? 0 : TMP(ixp, c)[eval(j - 1)]ifelse(c,0,,`,') '));

      		  const __m128d TMP(rcoeff, j) = TMP(curr_rxp, j) * TMP(rinvz, j) - TMP(curr_ixp, j) * TMP(iinvz, j);
      		  const __m128d TMP(icoeff, j) = TMP(curr_rxp, j) * TMP(iinvz, j) + TMP(curr_ixp, j) * TMP(rinvz, j);
      		')

		/*not needed as long as we evaluate grad(pot)
		{
			__m128d rpartial = mass * rlogmz0 LUNROLL(k, 1, eval(ORDER),` mysign(k) TMP(rcoeff, k)  ');
        		__m128d ipartial = mass * ilogmz0 LUNROLL(k, 1, eval(ORDER),` mysign(k) TMP(icoeff, k)  ');
			
			if (ie + 2 <= ne)
			{
			   rpartial = _mm_hadd_pd(rpartial, rpartial);
			   ipartial = _mm_hadd_pd(ipartial, ipartial);
			}

			double tmp0, tmp1;
			_mm_store_sd(&tmp0, rpartial);
			_mm_store_sd(&tmp1, ipartial);

			rlocal[0] += tmp0;
       			ilocal[0] += tmp1;
      		}
		*/ 

      		LUNROLL(l, 1, eval(ORDER),`
      		{
			const __m128d TMP(prefac, l) = ifelse(l,1,
			`_mm_setzero_pd() - mass', 
		   	`mass * _mm_set1_pd(esyscmd(echo -1/eval(l) | bc --mathlib ))');

			pushdef(`BINFAC', `BINOMIAL(eval(l + k - 1), eval(k - 1)).f')dnl
			const __m128d TMP(rtmp, l) = TMP(prefac, l) LUNROLL(k, 1, eval(ORDER),`
        		mysign(k) ifelse(BINFAC,1.f,,`_mm_set1_pd(BINFAC) *') TMP(rcoeff, k)');

        		const __m128d TMP(itmp, l) = _mm_setzero_pd() LUNROLL(k, 1, eval(ORDER),`
        		mysign(k) ifelse(BINFAC,1.f,,`_mm_set1_pd(BINFAC) *') TMP(icoeff, k)');
			popdef(`BINFAC')dnl
        		
        		__m128d rpartial = TMP(rtmp, l) * TMP(rinvz, l) - TMP(itmp, l) * TMP(iinvz, l);
        		__m128d ipartial = TMP(rtmp, l) * TMP(iinvz, l) + TMP(itmp, l) * TMP(rinvz, l);

			if (ie + 2 <= ne)
			{
			   rpartial = _mm_hadd_pd(rpartial, rpartial);
			   ipartial = _mm_hadd_pd(ipartial, ipartial);
			}

			double tmp0, tmp1;
			_mm_store_sd(&tmp0, rpartial);
			_mm_store_sd(&tmp1, ipartial);

			rlocal[l] += tmp0;
       			ilocal[l] += tmp1;
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
	LUNROLL(pass, 0, 1,`
	{
          const __m128d rz_1 = _mm_set1_pd(rxbase) + _mm_set_pd(eval(2 * pass) + 1, eval(2 * pass)) * _mm_set1_pd(h);
          const __m128d iz_1 = _mm_set1_pd(rybase + iy * h);

          const __m128d TMP(rresult, 1) = _mm_set1_pd(rlocal[1]);
          const __m128d TMP(iresult, 1) = _mm_set1_pd(ilocal[1]);

          LUNROLL(l, 2, eval(ORDER),`
          const __m128d TMP(rz, l) = TMP(rz, eval(l - 1)) * rz_1 - TMP(iz, eval(l - 1)) * iz_1;

          const __m128d TMP(iz, l) = TMP(rz, eval(l - 1)) * iz_1 + TMP(iz, eval(l - 1)) * rz_1;

          const __m128d TMP(rresult, l) = TMP(rresult, eval(l - 1)) +
          _mm_set1_pd(l) * (_mm_set1_pd(rlocal[l]) * TMP(rz, eval(l - 1)) - _mm_set1_pd(ilocal[l]) * TMP(iz, eval(l - 1)));

          const __m128d TMP(iresult, l) = TMP(iresult, eval(l - 1)) +
          _mm_set1_pd(l) * (_mm_set1_pd(rlocal[l]) * TMP(iz, eval(l - 1)) + _mm_set1_pd(ilocal[l]) * TMP(rz, eval(l - 1)));
          ')

	  _mm_storeu_pd(xresult + eval(2 * pass) + stride * iy, _mm_loadu_pd(xresult + eval(2 * pass) + stride * iy) + TMP(rresult, ORDER));
	  _mm_storeu_pd(yresult + eval(2 * pass) + stride * iy, _mm_loadu_pd(yresult + eval(2 * pass) + stride * iy) - TMP(iresult, ORDER));
        }')')

	__asm__("L_END_DOWNWARD_L2P_TILED:");
      }

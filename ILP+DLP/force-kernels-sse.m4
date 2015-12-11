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

include(unroll.m4)
#include <pmmintrin.h>                                                                                                                                                                                                                                                         

#if defined(__INTEL_COMPILER)                                                                                                                                                                                                                                                  
inline __m128d operator+(__m128d a, __m128d b){ return _mm_add_pd(a, b); }
inline __m128d operator/(__m128d a, __m128d b){ return _mm_div_pd(a, b); }
inline __m128d operator*(__m128d a, __m128d b){ return _mm_mul_pd(a, b); }
inline __m128d operator-(__m128d a, __m128d b){ return _mm_sub_pd(a, b); }
inline __m128d operator += (__m128d& a, __m128d b){ return a = _mm_add_pd(a, b); }
inline __m128d operator -= (__m128d& a, __m128d b){ return a = _mm_sub_pd(a, b); }
#endif                                                                                                                                                                                                                                                                          
#define EPS (10 * __DBL_EPSILON__)                                                                                                                                                                                                                                             
#define MAX(a,b) (((a)>(b))?(a):(b))                                                                                                                                                                                                                                           


define(NACC, 4)

#ifdef __cplusplus
extern "C"
#endif                       
void force_p2p(const realtype * __restrict__ const _xsrc,
	const realtype * __restrict__ const _ysrc,
	const realtype * __restrict__ const _vsrc,
	const int nsources,
	const realtype xt,
	const realtype yt,
	realtype * const xresult,
	realtype * const yresult)
	{
		__asm__("L_START_FORCE_P2P:");


		const __m128d xt2 = _mm_set1_pd(xt);
		const __m128d yt2 = _mm_set1_pd(yt);
		const __m128d EPS2 = _mm_set1_pd(EPS);
		__m128d LUNROLL(`i', 0, NACC, `ifelse(i, 0,, `,')
		TMP(xs,i) =_mm_setzero_pd(),
		TMP(ys,i) =_mm_setzero_pd()') ;

		const int nnice = eval(2 * NACC) * (nsources / eval(2 * NACC));

		for(int i = 0; i < nnice; i += eval(2 * NACC))
		{
			const realtype * __restrict__ const xsrc = _xsrc + i;
			const realtype * __restrict__ const ysrc = _ysrc + i;
			const realtype * __restrict__ const vsrc = _vsrc + i;

			LUNROLL(j, 0, eval(NACC - 1), `
			pushdef(base, `eval(2 * j)')
			const __m128d TMP(xsrc, j) = _mm_loadu_pd(xsrc + base);
			const __m128d TMP(ysrc, j) = _mm_loadu_pd(ysrc + base);
			const __m128d TMP(vsrc, j) = _mm_loadu_pd(vsrc + base);
			popdef(base)

			const __m128d TMP(xr, j) = xt2 - TMP(xsrc, j);
			const __m128d TMP(yr, j) = yt2 - TMP(ysrc, j);

			const __m128d TMP(factor, j) = TMP(vsrc, j) / (TMP(xr, j) * TMP(xr, j) + TMP(yr, j) * TMP(yr, j) + EPS2) ;')

			LUNROLL(j, 0, eval(NACC - 1), `
			TMP(xs, j) += TMP(xr, j) * TMP(factor, j);
			TMP(ys, j) += TMP(yr, j) * TMP(factor, j);
			')
		}

		__m128d xsum2 = _mm_setzero_pd(), ysum2 = _mm_setzero_pd();
		LUNROLL(i, 0, eval(NACC - 1), `
		xsum2 += TMP(xs, i);
		ysum2 += TMP(ys, i);')

		realtype xsum, ysum;
		_mm_store_sd(&xsum, _mm_hadd_pd(xsum2, xsum2));
		_mm_store_sd(&ysum, _mm_hadd_pd(ysum2, ysum2));
		

		for(int i = nnice; i < nsources; ++i)
		{
			const realtype xr = xt - _xsrc[i];
			const realtype yr = yt - _ysrc[i];
			const realtype factor = _vsrc[i] / (xr * xr + yr * yr + EPS);

			xsum += xr * factor;
			ysum += yr * factor;
		}

		*xresult = xsum;
		*yresult = ysum;
		__asm__("L_END_FORCE_P2P:");
	}


#ifdef __cplusplus
extern "C"
#endif                       
	void force_e2p(const realtype mass,
		const realtype rz,
		const realtype iz,
		const realtype * __restrict__ const rxp,
		const realtype * __restrict__ const ixp,
		realtype * const xresult,
		realtype * const yresult)
		{
			const realtype r2 = rz * rz + iz * iz;
			
			const realtype rinvz_1 = rz / r2;
			const realtype iinvz_1 = -iz / r2;

			LUNROLL(n, 2, 3, `
			const realtype TMP(rinvz, n) = TMP(rinvz, eval(n - 1)) * rinvz_1 - TMP(iinvz, eval(n - 1)) * iinvz_1;
			const realtype TMP(iinvz, n) = TMP(rinvz, eval(n - 1)) * iinvz_1 + TMP(iinvz, eval(n - 1)) * rinvz_1;')dnl

			const __m128d rz2 = _mm_set1_pd(TMP(rinvz, 2));
			const __m128d iz2 = _mm_set1_pd(TMP(iinvz, 2)); 

			const __m128d rbase_0 = _mm_set_pd(TMP(rinvz, 3), TMP(rinvz, 2));
			const __m128d ibase_0 = _mm_set_pd(TMP(iinvz, 3), TMP(iinvz, 2));

			const __m128d TMP(k2, 0) = _mm_set_pd(2, 1);dnl

			LUNROLL(j, 0, eval(ORDER/2 - 1), `
				const __m128d TMP(rbase, eval(j + 1)) = TMP(rbase, j) * rz2 - TMP(ibase, j) * iz2;
				const __m128d TMP(ibase, eval(j + 1)) = TMP(rbase, j) * iz2 + TMP(ibase, j) * rz2;

				const __m128d TMP(rxp2, j) = _mm_loadu_pd(rxp + eval(2 * j));
				const __m128d TMP(ixp2, j) = _mm_loadu_pd(ixp + eval(2 * j));

				const __m128d TMP(rsum, j) = (TMP(rxp2, j) * TMP(rbase, j) - TMP(ixp2, j) * TMP(ibase, j)) * TMP(k2, j) ifelse(j, 0, ,+ TMP(rsum, eval(j - 1)));
				const __m128d TMP(isum, j) = (TMP(rxp2, j) * TMP(ibase, j) + TMP(ixp2, j) * TMP(rbase, j)) * TMP(k2, j) ifelse(j, 0, ,+ TMP(isum, eval(j - 1)));

				const __m128d TMP(k2, eval(j+1)) = TMP(k2, j) + _mm_set1_pd(2);
			')dnl

			double rpartial, ipartial;
			_mm_store_sd(&rpartial, _mm_hadd_pd(TMP(rsum, eval(ORDER/2 - 1)), TMP(rsum, eval(ORDER/2 - 1))));
			_mm_store_sd(&ipartial, _mm_hadd_pd(TMP(isum, eval(ORDER/2 - 1)), TMP(isum, eval(ORDER/2 - 1))));


			*xresult = mass * rinvz_1 - rpartial;
			*yresult = -(mass * iinvz_1 - ipartial);

			__asm__("L_END_FORCE_E2P:");
		}


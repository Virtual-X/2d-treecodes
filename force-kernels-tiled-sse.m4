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
#define EPS (10 * __DBL_EPSILON__)
#include <emmintrin.h>
#if defined(__INTEL_COMPILER)
inline __m128d operator+(__m128d a, __m128d b){ return _mm_add_pd(a, b); }
inline __m128d operator/(__m128d a, __m128d b){ return _mm_div_pd(a, b); }
inline __m128d operator*(__m128d a, __m128d b){ return _mm_mul_pd(a, b); }
inline __m128d operator-(__m128d a, __m128d b){ return _mm_sub_pd(a, b); }
inline __m128d operator += (__m128d a, __m128d b){ return a = _mm_add_pd(a, b); }
inline __m128d operator -= (__m128d a, __m128d b){ return a = _mm_sub_pd(a, b); }
#endif


	void force_p2p_tiled(const realtype * __restrict__ const xsrc,
			 const realtype * __restrict__ const ysrc,
			 const realtype * __restrict__ const vsrc,
			 const int nsources,
			 const realtype _xt,
			 const realtype _yt,
			 const realtype h,
			 realtype * const xresult,
			 realtype * const yresult,
			 const int stride)
	{
		const __m128d xtA = _mm_set1_pd(_xt) + _mm_set_pd(1,0) * _mm_set1_pd(h);
		const __m128d xtB = _mm_set1_pd(_xt) + _mm_set_pd(3,2) * _mm_set1_pd(h);
		const __m128d eps = _mm_set1_pd(EPS);

		LUNROLL(iy, 0, 3, `
			    const realtype TMP(syt, iy) = _yt + iy * h;
			    const __m128d TMP(yt, iy) = _mm_set1_pd(TMP(syt, iy)); ')

		LUNROLL(iy, 0, 3, `
		__m128d TMP(xsA, iy) =_mm_setzero_pd();
		__m128d TMP(xsB, iy) =_mm_setzero_pd();
		__m128d TMP(ysA, iy) =_mm_setzero_pd();
		__m128d TMP(ysB, iy) =_mm_setzero_pd();
		')

		const int nnice = 4 * (nsources / 4);

		for(int j = 0; j < nnice; j += 4)
		{
			LUNROLL(pass, 0, 3, `
			{
				const __m128d ycurr = _mm_set1_pd(ysrc[j + pass]);
				const __m128d vcurr = _mm_set1_pd(vsrc[j + pass]);

				const __m128d xrA = xtA - _mm_set1_pd(xsrc[j + pass]);
				const __m128d xrB = xtB - _mm_set1_pd(xsrc[j + pass]);

				const __m128d xr2A = xrA * xrA;
				const __m128d xr2B = xrB * xrB;

				LUNROLL(iy, 0, 3, `
				    const __m128d TMP(yr, iy) = TMP(yt, iy) - ycurr;
				    const __m128d TMP(factorA, iy) = vcurr / (xr2A + TMP(yr, iy) * TMP(yr, iy) + eps);
				    const __m128d TMP(factorB, iy) = vcurr / (xr2B + TMP(yr, iy) * TMP(yr, iy) + eps);')

				LUNROLL(iy, 0, 3, `
				    TMP(xsA, iy) += xrA * TMP(factorA, iy);
				    TMP(xsB, iy) += xrB * TMP(factorB, iy);
				    TMP(ysA, iy) += TMP(yr, iy) * TMP(factorA, iy);
				    TMP(ysB, iy) += TMP(yr, iy) * TMP(factorB, iy);')
			}')
		}

		for(int j = nnice; j < nsources; ++j)
		{
			const __m128d ycurr = _mm_set1_pd(ysrc[j]);
			const __m128d vcurr = _mm_set1_pd(vsrc[j]);

			const __m128d xrA = xtA - _mm_set1_pd(xsrc[j]);
			const __m128d xrB = xtB - _mm_set1_pd(xsrc[j]);

			const __m128d xr2A = xrA * xrA;
			const __m128d xr2B = xrB * xrB;

			LUNROLL(iy, 0, 3, `
			    const __m128d TMP(yr, iy) = TMP(yt, iy) - ycurr;
			    const __m128d TMP(factorA, iy) = vcurr / (xr2A + TMP(yr, iy) * TMP(yr, iy) + eps);
			    const __m128d TMP(factorB, iy) = vcurr / (xr2B + TMP(yr, iy) * TMP(yr, iy) + eps);')

			LUNROLL(iy, 0, 3, `
			    TMP(xsA, iy) += xrA * TMP(factorA, iy);
			    TMP(xsB, iy) += xrB * TMP(factorB, iy);
			    TMP(ysA, iy) += TMP(yr, iy) * TMP(factorA, iy);
			    TMP(ysB, iy) += TMP(yr, iy) * TMP(factorB, iy);')
		}

		LUNROLL(iy, 0, 3, `
			_mm_storeu_pd(xresult + stride * iy, _mm_loadu_pd(xresult + stride * iy) + TMP(xsA, iy));
			_mm_storeu_pd(xresult + 2 + stride * iy, _mm_loadu_pd(xresult + 2 + stride * iy) + TMP(xsB, iy));
			_mm_storeu_pd(yresult + stride * iy,	_mm_loadu_pd(yresult + stride * iy) + TMP(ysA, iy));
			_mm_storeu_pd(yresult + 2 +stride * iy, _mm_loadu_pd(yresult + 2 + stride * iy) + TMP(ysB, iy));
			')

	}


	void force_e2p_tiled(
		const realtype mass,
		const realtype scalar_rz,
		const realtype scalar_iz,
		const realtype h,
		const realtype * __restrict__ const rxp,
		const realtype * __restrict__ const ixp,
		realtype * const xresult,
		realtype * const yresult,
		const int stride)
		{
			const __m128d mass2 = _mm_set1_pd(mass);
			const __m128d xdisplA = _mm_set_pd(1,0) * _mm_set1_pd(h);
			const __m128d xdisplB = _mm_set_pd(3,2) * _mm_set1_pd(h);	   
			const __m128d rz0A = _mm_set1_pd(scalar_rz);
			const __m128d rz0B = _mm_set1_pd(scalar_rz);

			const __m128d rzA = rz0A + xdisplA;
			const __m128d rzB = rz0B + xdisplB;
			const __m128d rz2A = rzA * rzA;
			const __m128d rz2B = rzB * rzB;

			LUNROLL(iy, 0, 3, `const realtype TMP(siz,iy) = scalar_iz + iy * h;')

			LUNROLL(iy, 0, 3, `const __m128d TMP(iz, iy) = _mm_set1_pd(TMP(siz,iy));')

			LUNROLL(iy, 0, 3, `
			const __m128d TMP(iz2, iy) = TMP(iz,iy) * TMP(iz, iy);
			const __m128d TMP(r2A, iy) = rz2A + TMP(iz2, iy);
			const __m128d TMP(r2B, iy) = rz2B + TMP(iz2, iy);
			')

			LUNROLL(iy, 0, 3, `
			const __m128d TMP(rinvzA,iy) = rzA / TMP(r2A, iy);
			const __m128d TMP(rinvzB,iy) = rzB / TMP(r2B, iy);
			')

			LUNROLL(iy, 0, 3, `
			const __m128d TMP(iinvzA,iy) = -TMP(iz, iy) / TMP(r2A, iy);
			const __m128d TMP(iinvzB,iy) = -TMP(iz, iy) / TMP(r2B, iy);
			')

			LUNROLL(iy, 0, 3, `
			__m128d TMP(xsA, iy) = mass2 * TMP(rinvzA, iy);
			__m128d TMP(xsB, iy) = mass2 * TMP(rinvzB, iy); 
			')

 			LUNROLL(iy, 0, 3, `
			__m128d TMP(ysA, iy) = mass2 * TMP(iinvzA, iy);
			__m128d TMP(ysB, iy) = mass2 * TMP(iinvzB, iy);
			')

			LUNROLL(iy, 0, 3, `
			__m128d TMP(rprodA, iy) = TMP(rinvzA, iy);
			__m128d TMP(rprodB, iy) = TMP(rinvzB, iy);
			')

 			LUNROLL(iy, 0, 3, `
			__m128d TMP(iprodA, iy) = TMP(iinvzA, iy);
			__m128d TMP(iprodB, iy) = TMP(iinvzB, iy); 
			')

			LUNROLL(n, 0, eval(ORDER - 1), `
			{
			   LUNROLL(iy, 0, 3, `
			   const __m128d TMP(rtmpA, iy) = TMP(rinvzA, iy) * TMP(rprodA, iy) - TMP(iinvzA, iy) * TMP(iprodA, iy);
			   const __m128d TMP(rtmpB, iy) = TMP(rinvzB, iy) * TMP(rprodB, iy) - TMP(iinvzB, iy) * TMP(iprodB, iy);
			   ')

 			   LUNROLL(iy, 0, 3, `
			   const __m128d TMP(itmpA, iy) = TMP(iinvzA, iy) * TMP(rprodA, iy) + TMP(rinvzA, iy) * TMP(iprodA, iy);
			   const __m128d TMP(itmpB, iy) = TMP(iinvzB, iy) * TMP(rprodB, iy) + TMP(rinvzB, iy) * TMP(iprodB, iy);
			   ')

			   LUNROLL(iy, 0, 3, `
			   TMP(rprodA, iy) = TMP(rtmpA, iy);
			   TMP(rprodB, iy) = TMP(rtmpB, iy);
			   ')

			   LUNROLL(iy, 0, 3, `
			   TMP(iprodA, iy) = TMP(itmpA, iy);
			   TMP(iprodB, iy) = TMP(itmpB, iy);
			   ')

			   const __m128d rxp4 = _mm_set1_pd(rxp[n]);
			   const __m128d ixp4 = _mm_set1_pd(ixp[n]);

			   ifelse(1,eval(n + 1),,`const __m128d pre4 = _mm_set1_pd(eval(n+1));')
			   pushdef(`FACTOR', ifelse(1,eval(n + 1), ,pre4 *))

			   LUNROLL(iy, 0, 3, `
			   TMP(xsA, iy) -= FACTOR (TMP(rprodA, iy) * rxp4 - TMP(iprodA, iy) * ixp4);
			   TMP(xsB, iy) -= FACTOR (TMP(rprodB, iy) * rxp4 - TMP(iprodB, iy) * ixp4);
			   ')

 			   LUNROLL(iy, 0, 3, `
			   TMP(ysA, iy) -= FACTOR (TMP(rprodA, iy) * ixp4 + TMP(iprodA, iy) * rxp4);
			   TMP(ysB, iy) -= FACTOR (TMP(rprodB, iy) * ixp4 + TMP(iprodB, iy) * rxp4);
			   ')

			   popdef(`FACTOR')
			}')

			LUNROLL(iy, 0, 3, `
			_mm_storeu_pd(xresult + stride * iy, _mm_loadu_pd(xresult + stride * iy) + TMP(xsA, iy));
			_mm_storeu_pd(xresult + 2 + stride * iy, _mm_loadu_pd(xresult + 2 + stride * iy) + TMP(xsB, iy));
			_mm_storeu_pd(yresult + stride * iy, _mm_loadu_pd(yresult + stride * iy) - TMP(ysA, iy));
			_mm_storeu_pd(yresult + 2 + stride * iy, _mm_loadu_pd(yresult + 2 + stride * iy) - TMP(ysB, iy));
			')

			__asm__("L_END_FORCE_E2P_TILED:");
		}
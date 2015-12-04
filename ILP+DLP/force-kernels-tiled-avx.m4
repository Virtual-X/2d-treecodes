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

#include <immintrin.h>

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
		const __m256d xt = _mm256_set1_pd(_xt) + _mm256_set_pd(3,2,1,0) * _mm256_set1_pd(h);
		const __m256d eps = _mm256_set1_pd(EPS);

		LUNROLL(iy, 0, 3, `
			    const realtype TMP(syt, iy) = _yt + iy * h;
			    const __m256d TMP(yt,iy) = _mm256_set1_pd(TMP(syt, iy)); ')

		LUNROLL(iy, 0, 3, `__m256d TMP(xs, iy) =_mm256_setzero_pd();')
		LUNROLL(iy, 0, 3, `__m256d TMP(ys, iy) =_mm256_setzero_pd();')

		const int nnice = 4 * (nsources / 4);

		for(int j = 0; j < nnice; j += 4)
		{
			LUNROLL(pass, 0, 3, `
			{
				const __m256d ycurr = _mm256_set1_pd(ysrc[j + pass]);
				const __m256d vcurr = _mm256_set1_pd(vsrc[j + pass]);

				const __m256d xr = xt - _mm256_set1_pd(xsrc[j + pass]);
				const __m256d xr2 = xr * xr;

				LUNROLL(iy, 0, 3, `
				    const __m256d TMP(yr, iy) = TMP(yt, iy) - ycurr;
				    const __m256d TMP(factor, iy) = vcurr / (xr2 + TMP(yr, iy) * TMP(yr, iy) + eps);')

				LUNROLL(iy, 0, 3, `
				    TMP(xs, iy) += xr * TMP(factor, iy);
				    TMP(ys, iy) += TMP(yr, iy) * TMP(factor, iy);')
			}')
		}

		for(int j = nnice; j < nsources; ++j)
		{
			const __m256d xcurr = _mm256_set1_pd(xsrc[j]);
			const __m256d ycurr = _mm256_set1_pd(ysrc[j]);
			const __m256d vcurr = _mm256_set1_pd(vsrc[j]);

			const __m256d xr = xt - xcurr;
			const __m256d xr2 = xr * xr;

			LUNROLL(iy, 0, 3, `
				    const __m256d TMP(yr, iy) = TMP(yt, iy) - ycurr;
				    const __m256d TMP(factor, iy) = vcurr / (xr2 + TMP(yr, iy) * TMP(yr, iy) + eps);')

			LUNROLL(iy, 0, 3, `
				    TMP(xs, iy) += xr * TMP(factor, iy);
				    TMP(ys, iy) += TMP(yr, iy) * TMP(factor, iy);')
		}

		LUNROLL(iy, 0, 3, `
			_mm256_storeu_pd(xresult + stride * iy,
					_mm256_loadu_pd(xresult + stride * iy) + TMP(xs, iy));
			_mm256_storeu_pd(yresult + stride * iy,
					_mm256_loadu_pd(yresult + stride * iy) + TMP(ys, iy));
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
			const __m256d mass4 = _mm256_set1_pd(mass);
			const __m256d xdispl = _mm256_set_pd(3,2,1,0) * _mm256_set1_pd(h);
			const __m256d rz0 = _mm256_set1_pd(scalar_rz);

			const __m256d rz = rz0 + xdispl;
			const __m256d rz2 = rz * rz;

			LUNROLL(iy, 0, 3, `const realtype TMP(siz,iy) = scalar_iz + iy * h;')

			LUNROLL(iy, 0, 3, `const __m256d TMP(iz, iy) = _mm256_set1_pd(TMP(siz,iy));')

			LUNROLL(iy, 0, 3, `const __m256d TMP(r2, iy) = rz2 + TMP(iz,iy) * TMP(iz, iy);')

			LUNROLL(iy, 0, 3, `const __m256d TMP(rinvz,iy) = rz / TMP(r2, iy);')

			LUNROLL(iy, 0, 3, `const __m256d TMP(iinvz,iy) = -TMP(iz, iy) / TMP(r2, iy);')

			LUNROLL(iy, 0, 3, `__m256d TMP(xs, iy) = mass4 * TMP(rinvz, iy);')

 			LUNROLL(iy, 0, 3, `__m256d TMP(ys, iy) = mass4 * TMP(iinvz, iy);')

			LUNROLL(iy, 0, 3, `__m256d TMP(rprod, iy) = TMP(rinvz, iy);')

 			LUNROLL(iy, 0, 3, `__m256d TMP(iprod, iy) = TMP(iinvz, iy);')

			LUNROLL(n, 0, eval(ORDER - 1), `
			{
			   LUNROLL(iy, 0, 3, `const __m256d TMP(rtmp, iy) = TMP(rinvz, iy) * TMP(rprod, iy) - TMP(iinvz, iy) * TMP(iprod, iy);')
 			   LUNROLL(iy, 0, 3, `const __m256d TMP(itmp, iy) = TMP(iinvz, iy) * TMP(rprod, iy) + TMP(rinvz, iy) * TMP(iprod, iy);')

			   LUNROLL(iy, 0, 3, `TMP(rprod, iy) = TMP(rtmp, iy);')
			   LUNROLL(iy, 0, 3, `TMP(iprod, iy) = TMP(itmp, iy);')

			   const __m256d rxp4 = _mm256_set1_pd(rxp[n]);
			   const __m256d ixp4 = _mm256_set1_pd(ixp[n]);

			   ifelse(1,eval(n + 1),,`const __m256d pre4 = _mm256_set1_pd(eval(n+1));')
			   pushdef(`FACTOR', ifelse(1,eval(n + 1), ,pre4 *))

			   LUNROLL(iy, 0, 3, `TMP(xs, iy) -= FACTOR (TMP(rprod, iy) * rxp4 - TMP(iprod, iy) * ixp4);')
 			   LUNROLL(iy, 0, 3, `TMP(ys, iy) -= FACTOR (TMP(rprod, iy) * ixp4 + TMP(iprod, iy) * rxp4);')

			   popdef(`FACTOR')
			}')

			LUNROLL(iy, 0, 3, `
			_mm256_storeu_pd(xresult + stride * iy,
					_mm256_loadu_pd(xresult + stride * iy) + TMP(xs, iy));
			_mm256_storeu_pd(yresult + stride * iy,
					_mm256_loadu_pd(yresult + stride * iy) - TMP(ys, iy));
			')

			__asm__("L_END_FORCE_E2P_TILED:");
		}

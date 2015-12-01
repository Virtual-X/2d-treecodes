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
	typedef realtype V4 __attribute__ ((vector_size (sizeof(realtype) * 4)));
	
divert(ifelse(TUNED4AVXDP, 1, -1, 0))
	void force_e2p_tiled(
		const realtype mass,
		const realtype scalar_rz,
		const realtype scalar_iz,
		const realtype h,
		const realtype * __restrict__ const rxp,
		const realtype * __restrict__ const ixp,
		realtype * const xresult,
		realtype * const yresult)
		{
			const V4 mass4 = {mass, mass, mass, mass};
			const V4 xdispl = {0, h, 2 * h, 3 * h};
			const V4 rz0 = {scalar_rz, scalar_rz, scalar_rz, scalar_rz};

			const V4 rz = rz0 + xdispl;
			const V4 rz2 = rz * rz;

			LUNROLL(iy, 0, 3, `const realtype TMP(siz,iy) = scalar_iz + iy * h;')

			LUNROLL(iy, 0, 3, `const V4 TMP(iz, iy) = { TMP(siz,iy), TMP(siz,iy), TMP(siz,iy), TMP(siz,iy)};')

			LUNROLL(iy, 0, 3, `const V4 TMP(r2, iy) = rz2 + TMP(iz,iy) * TMP(iz, iy);')

			LUNROLL(iy, 0, 3, `const V4 TMP(rinvz,iy) = rz / TMP(r2, iy);')

			LUNROLL(iy, 0, 3, `const V4 TMP(iinvz,iy) = -TMP(iz, iy) / TMP(r2, iy);')

			LUNROLL(iy, 0, 3, `V4 TMP(xs, iy) = mass4 * TMP(rinvz, iy);')

 			LUNROLL(iy, 0, 3, `V4 TMP(ys, iy) = mass4 * TMP(iinvz, iy);')

			LUNROLL(iy, 0, 3, `V4 TMP(rprod, iy) = TMP(rinvz, iy);')

 			LUNROLL(iy, 0, 3, `V4 TMP(iprod, iy) = TMP(iinvz, iy);')

			LUNROLL(n, 0, eval(ORDER - 1), `
			{
			   LUNROLL(iy, 0, 3, `const V4 TMP(rtmp, iy) = TMP(rinvz, iy) * TMP(rprod, iy) - TMP(iinvz, iy) * TMP(iprod, iy);')
 			   LUNROLL(iy, 0, 3, `const V4 TMP(itmp, iy) = TMP(iinvz, iy) * TMP(rprod, iy) + TMP(rinvz, iy) * TMP(iprod, iy);')

			   LUNROLL(iy, 0, 3, `TMP(rprod, iy) = TMP(rtmp, iy);')
			   LUNROLL(iy, 0, 3, `TMP(iprod, iy) = TMP(itmp, iy);')

			   const V4 rxp4 = {rxp[n], rxp[n], rxp[n], rxp[n]};
			   const V4 ixp4 = {ixp[n], ixp[n], ixp[n], ixp[n]};

			   ifelse(1,eval(n + 1),,`const V4 pre4 = {eval(n+1), eval(n+1), eval(n+1), eval(n+1)};')
			   pushdef(`FACTOR', ifelse(1,eval(n + 1), ,pre4 *))

			   LUNROLL(iy, 0, 3, `TMP(xs, iy) -= FACTOR (TMP(rprod, iy) * rxp4 - TMP(iprod, iy) * ixp4);')
 			   LUNROLL(iy, 0, 3, `TMP(ys, iy) -= FACTOR (TMP(rprod, iy) * ixp4 + TMP(iprod, iy) * rxp4);')

			   popdef(`FACTOR')
			}')

			LUNROLL(iy, 0, 3, `
			LUNROLL(ix, 0, 3, `xresult[eval(4 * iy) + ix] += TMP(xs, iy)[ix];')
 			LUNROLL(ix, 0, 3, `yresult[eval(4 * iy) + ix] -= TMP(ys, iy)[ix];')
			')

			__asm__("L_END_FORCE_E2P_TILED:");
		}
		
divert(0)
divert(ifelse(TUNED4AVXDP, 1, 0, -1))
#include "immintrin.h"

	void force_e2p_tiled(
		const realtype mass,
		const realtype scalar_rz,
		const realtype scalar_iz,
		const realtype h,
		const realtype * __restrict__ const rxp,
		const realtype * __restrict__ const ixp,
		realtype * const xresult,
		realtype * const yresult)
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
			_mm256_storeu_pd(xresult + eval(4 * iy),
					_mm256_loadu_pd(xresult + eval(4 * iy)) + TMP(xs, iy));
			_mm256_storeu_pd(yresult + eval(4 * iy),
					_mm256_loadu_pd(yresult + eval(4 * iy)) - TMP(ys, iy));
			')

			__asm__("L_END_FORCE_E2P_TILED:");
		}
divert(0)

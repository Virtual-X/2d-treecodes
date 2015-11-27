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
#define MAX(a,b) (((a)>(b))?(a):(b))

void force_p2p(const realtype * __restrict__ const _xsrc,
	const realtype * __restrict__ const _ysrc,
	const realtype * __restrict__ const _vsrc,
	const int nsources,
	const realtype xt,
	const realtype yt,
	realtype * const xresult,
	realtype * const yresult)
	{
		realtype dummy LUNROLL(`i', 0, NACC, `, TMP(xs,i) = 0, TMP(ys,i) = 0') ;

		const int nnice = NACC * (nsources / NACC);

		for(int i = 0; i < nnice; i += NACC)
		{
			const realtype * __restrict__ const xsrc = _xsrc + i;
			const realtype * __restrict__ const ysrc = _ysrc + i;
			const realtype * __restrict__ const vsrc = _vsrc + i;

			LUNROLL(j, 0, eval(NACC - 1), `
			const realtype TMP(xr, j) = xt - xsrc[j];
			const realtype TMP(yr, j) = yt - ysrc[j];
			const realtype TMP(factor, j) = vsrc[j] * (((realtype)1) / (TMP(xr, j) * TMP(xr, j) + TMP(yr, j) * TMP(yr, j) + EPS)) ;')

			LUNROLL(j, 0, eval(NACC - 1), `
			TMP(xs, j) += TMP(xr, j) * TMP(factor, j);
			TMP(ys, j) += TMP(yr, j) * TMP(factor, j);
			')
		}

		realtype xsum = 0, ysum = 0;

		for(int i = nnice; i < nsources; ++i)
		{
			const realtype xr = xt - _xsrc[i];
			const realtype yr = yt - _ysrc[i];
			const realtype factor = _vsrc[i] / (xr * xr + yr * yr + EPS);

			xsum += xr * factor;
			ysum += yr * factor;
		}

		LUNROLL(i, 0, eval(NACC - 1), `
		xsum += TMP(xs, i);
		ysum += TMP(ys, i);')

		*xresult = xsum;
		*yresult = ysum;
	}

divert(ifelse(TUNED4AVXDP, 1, -1, 0))
    	typedef realtype V4 __attribute__ ((vector_size (sizeof(realtype) * 4)));

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

			LUNROLL(n, 2, eval(5), `
			const realtype TMP(rinvz, n) = TMP(rinvz, eval(n - 1)) * rinvz_1 - TMP(iinvz, eval(n - 1)) * iinvz_1;
			const realtype TMP(iinvz, n) = TMP(rinvz, eval(n - 1)) * iinvz_1 + TMP(iinvz, eval(n - 1)) * rinvz_1;')dnl

			const V4 rz4 = { TMP(rinvz, 4), TMP(rinvz, 4), TMP(rinvz, 4), TMP(rinvz, 4) };
			const V4 iz4 = { TMP(iinvz, 4), TMP(iinvz, 4), TMP(iinvz, 4), TMP(iinvz, 4) };

			const V4 rbase_0 = { TMP(rinvz, 2), TMP(rinvz, 3), TMP(rinvz, 4), TMP(rinvz, 5) };
			const V4 ibase_0 = { TMP(iinvz, 2), TMP(iinvz, 3), TMP(iinvz, 4), TMP(iinvz, 5) };

			const V4 TMP(k4, 0) = {1, 2, 3, 4};dnl

			LUNROLL(j, 0, eval(ORDER/4 - 1), `
				const V4 TMP(rbase, eval(j + 1)) = TMP(rbase, j) * rz4 - TMP(ibase, j) * iz4;
				const V4 TMP(ibase, eval(j + 1)) = TMP(rbase, j) * iz4 + TMP(ibase, j) * rz4;

				const V4 TMP(rxp4, j) = { rxp[eval(4 * j)], rxp[eval(4 * j + 1)], rxp[eval(4 * j + 2)], rxp[eval(4 * j + 3)] };
				const V4 TMP(ixp4, j) = { ixp[eval(4 * j)], ixp[eval(4 * j + 1)], ixp[eval(4 * j + 2)], ixp[eval(4 * j + 3)] };

				const V4 TMP(rsum, j) = (TMP(rxp4, j) * TMP(rbase, j) - TMP(ixp4, j) * TMP(ibase, j)) * TMP(k4, j) ifelse(j, 0, ,+ TMP(rsum, eval(j - 1)));
				const V4 TMP(isum, j) = (TMP(rxp4, j) * TMP(ibase, j) + TMP(ixp4, j) * TMP(rbase, j)) * TMP(k4, j) ifelse(j, 0, ,+ TMP(isum, eval(j - 1)));

				const V4 TMP(k4, eval(j+1)) = TMP(k4, j) + 4;
			')dnl

			*xresult = mass * rinvz_1 - (TMP(rsum, eval(ORDER/4 - 1))[0] + TMP(rsum, eval(ORDER/4 - 1))[1] + TMP(rsum, eval(ORDER/4 - 1))[2] + TMP(rsum, eval(ORDER/4 - 1))[3]);
			*yresult = -(mass * iinvz_1 - (TMP(isum, eval(ORDER/4 - 1))[0] + TMP(isum, eval(ORDER/4 - 1))[1] + TMP(isum, eval(ORDER/4 - 1))[2] + TMP(isum, eval(ORDER/4 - 1))[3]));
		}
divert(0)		
divert(ifelse(TUNED4AVXDP, 1, 0, -1))
#include "immintrin.h"
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

			LUNROLL(n, 2, eval(5), `
			const realtype TMP(rinvz, n) = TMP(rinvz, eval(n - 1)) * rinvz_1 - TMP(iinvz, eval(n - 1)) * iinvz_1;
			const realtype TMP(iinvz, n) = TMP(rinvz, eval(n - 1)) * iinvz_1 + TMP(iinvz, eval(n - 1)) * rinvz_1;')dnl

			const __m256d rz4 = _mm256_set1_pd(TMP(rinvz, 4));
			const __m256d iz4 = _mm256_set1_pd(TMP(iinvz, 4));

			const __m256d rbase_0 = _mm256_set_pd(TMP(rinvz, 5), TMP(rinvz, 4), TMP(rinvz, 3), TMP(rinvz, 2));
			const __m256d ibase_0 = _mm256_set_pd(TMP(iinvz, 5), TMP(iinvz, 4), TMP(iinvz, 3), TMP(iinvz, 2));

			const __m256d TMP(k4, 0) = _mm256_set_pd(4, 3, 2, 1);dnl

			LUNROLL(j, 0, eval(ORDER/4 - 1), `
				const __m256d TMP(rbase, eval(j + 1)) = TMP(rbase, j) * rz4 - TMP(ibase, j) * iz4;
				const __m256d TMP(ibase, eval(j + 1)) = TMP(rbase, j) * iz4 + TMP(ibase, j) * rz4;

				const __m256d TMP(rxp4, j) = _mm256_loadu_pd(rxp + eval(4 * j));
				const __m256d TMP(ixp4, j) = _mm256_loadu_pd(ixp + eval(4 * j));

				const __m256d TMP(rsum, j) = (TMP(rxp4, j) * TMP(rbase, j) - TMP(ixp4, j) * TMP(ibase, j)) * TMP(k4, j) ifelse(j, 0, ,+ TMP(rsum, eval(j - 1)));
				const __m256d TMP(isum, j) = (TMP(rxp4, j) * TMP(ibase, j) + TMP(ixp4, j) * TMP(rbase, j)) * TMP(k4, j) ifelse(j, 0, ,+ TMP(isum, eval(j - 1)));

				const __m256d TMP(k4, eval(j+1)) = TMP(k4, j) + 4;
			')dnl

			*xresult = mass * rinvz_1 - (TMP(rsum, eval(ORDER/4 - 1))[0] + TMP(rsum, eval(ORDER/4 - 1))[1] + TMP(rsum, eval(ORDER/4 - 1))[2] + TMP(rsum, eval(ORDER/4 - 1))[3]);
			*yresult = -(mass * iinvz_1 - (TMP(isum, eval(ORDER/4 - 1))[0] + TMP(isum, eval(ORDER/4 - 1))[1] + TMP(isum, eval(ORDER/4 - 1))[2] + TMP(isum, eval(ORDER/4 - 1))[3]));
		}
divert(0)

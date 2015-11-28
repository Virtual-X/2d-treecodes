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
define(NACC, 4)

#define EPS (10 * __DBL_EPSILON__)
#define MAX(a,b) (((a)>(b))?(a):(b))
divert(ifelse(TUNED4AVXDP, 1, -1, 0))
	typedef realtype V4 __attribute__ ((vector_size (sizeof(realtype) * 4)));
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

		const V4 xt4 = {xt, xt, xt, xt};
		const V4 yt4 = {yt, yt, yt, yt};

		V4 LUNROLL(`i', 0, NACC, `ifelse(i, 0,, `,')
		TMP(xs,i) = {0, 0, 0, 0},
		TMP(ys,i) = {0, 0, 0, 0}') ;

		const int nnice = eval(4 * NACC) * (nsources / eval(4 * NACC));

		for(int i = 0; i < nnice; i += eval(4 * NACC))
		{
			const realtype * __restrict__ const xsrc = _xsrc + i;
			const realtype * __restrict__ const ysrc = _ysrc + i;
			const realtype * __restrict__ const vsrc = _vsrc + i;

			LUNROLL(j, 0, eval(NACC - 1), `
			pushdef(base, `eval(4 * j)')
			const V4 TMP(xsrc, j) = {xsrc[base], xsrc[base + 1], xsrc[base + 2], xsrc[base + 3]};
			const V4 TMP(ysrc, j) = {ysrc[base], ysrc[base + 1], ysrc[base + 2], ysrc[base + 3]};
			const V4 TMP(vsrc, j) = {vsrc[base], vsrc[base + 1], vsrc[base + 2], vsrc[base + 3]};
			popdef(base)

			const V4 TMP(xr, j) = xt4 - TMP(xsrc, j);
			const V4 TMP(yr, j) = yt4 - TMP(ysrc, j);

			const V4 TMP(factor, j) = TMP(vsrc, j) / (TMP(xr, j) * TMP(xr, j) + TMP(yr, j) * TMP(yr, j) + EPS) ;')

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

		V4 xsum4 = {0, 0, 0, 0}, ysum4 =  {0, 0, 0, 0};
		LUNROLL(i, 0, eval(NACC - 1), `
		xsum4 += TMP(xs, i);
		ysum4 += TMP(ys, i);')

		*xresult = xsum + xsum4[0] + xsum4[1] + xsum4[2] + xsum4[3];
		*yresult = ysum + ysum4[0] + ysum4[1] + ysum4[2] + ysum4[3];
		__asm__("L_END_FORCE_P2P:");
	}
divert(0)
divert(ifelse(TUNED4AVXDP, 1, 0, -1))
#include "immintrin.h"
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

		const __m256d xt4 = _mm256_set1_pd(xt);
		const __m256d yt4 = _mm256_set1_pd(yt);

		__m256d LUNROLL(`i', 0, NACC, `ifelse(i, 0,, `,')
		TMP(xs,i) =_mm256_setzero_pd(),
		TMP(ys,i) =_mm256_setzero_pd()') ;

		const int nnice = eval(4 * NACC) * (nsources / eval(4 * NACC));

		for(int i = 0; i < nnice; i += eval(4 * NACC))
		{
			const realtype * __restrict__ const xsrc = _xsrc + i;
			const realtype * __restrict__ const ysrc = _ysrc + i;
			const realtype * __restrict__ const vsrc = _vsrc + i;

			LUNROLL(j, 0, eval(NACC - 1), `
			pushdef(base, `eval(4 * j)')
			const __m256d TMP(xsrc, j) = _mm256_loadu_pd(xsrc + base);
			const __m256d TMP(ysrc, j) = _mm256_loadu_pd(ysrc + base);
			const __m256d TMP(vsrc, j) = _mm256_loadu_pd(vsrc + base);
			popdef(base)

			const __m256d TMP(xr, j) = xt4 - TMP(xsrc, j);
			const __m256d TMP(yr, j) = yt4 - TMP(ysrc, j);

			const __m256d TMP(factor, j) = TMP(vsrc, j) / (TMP(xr, j) * TMP(xr, j) + TMP(yr, j) * TMP(yr, j) + EPS) ;')

			LUNROLL(j, 0, eval(NACC - 1), `
			TMP(xs, j) += TMP(xr, j) * TMP(factor, j);
			TMP(ys, j) += TMP(yr, j) * TMP(factor, j);
			')
		}

		__m256d xsum4 = {0, 0, 0, 0}, ysum4 =  {0, 0, 0, 0};
		LUNROLL(i, 0, eval(NACC - 1), `
		xsum4 += TMP(xs, i);
		ysum4 += TMP(ys, i);')

		realtype xsum = 0, ysum = 0;

		for(int i = nnice; i < nsources; ++i)
		{
			const realtype xr = xt - _xsrc[i];
			const realtype yr = yt - _ysrc[i];
			const realtype factor = _vsrc[i] / (xr * xr + yr * yr + EPS);

			xsum += xr * factor;
			ysum += yr * factor;
		}

		*xresult = xsum + xsum4[0] + xsum4[1] + xsum4[2] + xsum4[3];
		*yresult = ysum + ysum4[0] + ysum4[1] + ysum4[2] + ysum4[3];
		__asm__("L_END_FORCE_P2P:");
	}
divert(0)

divert(ifelse(TUNED4AVXDP, 1, -1, 0))


	void force_e2p(const realtype mass,
		const realtype rz,
		const realtype iz,
		const realtype * __restrict__ const rxp,
		const realtype * __restrict__ const ixp,
		realtype * const xresult,
		realtype * const yresult)
		{
			__asm__("L_START_FORCE_E2P:");

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

			__asm__("L_END_FORCE_E2P:");
		}
divert(0)
divert(ifelse(TUNED4AVXDP, 1, 0, -1))
	void force_e2p(const realtype mass,
		const realtype rz,
		const realtype iz,
		const realtype * __restrict__ const rxp,
		const realtype * __restrict__ const ixp,
		realtype * const xresult,
		realtype * const yresult)
		{
			__asm__("L_START_FORCE_E2P:");
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

				const __m256d TMP(k4, eval(j+1)) = TMP(k4, j) + _mm256_set1_pd(4);
			')dnl

			*xresult = mass * rinvz_1 - (TMP(rsum, eval(ORDER/4 - 1))[0] + TMP(rsum, eval(ORDER/4 - 1))[1] + TMP(rsum, eval(ORDER/4 - 1))[2] + TMP(rsum, eval(ORDER/4 - 1))[3]);
			*yresult = -(mass * iinvz_1 - (TMP(isum, eval(ORDER/4 - 1))[0] + TMP(isum, eval(ORDER/4 - 1))[1] + TMP(isum, eval(ORDER/4 - 1))[2] + TMP(isum, eval(ORDER/4 - 1))[3]));

			__asm__("L_END_FORCE_E2P:");
		}
divert(0)

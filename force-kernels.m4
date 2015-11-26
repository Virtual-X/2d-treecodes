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

	void force_e2p_old(const realtype mass,
		const realtype rz,
		const realtype iz,
		const realtype * __restrict__ const rxp,
		const realtype * __restrict__ const ixp,
		realtype * const xresult,
		realtype * const yresult)
		{
			const realtype r2 = rz * rz + iz * iz;

			const realtype rinvz = rz / r2;
			const realtype iinvz = -iz / r2;

			realtype rinvz4 = rinvz;
			realtype iinvz4 = iinvz;

			realtype xs = mass * rinvz;
			realtype ys = mass * iinvz;

			const realtype rprod_0 = rinvz * rinvz - iinvz * iinvz;
			const realtype iprod_0 = 2 * rinvz * iinvz;

			xs -= rprod_0 * rxp[0] - iprod_0 * ixp[0];
			ys -= rprod_0 * ixp[0] + iprod_0 * rxp[0];

			LUNROLL(n, 1, eval(ORDER - 1), `
			const realtype TMP(rprod, n) = rinvz * TMP(rprod, eval(n - 1)) - iinvz * TMP(iprod, eval(n - 1));
			const realtype TMP(iprod, n) = iinvz * TMP(rprod, eval(n - 1)) + rinvz * TMP(iprod, eval(n - 1));

			xs -= (n + 1) * (TMP(rprod, n) * rxp[n] - TMP(iprod, n) * ixp[n]);
			ys -= (n + 1) * (TMP(rprod, n) * ixp[n] + TMP(iprod, n) * rxp[n]);
			')

			*xresult = xs;
			*yresult = -ys;
		}

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
				const realtype TMP(iinvz, n) = TMP(rinvz, eval(n - 1)) * iinvz_1 + TMP(iinvz, eval(n - 1)) * rinvz_1;')

				V4 rz4 = { TMP(rinvz, 4), TMP(rinvz, 4), TMP(rinvz, 4), TMP(rinvz, 4) };
				V4 iz4 = { TMP(iinvz, 4), TMP(iinvz, 4), TMP(iinvz, 4), TMP(iinvz, 4) };
				V4 rbase = { TMP(rinvz, 2), TMP(rinvz, 3), TMP(rinvz, 4), TMP(rinvz, 5) };
				V4 ibase = { TMP(iinvz, 2), TMP(iinvz, 3), TMP(iinvz, 4), TMP(iinvz, 5) };

				V4 rsum = {0, 0, 0, 0};
				V4 isum = {0, 0, 0, 0};
				V4 k4 = {1, 2, 3, 4};

				LUNROLL(j, 0, eval(ORDER/4 - 1), `
				{
					V4 tmp0 = rbase * rz4 - ibase * iz4;
					V4 tmp1 = rbase * iz4 + ibase * rz4;

					V4 rxp4 = { rxp[eval(4 * j)], rxp[eval(4 * j + 1)], rxp[eval(4 * j + 2)], rxp[eval(4 * j + 3)] };
					V4 ixp4 = { ixp[eval(4 * j)], ixp[eval(4 * j + 1)], ixp[eval(4 * j + 2)], ixp[eval(4 * j + 3)] };

					rsum -= (rxp4 * rbase - ixp4 * ibase) * k4;
					isum -= (rxp4 * ibase + ixp4 * rbase) * k4;

					rbase = tmp0;
					ibase = tmp1;

					k4 += 4;
				}
				')

				*xresult = mass * rinvz_1 + rsum[0] + rsum[1] + rsum[2] + rsum[3];
				*yresult = -(mass * iinvz_1 + isum[0] + isum[1] + isum[2] + isum[3]);
			}

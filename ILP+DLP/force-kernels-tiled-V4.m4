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

	typedef realtype V4 __attribute__ ((vector_size (sizeof(realtype) * 4)));

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
		const V4 xt = { _xt, _xt + h, _xt + 2 * h, _xt + 3 * h};
		const V4 eps = {EPS, EPS, EPS, EPS };

		LUNROLL(iy, 0, 3, `
			    const realtype TMP(syt, iy) = _yt + iy * h;
			    const V4 TMP(yt,iy) = { TMP(syt, iy), TMP(syt, iy), TMP(syt, iy), TMP(syt, iy)}; ')

		LUNROLL(iy, 0, 3, `V4 TMP(xs, iy) = {0, 0, 0, 0};')
		LUNROLL(iy, 0, 3, `V4 TMP(ys, iy) = {0, 0, 0, 0};')

		const int nnice = 4 * (nsources / 4);

		for(int j = 0; j < nnice; j += 4)
		{
			LUNROLL(pass, 0, 3, `
			{
				const realtype xcurr = xsrc[j + pass];
				const realtype ycurr = ysrc[j + pass];
				const realtype vcurr = vsrc[j + pass];

				const V4 xr = xt - xcurr;
				const V4 xr2 = xr * xr;

				LUNROLL(iy, 0, 3, `
				    const V4 TMP(yr, iy) = TMP(yt, iy) - ycurr;
				    const V4 TMP(factor, iy) = vcurr / (xr2 + TMP(yr, iy) * TMP(yr, iy) + eps);')

				LUNROLL(iy, 0, 3, `
				    TMP(xs, iy) += xr * TMP(factor, iy);
				    TMP(ys, iy) += TMP(yr, iy) * TMP(factor, iy);')
			}')
		}

		for(int j = nnice; j < nsources; ++j)
		{
			const realtype xcurr = xsrc[j];
			const realtype ycurr = ysrc[j];
			const realtype vcurr = vsrc[j];

			const V4 xr = xt - xcurr;
			const V4 xr2 = xr * xr;

			LUNROLL(iy, 0, 3, `
				    const V4 TMP(yr, iy) = TMP(yt, iy) - ycurr;
				    const V4 TMP(factor, iy) = vcurr / (xr2 + TMP(yr, iy) * TMP(yr, iy) + eps);')

			LUNROLL(iy, 0, 3, `
				    TMP(xs, iy) += xr * TMP(factor, iy);
				    TMP(ys, iy) += TMP(yr, iy) * TMP(factor, iy);')
		}

		LUNROLL(iy, 0, 3, `
			    LUNROLL(ix, 0, 3, `xresult[stride * iy + ix] += TMP(xs, iy)[ix];')
    			    LUNROLL(ix, 0, 3, `yresult[stride * iy + ix] += TMP(ys, iy)[ix];')')
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
			LUNROLL(ix, 0, 3, `xresult[stride * iy + ix] += TMP(xs, iy)[ix];')
 			LUNROLL(ix, 0, 3, `yresult[stride * iy + ix] -= TMP(ys, iy)[ix];')
			')

			__asm__("L_END_FORCE_E2P_TILED:");
		}

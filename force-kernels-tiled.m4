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
			const V4 xdispl = {0, h, 2 * h, 3 * h};
			const V4 rz0 = {scalar_rz, scalar_rz, scalar_rz, scalar_rz};

			const V4 rz = rz0 + xdispl;
			
			LUNROLL(iy, 0, 3, `
			{
			const V4 iz = {scalar_iz + iy * h, scalar_iz + iy * h, scalar_iz + iy * h, scalar_iz + iy * h};
			
			const V4 r2 = rz * rz + iz * iz;

			const V4 rinvz = rz / r2;
			const V4 iinvz = -iz / r2;
			
			V4 xs = mass * rinvz;
 			V4 ys = mass * iinvz;
 
 			const V4 rprod_0 = rinvz * rinvz - iinvz * iinvz;
 			const V4 iprod_0 = 2 * rinvz * iinvz;
 
 			xs -= rprod_0 * rxp[0] - iprod_0 * ixp[0];
 			ys -= rprod_0 * ixp[0] + iprod_0 * rxp[0];
 
 			LUNROLL(n, 1, eval(ORDER - 1), `
 			const V4 TMP(rprod, n) = rinvz * TMP(rprod, eval(n - 1)) - iinvz * TMP(iprod, eval(n - 1));
 			const V4 TMP(iprod, n) = iinvz * TMP(rprod, eval(n - 1)) + rinvz * TMP(iprod, eval(n - 1));
 
 			xs -= (n + 1) * (TMP(rprod, n) * rxp[n] - TMP(iprod, n) * ixp[n]);
 			ys -= (n + 1) * (TMP(rprod, n) * ixp[n] + TMP(iprod, n) * rxp[n]);
 			')
 
 			LUNROLL(i, 0, 3, `xresult[eval(4 * iy) + i] += xs[i];')
 			LUNROLL(i, 0, 3, `yresult[eval(4 * iy) + i] -= ys[i];')
			}
			')

			__asm__("L_END_FORCE_E2P_TILED:");
		}

	

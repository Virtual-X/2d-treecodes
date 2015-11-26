/*
 *  treecode-kernels.c
 *  Part of MRAG/2d-treecode-potential
 *
 *  Created and authored by Diego Rossinelli on 2015-09-25.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

typedef REAL realtype;

#ifdef __cplusplus
extern "C"
{
#endif

    realtype treecode_p2p(const realtype * __restrict__ const xsources,
			  const realtype * __restrict__ const ysources,
			  const realtype * __restrict__ const sources,
			  const int nsources,
			  const realtype _xt,
			  const realtype _yt);

    realtype treecode_e2p(const realtype mass,
			  const realtype rx,
			  const realtype ry,
			  const realtype * __restrict__ const rxp,
			  const realtype * __restrict__ const ixp);
#ifdef __cplusplus
}
#endif

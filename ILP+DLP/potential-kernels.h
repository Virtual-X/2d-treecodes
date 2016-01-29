/*
 *  potential-kernels.h
 *  Part of MRAG/2d-treecode-potential
 *
 *  Created and authored by Diego Rossinelli on 2015-11-25.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

__device__ realtype potential_p2p(const realtype * __restrict__ const xsources,
				  const realtype * __restrict__ const ysources,
				  const realtype * __restrict__ const sources,
				  const int nsources,
				  const realtype _xt,
				  const realtype _yt);

__device__ realtype potential_e2p(const realtype mass,
				  const realtype rx,
				  const realtype ry,
				  const realtype * __restrict__ const rxp,
				  const realtype * __restrict__ const ixp,
				  realtype * const scratch);


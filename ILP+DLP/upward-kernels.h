/*
 *  upward-kernels.h
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

#define _CONCATENATE_BODY(a, b) a ## b
#define _CONCATENATE(a, b) _CONCATENATE_BODY(a, b)
#define P2E_KERNEL _CONCATENATE(reference_upward_p2e_order, ORDER)
#define E2E_KERNEL _CONCATENATE(reference_upward_e2e_order, ORDER)

#ifdef __cplusplus
extern "C"
{
#endif
    void P2E_KERNEL(const realtype * __restrict__ const xsources,
		    const realtype * __restrict__ const ysources,
		    const realtype * __restrict__ const sources,
		    const int nsources,
		    const realtype x0,
		    const realtype y0,
		    const realtype h,
		    realtype * const mass,
		    realtype * const weight,
		    realtype * const xsum,
		    realtype * const ysum,
		    realtype * const radius,
		    realtype * __restrict__ const rexpansions,
		    realtype * __restrict__ const iexpansions);

    typedef realtype V4 __attribute__ ((vector_size (sizeof(realtype) * 4)));

    void E2E_KERNEL(const realtype * const x0s,
		    const realtype * const y0s,
		    const realtype * const masses,
		    const realtype * __restrict__ const * vrexpansions,
		    const realtype * __restrict__ const * viexpansions,
		    realtype * __restrict__ const rdstxp,
		    realtype * __restrict__ const idstxp);
#ifdef __cplusplus
}
#endif


#define P2E_KERNEL_CUDA _CONCATENATE(upward_p2e_order, ORDER)

__device__ void P2E_KERNEL_CUDA(const realtype xcom,
		     const realtype ycom,
		     const realtype * __restrict__ const xsources,
		     const realtype * __restrict__ const ysources,
		     const realtype * __restrict__ const vsources,
		     const int nsources,
		     realtype * __restrict__ const rexpansions,
		     realtype * __restrict__ const iexpansions);

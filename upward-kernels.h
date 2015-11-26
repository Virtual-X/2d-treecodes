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

typedef REAL realtype;

#define _CONCATENATE_BODY(a, b) a ## b
#define _CONCATENATE(a, b) _CONCATENATE_BODY(a, b)
#define P2E_KERNEL _CONCATENATE(treecode_p2e_order, ORDER)
#define E2E_KERNEL _CONCATENATE(treecode_e2e_order, ORDER)

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

    void E2E_KERNEL(const V4 srcmass, const V4 rx, const V4 ry,
		    const V4 * __restrict__ const rsrcxp,
		    const V4 * __restrict__ const isrcxp,
		    realtype * __restrict__ const rdstxp,
		    realtype * __restrict__ const idstxp);
#ifdef __cplusplus
}
#endif

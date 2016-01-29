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

__device__ void upward_p2e(
    const realtype xcom,
    const realtype ycom,
    const realtype * __restrict__ const xsources,
    const realtype * __restrict__ const ysources,
    const realtype * __restrict__ const vsources,
    const int nsources,
    realtype * __restrict__ const rexpansions,
    realtype * __restrict__ const iexpansions);

__device__ void upward_e2e(
    const realtype  x0,
    const realtype  y0,
    const realtype  mass,
    const realtype * const rsrcxp,
    const realtype * const isrcxp,
    realtype * __restrict__ const rdstxp,
    realtype * __restrict__ const idstxp);

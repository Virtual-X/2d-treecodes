/*
 *  force-downward-kernels.h
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

__device__ void force_downward_e2l(
    const realtype x0,
    const realtype y0,
    const realtype mass,
    const realtype * const rxp,
    const realtype * const ixp,
    realtype * const rlocal,
    realtype * const ilocal);

__device__ void force_downward_l2p(
    const realtype rz_1,
    const realtype iz_1,
    const realtype * const rlocal,
    const realtype * const ilocal,
    realtype& xresult,
    realtype& yresult);

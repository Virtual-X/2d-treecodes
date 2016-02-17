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

extern "C"
__attribute__ ((visibility ("hidden")))
void upward_p2e(
    const  realtype xsources[],
    const  realtype ysources[],
    const  realtype vsources[],
    const  int nsources,
    const  realtype x0,
    const  realtype y0,
    const  realtype h,
    realtype *  mass,
    realtype *  weight,
    realtype *  xsum,
    realtype *  ysum,
    realtype *  radius,
    realtype rexpansions[],
    realtype iexpansions[]);


extern "C"
__attribute__ ((visibility ("hidden")))
void upward_e2e(
    const realtype x0s[],
    const realtype y0s[],
    const realtype masses[],
    const realtype *  vrxps[],
    const realtype *  vixps[],
    realtype rdstxp[],
    realtype idstxp[]);

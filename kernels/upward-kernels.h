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
{
    __attribute__ ((visibility ("hidden"))) 
    void minmax_vec(
	const  realtype xsrc[],
	const  realtype ysrc[],
	const  int nsources,
	realtype xmin_xmax_ymin_ymax[]);
    
    __attribute__ ((visibility ("hidden"))) 
    int lower_bound_vec(int s, int e, const  int val, const int keys[]);
	
    __attribute__ ((visibility ("hidden"))) 
    int upper_bound_vec(int s, int e, const  int val, const int keys[]);

    __attribute__ ((visibility ("hidden")))    
    void upward_setup(
	const realtype xsources[],
	const realtype ysources[],
	const realtype vsources[],
	const int nsources,
	realtype * const mass,
	realtype * const w,
	realtype * const wx,
	realtype * const wy,
	realtype * const radius);
    
    __attribute__ ((visibility ("hidden")))
    void upward_p2e(
	const  realtype xsources[],
	const  realtype ysources[],
	const  realtype vsources[],
	const int nsources,
	const realtype xcom,
	const realtype ycom,
	realtype * __restrict__ const rexpansions,
	realtype * __restrict__ const iexpansions);

    __attribute__ ((visibility ("hidden")))
    void upward_e2e(
	const realtype x0s[],
	const realtype y0s[],
	const realtype masses[],
	const realtype *  vrxps[],
	const realtype *  vixps[],
	realtype rdstxp[],
	realtype idstxp[]);
}

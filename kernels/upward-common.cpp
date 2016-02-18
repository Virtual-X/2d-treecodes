/*
 *  upward-common.ispc
 *  Part of 2d-treecodes
 *
 *  Created and authored by Diego Rossinelli on 2015-11-25.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cmath>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

extern "C" void upward_setup(
    const  realtype xsources[],
    const  realtype ysources[],
    const  realtype vsources[],
    const  int nsources,
     realtype *  mass,
     realtype *  weight,
     realtype *  xsum,
     realtype *  ysum,
     realtype *  radius)
{
    realtype msum = 0, wsum = 0, wxsum = 0, wysum = 0;

    for(int i = 0; i < nsources; ++i)
    {
	const realtype x = xsources[i];
	const realtype y = ysources[i];
	const realtype m = vsources[i];
	const realtype w = fabs(m);

	msum += m;
	wsum += w;
	wxsum += x * w;
	wysum += y * w;
    }

    *mass = msum;
    *weight = wsum;
    *xsum = wxsum;
    *ysum = wysum;

    const realtype xcom = wsum ? *xsum / *weight : 0;
    const realtype ycom = wsum ? *ysum / *weight : 0;

    realtype r2 = 0;
    for(int i = 0; i < nsources; ++i)
    {
	const realtype xr = xsources[i] - xcom;
	const realtype yr = ysources[i] - ycom;

	r2 = fmax(r2, xr * xr + yr * yr);
    }

    *radius = sqrt(r2);
}

extern "C"  int lower_bound_vec(
     int s,
     int e,
    const  int val,
     const int keys[])
{
     int c = e - s;

    if (keys[s] >= val)
	return s;

    if (keys[e - 1] < val)
	return e;

    while (c)
    {
	const int s0 = s, e0 = e;
	
	const float h = (e - s) * 1.f / 8;

	for(int programIndex = 0; programIndex < 8; ++programIndex)
	{
	    //int candidate_s = s0, candidate_e = e0;
	    const int i = MIN(e0 - 1, (int)(s0 + programIndex * h + 0.499999f));

	    const bool isless = keys[i] < val;
	    const int candidate_s = isless ? i : s0;
	    const int candidate_e = isless ? e0 : i;

	    s = MAX(s, candidate_s);
	    e = MIN(e, candidate_e);
	}
	
	c = MIN(c / 8, e - s);
    }

    return s + 1;
}

extern "C"  int upper_bound_vec
(
     int s,
     int e,
    const  int val,
     const int keys[])
{
    int c = e - s;

    if (keys[s] > val)
	return s;

    if (keys[e - 1] <= val)
	return e;

    while (c)
    {
	const int s0 = s, e0 = e;

	const float h = (e - s) * 1.f / 8;

	for(int programIndex = 0; programIndex < 8; ++programIndex)
	{
	    //int candidate_s = s0, candidate_e = e0;
	    const int i = MIN(e0 - 1, (int)(s0 + programIndex * h + 0.499999f));
	    
	    const bool isless = keys[i] <= val;
	    const int candidate_s = isless ? i : s0;
	    const int candidate_e = isless ? e0 : i;

	    s = MAX(s, candidate_s);
	    e = MIN(e, candidate_e);
	}
	
	c = MIN(c / 8, e - s);
    }

    return s + 1;
}

extern "C" void minmax_vec(
    const  realtype xsrc[],
    const  realtype ysrc[],
    const  int nsources,
     realtype xmin_xmax_ymin_ymax[])
{
    realtype lxmi = 1e13, lymi = 1e13, lxma = 0, lyma = 0;

    for(int i = 0; i < nsources; ++i)
    {
	const realtype xval = xsrc[i];
	const realtype yval = ysrc[i];

	lxmi = fmin(lxmi, xval);
	lxma = fmax(lxma, xval);

	lymi = fmin(lymi, yval);
	lyma = fmax(lyma, yval);
    }

    xmin_xmax_ymin_ymax[0] = lxmi;
    xmin_xmax_ymin_ymax[1] = lxma;
    xmin_xmax_ymin_ymax[2] = lymi;
    xmin_xmax_ymin_ymax[3] = lyma;
}


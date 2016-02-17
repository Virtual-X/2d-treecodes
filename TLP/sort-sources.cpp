/*
 *  sort-sources.cpp
 *  Part of MRAG/2d-treecode-potential
 *
 *  Created and authored by Diego Rossinelli on 2015-11-25.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */
#include <cassert>
#include <cstdio>
#include <parallel/algorithm>

typedef REAL realtype;

extern "C"
void sort_sources(const realtype * const xsrc,
		  const realtype * const ysrc,
		  const realtype * const vsrc,
		  const int nsrc,
		  int * const keysorted,
		  realtype * const xsorted,
		  realtype * const ysorted,
		  realtype * const vsorted,
		  realtype * const output_xmin,
		  realtype * const output_ymin,
		  realtype * const output_extent)
{
    realtype ext0, ext1, xmin, ymin;

    {
	const int nthreads = omp_get_max_threads();
	
	realtype xpartials[2][nthreads], ypartials[2][nthreads];

#pragma omp parallel
	{
	    realtype lxmi = HUGE_VAL, lymi = HUGE_VAL, lxma = 0, lyma = 0;

#pragma omp for
	    for(int i = 0; i < nsrc; ++i)
	    {
		const realtype xval = xsrc[i];
		const realtype yval = ysrc[i];

		lxmi = std::min(lxmi, xval);
		lxma = std::max(lxma, xval);

		lymi = std::min(lymi, yval);
		lyma = std::max(lyma, yval);
	    }

	    const int tid = omp_get_thread_num();

	    xpartials[0][tid] = lxmi;
	    xpartials[1][tid] = lxma;
	    ypartials[0][tid] = lymi;
	    ypartials[1][tid] = lyma;
	}

	xmin = *std::min_element(xpartials[0], xpartials[0] + nthreads);
	ymin = *std::min_element(ypartials[0], ypartials[0] + nthreads);

	ext0 = (*std::max_element(xpartials[1], xpartials[1] + nthreads) - xmin);
	ext1 = (*std::max_element(ypartials[1], ypartials[1] + nthreads) - ymin);
    }

    const realtype eps = 10000 * std::numeric_limits<realtype>::epsilon();

    const realtype ext = std::max(ext0, ext1) * (1 + 2 * eps);
    xmin -= eps * ext;
    ymin -= eps * ext;

    std::pair<int, int> * kv = NULL;
    posix_memalign((void **)&kv, 32, sizeof(*kv) * nsrc);
    
#pragma omp parallel for
    for(int i = 0; i < nsrc; ++i)
    {
	int x = floor((xsrc[i] - xmin) / ext * (1 << LMAX));
	int y = floor((ysrc[i] - ymin) / ext * (1 << LMAX));

	assert(x >= 0 && y >= 0);
	assert(x < (1 << LMAX) && y < (1 << LMAX));

	x = (x | (x << 8)) & 0x00FF00FF;
	x = (x | (x << 4)) & 0x0F0F0F0F;
	x = (x | (x << 2)) & 0x33333333;
	x = (x | (x << 1)) & 0x55555555;

	y = (y | (y << 8)) & 0x00FF00FF;
	y = (y | (y << 4)) & 0x0F0F0F0F;
	y = (y | (y << 2)) & 0x33333333;
	y = (y | (y << 1)) & 0x55555555;

	const int key = x | (y << 1);

	kv[i].first = key;
	kv[i].second = i;
    }
    
    __gnu_parallel::sort(kv, kv + nsrc);

#pragma omp parallel for //num_threads(24)
    for(int i = 0; i < nsrc; ++i)
    {
	keysorted[i] = kv[i].first;

	const int entry = kv[i].second;
	assert(entry >= 0 && entry < nsrc);

	xsorted[i] = xsrc[entry];
	ysorted[i] = ysrc[entry];
	vsorted[i] = vsrc[entry];
    }
    
    free(kv);
     
    *output_xmin = xmin;
    *output_ymin = ymin;
    *output_extent = ext;
}

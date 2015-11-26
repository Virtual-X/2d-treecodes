/*
 *  treecode.cpp
 *  Part of MRAG/2d-treecode-potential
 *
 *  Created and authored by Diego Rossinelli on 2015-09-25.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cassert>
#include <cmath>

#include <parallel/algorithm>
#include <limits>

#include "upward-kernels.h"
#include "upward.h"

#define LEAF_MAXCOUNT 96
#define LMAX 15

namespace Tree
{
    const realtype eps = 10 * std::numeric_limits<realtype>::epsilon();

    realtype ext, xmin, ymin;

    int * keys = NULL;

    realtype *xdata = NULL, *ydata = NULL, *vdata = NULL;

    Node * root = NULL;

    void _build(Node ** _node, const int x, const int y, const int l, const int s, const int e, const int mask) 
{
    const double h = ext / (1 << l);
    const double x0 = xmin + h * x, y0 = ymin + h * y;

    assert(x < (1 << l) && y < (1 << l) && x >= 0 && y >= 0);

#ifndef NDEBUG
    for(int i = s; i < e; ++i)
	assert(xdata[i] >= x0 && xdata[i] < x0 + h && ydata[i] >= y0 && ydata[i] < y0 + h);
#endif

    Node * const node = new Node{x, y, l, s, e, e - s <= LEAF_MAXCOUNT || l + 1 > LMAX};
    *_node = node;

    if (node->leaf)
    {
	P2E_KERNEL(&xdata[s], &ydata[s], &vdata[s], e - s,
		   x0, y0, h, &node->mass, &node->w, &node->wx, &node->wy, &node->r,
		   node->expansions[0], node->expansions[1]);

	assert(node->r < 1.5 * h);
    }
    else
    {
	for(int c = 0; c < 4; ++c)
	{
	    const int shift = 2 * (LMAX - l - 1);

	    const int key1 = mask | (c << shift);
	    const int key2 = key1 + (1 << shift) - 1;

	    const size_t indexmin = std::lower_bound(keys + s, keys + e, key1) - keys;
	    const size_t indexsup = std::upper_bound(keys + s, keys + e, key2) - keys;

	    Node ** chd = &node->children[c];

#pragma omp task firstprivate(chd, c, x, y, l, indexmin, indexsup, key1) if (c < 3 && l < 8)
	    {
		_build(chd, (x << 1) + (c & 1), (y << 1) + (c >> 1), l + 1, indexmin, indexsup, key1);
	    }

	}

#pragma omp taskwait

	for(int c = 0; c < 4; ++c)
	{
	    Node * chd = node->children[c];
	    node->mass += chd->mass;
	    node->w += chd->w;
	    node->wx += chd->wx;
	    node->wy += chd->wy;

	    node->children[c] = chd;
	}

	realtype rcandidates[4];
	node->r = 0;
	for(int c = 0; c < 4; ++c)
	    node->r = std::max(node->r,
			  node->children[c]->r +
			  sqrt(pow(node->xcom() - node->children[c]->xcom(), 2) +
			       pow(node->ycom() - node->children[c]->ycom(), 2)));

	node->r = std::min(node->r, 1.4143 * h);

	assert(node->r < 1.5 * h);

#ifndef NDEBUG
	{
	    realtype r = 0;

	    for(int i = s; i < e; ++i)
		r = max(r, pow(xdata[i] - node->xcom(), (realtype)2) + pow(ydata[i] - node->ycom(), (realtype)2));

	    assert (sqrt(r) <= node->r);
	}
#endif

	V4 srcmass, rx, ry, chldexp[2][ORDER];
	for(int c = 0; c < 4; ++c)
	{
	    Node * chd = node->children[c];

	    srcmass[c] = chd->mass;
	    rx[c] = chd->xcom();
	    ry[c] = chd->ycom();

	    for(int i = 0; i < 2; ++i)
		for(int j = 0; j < ORDER; ++j)
		    chldexp[i][j][c] = chd->expansions[i][j];
	}

	rx -= node->xcom();
	ry -= node->ycom();

	E2E_KERNEL(srcmass, rx, ry, chldexp[0], chldexp[1], node->expansions[0], node->expansions[1]);
    }

#ifndef NDEBUG
    {
	for(int i = 0; i < ORDER; ++i)
	    assert(!::isnan((double)node->expansions[0][i]) && !::isnan(node->expansions[1][i]));

	assert(node->xcom() >= x0 && node->xcom() < x0 + h && node->ycom() >= y0 && node->ycom() < y0 + h || node->e - node->s == 0);
    }
#endif
}
}

void Tree::build(const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
	   const realtype * const xdst, const realtype * const ydst, const int ndst, realtype * const vdst)
{
    posix_memalign((void **)&keys, 32, sizeof(int) * nsrc);
    posix_memalign((void **)&xdata, 32, sizeof(*xdata) * nsrc);
    posix_memalign((void **)&ydata, 32, sizeof(*ydata) * nsrc);
    posix_memalign((void **)&vdata, 32, sizeof(*vdata) * nsrc);

    xmin = *__gnu_parallel::min_element(xsrc, xsrc + nsrc);
    ymin = *__gnu_parallel::min_element(ysrc, ysrc + nsrc);

    const realtype ext0 = (*__gnu_parallel::max_element(xsrc, xsrc + nsrc) - xmin);
    const realtype ext1 = (*__gnu_parallel::max_element(ysrc, ysrc + nsrc) - ymin);

    ext = std::max(ext0, ext1) * (1 + 2 * eps);
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

    Node * myroot;
    
#pragma omp parallel shared(myroot)
    {

#pragma omp for
	for(int i = 0; i < nsrc; ++i)
	{
	    keys[i] = kv[i].first;

	    const int entry = kv[i].second;
	    assert(entry >= 0 && entry < nsrc);

	    xdata[i] = xsrc[entry];
	    ydata[i] = ysrc[entry];
	    vdata[i] = vsrc[entry];
	}

#pragma omp single
	{
	    free(kv);
	}

#pragma omp single //with nowait it crashes
	{
	    _build(&myroot, 0, 0, 0, 0, nsrc, 0);
	}

#pragma omp single
	{
	    free(keys);
	    root = myroot;
	}

    }
    
}
    
void Tree::dispose()
{
    free(xdata);
    free(ydata);
    free(vdata);

    delete root;
}


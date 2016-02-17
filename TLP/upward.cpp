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
#include <utility>

#include "upward.h"
#include "upward-kernels.h"
#include "sort-sources.h"

 inline int fetch_and_add( int * variable, int value ) {
      asm volatile("lock; xaddl %%eax, %2;"
                   :"=a" (value)                  //Output
                   :"a" (value), "m" (*variable)  //Input
                   :"memory");
      return value;
  }

namespace Tree
{
    struct NodeHelper
    {
	int x, y, l, mask, parent, validchildren;
	realtype w, wx, wy;

	void setup(int x, int y, int l, int mask, int parent)
	    {
		this->x = x;
		this->y = y;
		this->l = l;
	
		this->mask = mask;
		this->parent = parent;
		this->validchildren = 0;
	    }
    };

    Node * nodes;

    NodeHelper * bufhelpers;
    realtype * expansions;
    int currnnodes, maxnodes;
    
    int LEAF_MAXCOUNT;

    realtype ext, xmin, ymin;

    int * keys = NULL;
    realtype *xdata = NULL, *ydata = NULL, *vdata = NULL;
    
    void _build(const int nodeid,
		const int x, const int y, const int l, const int s, const int e, const int mask,
		const int parentid)
    {
	const double h = ext / (1 << l);
	const double x0 = xmin + h * x, y0 = ymin + h * y;

	assert(x < (1 << l) && y < (1 << l) && x >= 0 && y >= 0);

#ifndef NDEBUG
	for(int i = s; i < e; ++i)
	    assert(xdata[i] >= x0 && xdata[i] < x0 + h && ydata[i] >= y0 && ydata[i] < y0 + h);
#endif

	Node * const node = nodes + nodeid;
	node->setup(s, e);
	
	NodeHelper * const helper = bufhelpers + nodeid;
	helper->setup(x, y, l, mask, parentid);

	const bool leaf = e - s <= LEAF_MAXCOUNT || l + 1 > LMAX;
	if (leaf)
	{
	    upward_p2e(xdata + s, ydata + s, vdata + s, e - s,
		       x0, y0, h, &node->mass, &helper->w, &helper->wx, &helper->wy, &node->r,
		       expansions + ORDER * (2 * nodeid + 0),
		       expansions + ORDER * (2 * nodeid + 1));

	    node->xcom = helper->w ? helper->wx / helper->w : 0;
	    node->ycom = helper->w ? helper->wy / helper->w : 0;
	    	    
	    assert(node->r < 1.5 * h);
	}
	else
	{
	    const int childbase = fetch_and_add(&currnnodes, 4);
	    assert(nodeid < childbase);
	    assert(childbase + 4 <= maxnodes);
	    	    
	    node->state.childbase = childbase;
	    
	    for(int c = 0; c < 4; ++c)
	    {
		const int shift = 2 * (LMAX - l - 1);

		const int key1 = mask | (c << shift);
		const int key2 = key1 + (1 << shift) - 1;
	
		const size_t indexmin = c == 0 ? s : std::lower_bound(keys + s, keys + e, key1) - keys;
		const size_t indexsup = c == 3 ? e : std::upper_bound(keys + s, keys + e, key2) - keys;
		
#pragma omp task firstprivate(childbase, c, x, y, l, indexmin, indexsup, key1) if (indexsup - indexmin > 5e3 && c < 3)
		{
		    _build(childbase + c, (x << 1) + (c & 1), (y << 1) + (c >> 1), l + 1, indexmin, indexsup, key1, nodeid);
		}

	    }
	    
//#pragma omp taskyield
#pragma omp taskwait

		helper->w = 0;
		helper->wx = 0;
		helper->wy = 0;
	    for(int c = 0; c < 4; ++c)
	    {
		node->mass += nodes[childbase + c].mass;

		NodeHelper * chh = bufhelpers + childbase + c;
		helper->w += chh->w;
		helper->wx += chh->wx;
		helper->wy += chh->wy;
	    }
	    
	    node->xcom = helper->w ? helper->wx / helper->w : 0;
	    node->ycom = helper->w ? helper->wy / helper->w : 0;
	    
	    realtype rcandidates[4];
	    node->r = 0;
	    for(int c = 0; c < 4; ++c)
		if (bufhelpers[childbase + c].w)
		    node->r = std::max(node->r,
				       nodes[childbase + c].r +
				       sqrt(pow(node->xcom - nodes[childbase + c].xcom, 2) +
					    pow(node->ycom - nodes[childbase + c].ycom, 2)));

	    node->r = std::min(node->r, 1.4143 * h);
	    assert(node->r < 1.5 * h);

#ifndef NDEBUG
	    {
		realtype r = 0;

		for(int i = s; i < e; ++i)
		    r = std::max(r, pow(xdata[i] - node->xcom, (realtype)2) +
				 pow(ydata[i] - node->ycom, (realtype)2));
		
		assert (sqrt(r) <= node->r);
	    }
#endif

	    {
		realtype srcmass[4], rx[4], ry[4];
		const realtype * chldrxp[4], *chldixp[4];

		for(int c = 0; c < 4; ++c)
		{
		    const int childid = childbase + c;
		    Node * chd = nodes + childid;

		    srcmass[c] = chd->mass;
		    rx[c] = chd->xcom - node->xcom;
		    ry[c] = chd->ycom - node->ycom;
		    chldrxp[c] = expansions + ORDER * (2 * childid + 0);
		    chldixp[c] = expansions + ORDER * (2 * childid + 1);
		}
		upward_e2e(rx, ry, srcmass, chldrxp, chldixp,
			   expansions + ORDER * (2 * nodeid + 0),
			   expansions + ORDER * (2 * nodeid + 1));
	    }
	}

#ifndef NDEBUG
	{
	    assert(node->xcom >= x0 && node->xcom < x0 + h && node->ycom >= y0 && node->ycom < y0 + h || node->e - node->s == 0);
	}
#endif
    }
}

void Tree::build(const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc,
		 const int nsrc,
		 const int LEAF_MAXCOUNT)
{
  const double t0 = omp_get_wtime();
  const int isdynamic = omp_get_dynamic();
  const int maxthreads = omp_get_max_threads();

  Tree::LEAF_MAXCOUNT = LEAF_MAXCOUNT;

  posix_memalign((void **)&keys, 32, sizeof(int) * nsrc);
  posix_memalign((void **)&xdata, 32, sizeof(*xdata) * nsrc);
  posix_memalign((void **)&ydata, 32, sizeof(*ydata) * nsrc);
  posix_memalign((void **)&vdata, 32, sizeof(*vdata) * nsrc);

  const double t1 = omp_get_wtime();

  sort_sources(xsrc, ysrc, vsrc, nsrc, keys, xdata, ydata, vdata, &xmin, &ymin, &ext);
  
  const double t5 = omp_get_wtime();

  currnnodes = 1;
  maxnodes = 8e4;
  posix_memalign((void **)&nodes, 32, sizeof(Node) * maxnodes);
  posix_memalign((void **)&bufhelpers, 32, sizeof(NodeHelper) * maxnodes);
  posix_memalign((void **)&expansions, 32, sizeof(realtype) * 2 * ORDER * maxnodes);
  
#pragma omp parallel
  {
#pragma omp single
      { _build(0, 0, 0, 0, 0, nsrc, 0, -1); }
  }

  printf("DONE %d nodes!\n", currnnodes);
  free(keys);
}

void Tree::dispose()
{
  free(xdata);
  free(ydata);
  free(vdata);
  free(nodes);
  free(bufhelpers);
  free(expansions);
}

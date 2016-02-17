/*
 *  upward.cpp
 *  2d-treecodes
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

#include "upward.h"
#include "upward-kernels.h"
#include "sort-sources.h"

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

    inline int fetch_and_add( int * variable, int value )
    {
	asm volatile("lock; xaddl %%eax, %2;"
		     :"=a" (value)                  
		     :"a" (value), "m" (*variable)  
		     :"memory");
	return value;
    }

    realtype *xdata = NULL, *ydata = NULL, *vdata = NULL, * expansions = NULL;
    
    Node * nodes;
    NodeHelper * bufhelpers = NULL;
    
    int currnnodes, maxnodes;

    int * keys = NULL;
    realtype ext, xmin, ymin;
    
    void build_leaf(const int nodeid, const int s, const int e,
		      const realtype x0, const realtype y0, const realtype h)
    {
	Node * node = nodes + nodeid;
	NodeHelper * helper = bufhelpers + nodeid;

	upward_setup(xdata + s, ydata + s, vdata + s, e - s,
		     &node->mass, &helper->w, &helper->wx, &helper->wy, &node->r);

	node->xcom = helper->w ? helper->wx / helper->w : (x0 + 0.5 * h);
	node->ycom = helper->w ? helper->wy / helper->w : (y0 + 0.5 * h);
	
	upward_p2e(xdata + s, ydata + s, vdata + s, e - s,
		   node->xcom, node->ycom,
		   expansions + ORDER * (2 * nodeid + 0),
		   expansions + ORDER * (2 * nodeid + 1));
		
	assert(node->r < 1.5 * h);

#ifndef NDEBUG
	assert(node->xcom >= x0 && node->xcom < x0 + h &&
	       node->ycom >= y0 && node->ycom < y0 + h ||
	       node->e - node->s == 0);
#endif
	
	while(helper->parent >= 0)
	{
	    Node * parent = nodes + helper->parent;
	    NodeHelper * parenthelper = bufhelpers + helper->parent;
	    
	    const bool e2e = 3 == fetch_and_add(&parenthelper->validchildren, 1);

	    if (!e2e)
		break;

	    const int childbase = parent->state.childbase;
	    realtype msum = 0, wsum = 0, wxsum = 0, wysum = 0;
	    
	    for(int c = 0; c < 4; ++c)
	    {
		const int childid = childbase + c;
		const Node * child = nodes + childid;
		msum += child->mass;		
		
		const NodeHelper * childhelper = bufhelpers + childid;
		wsum += childhelper->w;
		wxsum += childhelper->wx;
		wysum += childhelper->wy;
	    }
	    
	    parent->mass = msum;
	    parenthelper->w = wsum;
	    parenthelper->wx = wxsum;
	    parenthelper->wy = wysum;
	    
	    assert(wsum);
	    const realtype xcom_parent = wxsum / wsum;
	    const realtype ycom_parent = wysum / wsum;
	    
	    realtype rr = 0;
	    for(int c = 0; c < 4; ++c)
	    {
		const int childid = childbase + c;
		const Node * child = nodes + childid;
		const NodeHelper * childhelper = bufhelpers + childid;
		    
		if (childhelper->w)
		{
		    const realtype rx = xcom_parent - child->xcom;
		    const realtype ry = ycom_parent - child->ycom;
		    
		    rr = std::max(rr, child->r + sqrt(rx * rx + ry * ry));
		}
	    }

	    parent->r = std::min(rr, 1.4143f * ext / (1 << parenthelper->l));
	    parent->xcom = xcom_parent;
	    parent->ycom = ycom_parent;

#ifndef NDEBUG
	    {
		realtype r = 0;
		
		for(int i = s; i < e; ++i)
		    r = std::max(r, pow(xdata[i] - parent->xcom, (realtype)2) +
				 pow(ydata[i] - parent->ycom, (realtype)2));
		
		assert (sqrt(r) <= parent->r);
	    }
#endif
	    
	    int zcount = 0, nzentry = 0;
	    for(int c = 0; c < 4; ++c)
	    {
		const realtype val = bufhelpers[childbase + c].w;
		const bool negligible = fabs(val) < 1e-16;
		
		zcount += negligible;
		nzentry += (!negligible) * (childbase + c);
	    }
		
	    realtype * const dst = expansions + ORDER * (2 * helper->parent + 0);

	    if (zcount == 4)
		for(int i = 0; i < 2 * ORDER; ++i)
		    dst[i] =0;
	    else
		if (zcount == 3)
		{
		    realtype * const src = expansions + ORDER * (2 * nzentry + 0);
		    
		    for(int i = 0; i < 2 * ORDER; ++i)
			dst[i] = src[i];
		}	
		else
		{
		    realtype srcmass[4], rx[4], ry[4];
		    const realtype * chldrxp[4], *chldixp[4];
		    
		    for(int c = 0; c < 4; ++c)
		    {
			const int childid = childbase + c;
		    
			srcmass[c] = nodes[childid].mass;
			rx[c] = nodes[childid].xcom - parent->xcom;
			ry[c] = nodes[childid].ycom - parent->ycom;
		
			if (bufhelpers[childid].w == 0)
			    rx[c] = ry[c] = 1;
			
			chldrxp[c] = expansions + ORDER * (2 * childid + 0);
			chldixp[c] = expansions + ORDER * (2 * childid + 1);
		    }

		    upward_e2e(rx, ry, srcmass, chldrxp, chldixp, dst, dst + ORDER);
		}
#ifndef NDEBUG
		for(int i = 0; i < 2 * ORDER; ++i)
		    assert(!isnan(expansions[ORDER * (2 * helper->parent + 0) + i]));
#endif
		
	    node = parent; 
	    helper = parenthelper;
	}
    }

    int LEAF_MAXCOUNT;
    
    void build_tree(const int nodeid)
    {
	Node * const node = nodes + nodeid;
	NodeHelper * const helper = bufhelpers + nodeid;
	
	const int s = node->s;
	const int e = node->e;
	const int x = helper->x;
	const int y = helper->y;
	const int l = helper->l;
	const int mask = helper->mask;
	
	const double h = ext / (1 << l);
	const double x0 = xmin + h * x, y0 = ymin + h * y;

	assert(x < (1 << l) && y < (1 << l) && x >= 0 && y >= 0);

#ifndef NDEBUG
	for(int i = s; i < e; ++i)
	    assert(xdata[i] >= x0 && xdata[i] < x0 + h && ydata[i] >= y0 && ydata[i] < y0 + h);
#endif

	const bool leaf = e - s <= LEAF_MAXCOUNT || l + 1 > LMAX;
	
	if (leaf)
	    build_leaf(nodeid, s, e, x0, y0, h);
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
	
		const size_t indexmin = c == 0 ? s : lower_bound_vec(s, e, key1, keys);
		const size_t indexsup = c == 3 ? e : upper_bound_vec(s, e, key2, keys);

		const int childid = childbase + c;
		nodes[childid].setup(indexmin, indexsup);
		bufhelpers[childid].setup((x << 1) + (c & 1), (y << 1) + (c >> 1), l + 1, key1, nodeid);
		
#pragma omp task firstprivate(childid) if (indexsup - indexmin > 5e3 && c < 3)
		{
		    build_tree(childid);
		}
	    }
	}
    }
}	    

void Tree::build(const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc,
		 const int nsrc,
		 const int leaf_maxcapacity)
{
    const double t0 = omp_get_wtime();
    const int isdynamic = omp_get_dynamic();
    const int maxthreads = omp_get_max_threads();

    Tree::LEAF_MAXCOUNT = leaf_maxcapacity;

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
	{
	    nodes[0].setup(0, nsrc);
	    bufhelpers[0].setup(0, 0, 0, 0, -1);

	    build_tree(0);
	}
    }

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

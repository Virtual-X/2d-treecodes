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

#pragma once

#include <cassert>
//#include <cmath>

//#include <parallel/algorithm>
//#include <limits>
#include <cinttypes>
#include <algorithm>
#include <utility>

typedef REAL realtype;

#include "reference-upward-kernels.h"

namespace Tree
{
    struct Node
    {
	int x, y, l, s, e;
	bool leaf;
	realtype w, wx, wy, mass, r;

	Node * children[4];

	void setup(int x, int y, int l, int s, int e, bool leaf)
	    {
		this->x = x;
		this->y = y;
		this->l = l;
		this->s = s;
		this->e = e;
		this->leaf = leaf;

	    }

	realtype xcom() const { return wx / w; }
	realtype ycom() const { return wy / w; }

	Node() 
	    { 
		for (int i = 0; i < 4; ++i) 
		    children[i] = nullptr; 

		w = wx = wy = mass = r = 0; 
	    }
		
	typedef realtype alignedvec[ORDER] __attribute__ ((aligned (32)));

	alignedvec rexpansions;
	alignedvec iexpansions;

	void allocate_children()
	    {
		for(int i = 0; i < 4; ++i)
		    children[i] = new Node;
	    }
	
	realtype * rexp(){return &rexpansions[0];} 
	realtype * iexp(){return &iexpansions[0];}
	
	void p2e(const realtype * __restrict__ const xsources,
		 const realtype * __restrict__ const ysources,
		 const realtype * __restrict__ const vsources,
		 const double x0, const double y0, const double h) 
	    {
		reference_upward_p2e(xsources, ysources, vsources, e - s,
				     x0, y0, h, &mass, &w, &wx, &wy, &r,
				     rexpansions, iexpansions);
	    }

	void e2e() 
	    {
		realtype srcmass[4], rx[4], ry[4];
		realtype * chldrxp[4], *chldixp[4];

		for(int c = 0; c < 4; ++c)
		{
		    Node * chd = children[c];

		    srcmass[c] = chd->mass;
		    rx[c] = chd->xcom() - xcom();
		    ry[c] = chd->ycom() - ycom();
		    chldrxp[c] = chd->rexpansions;
		    chldixp[c] = chd->iexpansions;
		}

		reference_upward_e2e(rx, ry, srcmass, chldrxp, chldixp, rexpansions, iexpansions);
#ifndef NDEBUG
		{
		    for(int i = 0; i < ORDER; ++i)
			assert(!std::isnan((double)rexpansions[i]) && !std::isnan(iexpansions[i]));
		}
#endif
	    }

	~Node() 
	    {
		for(int i = 0; i < 4; ++i)
		    if (children[i])
		    {
			delete children[i];

			children[i] = nullptr;
		    }
	    }
    };

    extern realtype *xdata, *ydata, *vdata;

    void build(const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
	       Node * const root, const int LEAF_MAXCOUNT, const int exporder);

    void dispose();
};

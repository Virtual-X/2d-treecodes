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
#include <cmath>

#include <parallel/algorithm>
#include <limits>

typedef REAL realtype;

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

	Node() { for (int i = 0; i < 4; ++i) children[i] = nullptr; w = wx = wy = mass = r = 0; }


	virtual void allocate_children() = 0;

	virtual void p2e(const realtype * __restrict__ const xsources,
			 const realtype * __restrict__ const ysources,
			 const realtype * __restrict__ const sources,
			 const double x0, const double y0, const double h) = 0;

	virtual void e2e() = 0;

	virtual ~Node()
	    {

	    }
    };

#ifdef ORDER
    template<int XXXDONTCARE>
	struct NodeImplementation : Node
    {
	realtype expansions[2][ORDER];

	void allocate_children() override
	{
	    for(int i = 0; i < 4; ++i)
		children[i] = new NodeImplementation;
	}

	void p2e(const realtype * __restrict__ const xsources,
		 const realtype * __restrict__ const ysources,
		 const realtype * __restrict__ const vsources,
		 const double x0, const double y0, const double h) override
	{
	    P2E_KERNEL(xsources, ysources, vsources, e - s,
		       x0, y0, h, &mass, &w, &wx, &wy, &r,
		       expansions[0], expansions[1]);
	}

	void e2e() override
	{
	    V4 srcmass, rx, ry, chldexp[2][ORDER];

	    for(int c = 0; c < 4; ++c)
	    {
		NodeImplementation * chd = (NodeImplementation *)children[c];

		srcmass[c] = chd->mass;
		rx[c] = chd->xcom();
		ry[c] = chd->ycom();

		for(int i = 0; i < 2; ++i)
		    for(int j = 0; j < ORDER; ++j)
			chldexp[i][j][c] = chd->expansions[i][j];
	    }

	    rx -= xcom();
	    ry -= ycom();

	    E2E_KERNEL(srcmass, rx, ry, chldexp[0], chldexp[1], expansions[0], expansions[1]);
#ifndef NDEBUG
	    {
		for(int i = 0; i < ORDER; ++i)
		    assert(!std::isnan((double)expansions[0][i]) && !std::isnan(expansions[1][i]));
	    }
#endif
	}

	~NodeImplementation() override
	{
	    for(int i = 0; i < 4; ++i)
		if (children[i])
		{
		    delete children[i];

		    children[i] = nullptr;
		}
	}
    };
#endif

    extern realtype *xdata, *ydata, *vdata;

    void build(const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
	       const realtype * const xdst, const realtype * const ydst, const int ndst,  Node * const root);

    void dispose();
};

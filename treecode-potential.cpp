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
#include <cstring>

#include "treecode.h"
#include "upward-kernels.h"
#include "potential-kernels.h"
#include "upward.h"

namespace EvaluatePotential
{
    realtype thetasquared;
    
    struct NodePotential : Tree::Node
    {
	realtype expansions[2][ORDER];
	
	void allocate_children() override
	    {
		for(int i = 0; i < 4; ++i)
		    children[i] = new NodePotential;
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
		    NodePotential * chd = (NodePotential *)children[c];

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

	~NodePotential() override
	    {
		for(int i = 0; i < 4; ++i)
		    if (children[i])
		    {
			delete children[i];
			    
			children[i] = nullptr;
		    }
	    }
    };

    realtype *xdata = nullptr, *ydata = nullptr, *vdata = nullptr;
    
    void evaluate(realtype * const result, const realtype xt, const realtype yt, const NodePotential & node)
    {
	const realtype r2 = pow(xt - node.xcom(), 2) + pow(yt - node.ycom(), 2);

	if (4 * node.r * node.r < thetasquared * r2)
	    *result = potential_e2p(node.mass, xt - node.xcom(), yt - node.ycom(), node.expansions[0], node.expansions[1]);
	else
	{
	    if (node.leaf)
	    {
		const int s = node.s;

		*result = potential_p2p(&xdata[s], &ydata[s], &vdata[s], node.e - s, xt, yt);
	    }
	    else
	    {
		realtype s[4] = {0, 0, 0, 0};

		for(int c = 0; c < 4; ++c)
		{
		    NodePotential * chd = (NodePotential *)node.children[c];
		    realtype * ptr = s + c;

		    evaluate(ptr, xt, yt, *chd);
		}

		*result = s[0] + s[1] + s[2] + s[3];
	    }
	}
    }
}

using namespace EvaluatePotential;

extern "C"
void treecode_potential(const realtype theta,
			const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
			const realtype * const xdst, const realtype * const ydst, const int ndst, realtype * const vdst)
{
    thetasquared = theta * theta;

    NodePotential root;
    
    Tree::build(xsrc, ysrc, vsrc, nsrc, xdst, ydst, ndst, &root);
    
    xdata = Tree::xdata;
    ydata = Tree::ydata;
    vdata = Tree::vdata;
        
#pragma omp parallel for schedule(static,1)
    for(int i = 0; i < ndst; ++i)
	evaluate(vdst + i, xdst[i], ydst[i], root);
}


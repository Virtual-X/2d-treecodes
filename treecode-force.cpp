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
#include "force-kernels.h"
#include "upward.h"

namespace EvaluateForce
{
    realtype thetasquared;
    
    struct NodeForce : Tree::Node
    {
	realtype expansions[2][ORDER];
	
	void allocate_children() override
	    {
		for(int i = 0; i < 4; ++i)
		    children[i] = new NodeForce;
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
		    NodeForce * chd = (NodeForce *)children[c];

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

	~NodeForce() override
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
    
    void evaluate(realtype * const xresult, realtype * const yresult, const realtype xt, const realtype yt, const NodeForce & node)
    {
	const realtype r2 = pow(xt - node.xcom(), 2) + pow(yt - node.ycom(), 2);

	if (4 * node.r * node.r < thetasquared * r2)
	    force_e2p(node.mass, xt - node.xcom(), yt - node.ycom(), node.expansions[0], node.expansions[1], xresult, yresult);
	else
	{
	    if (node.leaf)
	    {
		const int s = node.s;

		force_p2p(&xdata[s], &ydata[s], &vdata[s], node.e - s, xt, yt, xresult, yresult);
	    }
	    else
	    {
		realtype xs[4] = {0, 0, 0, 0}, ys[4] = {0, 0, 0, 0};

		for(int c = 0; c < 4; ++c)
		{
		    NodeForce * chd = (NodeForce *)node.children[c];
		    realtype * xptr = xs + c, * yptr = ys + c;

		    evaluate(xptr, yptr, xt, yt, *chd);
		}

		*xresult = xs[0] + xs[1] + xs[2] + xs[3];
		*yresult = ys[0] + ys[1] + ys[2] + ys[3];
	    }
	}
    }
}

using namespace EvaluateForce;

extern "C"
void treecode_force(const realtype theta,
		    const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
		    const realtype * const xdst, const realtype * const ydst, const int ndst, realtype * const xresult, realtype * const yresult)   
{
    thetasquared = theta * theta;

    NodeForce root;
    
    Tree::build(xsrc, ysrc, vsrc, nsrc, xdst, ydst, ndst,  &root);
    
    xdata = Tree::xdata;
    ydata = Tree::ydata;
    vdata = Tree::vdata;
        
#pragma omp parallel for schedule(static,1)
    for(int i = 0; i < ndst; ++i)
	evaluate(xresult + i, yresult + i, xdst[i], ydst[i], root);
}


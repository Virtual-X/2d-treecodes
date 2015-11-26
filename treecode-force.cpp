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
    struct NodeForce : Tree::NodeImplementation<ORDER> { };

    realtype thetasquared, *xdata = nullptr, *ydata = nullptr, *vdata = nullptr;
    
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


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

#include "treecode.h"
#include "potential-kernels.h"
#include "upward.h"

namespace EvaluatePotential
{
    realtype thetasquared;
    
    realtype *xdata = NULL, *ydata = NULL, *vdata = NULL;
    
    void evaluate(realtype * const result, const realtype xt, const realtype yt, const Node & node)
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
		    Node * chd = node.children[c];
		    realtype * ptr = s + c;

		    evaluate( ptr, xt, yt, *chd);
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

    const double tstart = omp_get_wtime();
    Tree::build(xsrc, ysrc, vsrc, nsrc, xdst, ydst, ndst, vdst);
    const double tend = omp_get_wtime();

    printf("tree built in %.2f ms\n", (tend - tstart) * 1e3);
    xdata = Tree::xdata;
    ydata = Tree::ydata;
    vdata = Tree::vdata;
    const double tstart2 = omp_get_wtime();
    
#pragma omp parallel for schedule(static,1)
    for(int i = 0; i < ndst; ++i)
	evaluate(vdst + i, xdst[i], ydst[i], *Tree::root);
    
    
    const double tend2 = omp_get_wtime();

    printf("tree evaluated in %.2f ms\n", (tend2 - tstart2) * 1e3);
}


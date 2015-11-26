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

#include "potential-kernels.h"
#include "upward.h"

realtype thetasquared;

void evaluate(Tree& tree, realtype * const result, const realtype xt, const realtype yt, const Node & node)
{
    const realtype r2 = pow(xt - node.xcom(), 2) + pow(yt - node.ycom(), 2);

    if (4 * node.r * node.r < thetasquared * r2)
	*result = treecode_e2p(node.mass, xt - node.xcom(), yt - node.ycom(), node.expansions[0], node.expansions[1]);
    else
    {
	if (node.leaf)
	{
	    const int s = node.s;

	    *result = treecode_p2p(&tree.xdata[s], &tree.ydata[s], &tree.vdata[s], node.e - s, xt, yt);
	}
	else
	{
	    realtype s[4] = {0, 0, 0, 0};

	    for(int c = 0; c < 4; ++c)
	    {
		Node * chd = node.children[c];
		realtype * ptr = s + c;

		evaluate(tree, ptr, xt, yt, *chd);
	    }

	    *result = s[0] + s[1] + s[2] + s[3];
	}
    }
}

extern "C"
void treecode_potential(const realtype theta,
			const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
			const realtype * const xdst, const realtype * const ydst, const int ndst, realtype * const vdst)
{
    thetasquared = theta * theta;
    
    Tree tree(xsrc, ysrc, vsrc, nsrc, xdst, ydst, ndst, vdst);
    
    
#pragma omp for schedule(static,1)
    for(int i = 0; i < ndst; ++i)
	evaluate(tree, vdst + i, xdst[i], ydst[i], *tree.root);   
}


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

#include "upward.h"
#include "potential-kernels.h"

namespace EvaluatePotential
{
    realtype thetasquared;
    
    void evaluate(realtype * const result, const realtype xt, const realtype yt)
    {
	enum { BUFSIZE = 16 };
	
	int stack[LMAX * 3];

	int bufcount = 0;
	realtype rzs[BUFSIZE], izs[BUFSIZE], masses[BUFSIZE];
	const realtype * rxps[BUFSIZE], *ixps[BUFSIZE];

	int stackentry = 0, maxentry = 0;

	stack[0] = 0;
	*result = 0;
	while(stackentry > -1)
	{
	    const int nodeid = stack[stackentry--];
	    const Tree::Node * const node = Tree::nodes + nodeid;

	    assert(nodeid < node->state.childbase || !node->state.innernode);

	    realtype tmp[2];
	    
	    const realtype r2 = pow(xt - node->xcom, 2) + pow(yt - node->ycom, 2);

	    if (node->r * node->r < thetasquared * r2)
	    {
		rzs[bufcount] = xt - node->xcom;
		izs[bufcount] = yt - node->ycom;
		masses[bufcount] = node->mass;
		rxps[bufcount] = Tree::expansions + ORDER * (2 * nodeid + 0);
		ixps[bufcount] = Tree::expansions + ORDER * (2 * nodeid + 1);
		++bufcount;
		
		if (bufcount == BUFSIZE)
		{
		    bufcount = 0;
		 
		    *result += potential_e2p(rzs, izs, masses, rxps, ixps, BUFSIZE);
		}
	    }
	    else
	    {
		if (!node->state.innernode)
		{
		    const int s = node->s;

		    *result += potential_p2p(&Tree::xdata[s], &Tree::ydata[s], &Tree::vdata[s], node->e - s, xt, yt);
		}
		else
		{
		    for(int c = 0; c < 4; ++c)
			stack[++stackentry] = node->state.childbase + c;

		    maxentry = std::max(maxentry, stackentry);
		}
	    }
	}

	if (bufcount)
	    *result += potential_e2p(rzs, izs, masses, rxps, ixps, bufcount);
    }
}

using namespace EvaluatePotential;

extern "C"
__attribute__ ((visibility ("default")))
void treecode_potential(const realtype theta,
			const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
			const realtype * const xdst, const realtype * const ydst, const int ndst, realtype * const vdst)
{
    thetasquared = theta * theta;

    Tree::build(xsrc, ysrc, vsrc, nsrc, 32 * 16);

#pragma omp parallel for schedule(static,1)
    for(int i = 0; i < ndst; ++i)
	evaluate(vdst + i, xdst[i], ydst[i]);
    
    Tree::dispose();    
}


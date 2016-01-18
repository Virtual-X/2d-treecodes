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

#include <omp.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>

#include "cuda-common.h"

typedef REAL realtype; 

//#include "treecode-potential.h"

#include "potential-kernels.h"
#include "upward.h"

#define _INSTRUMENTATION_

namespace EvaluatePotential
{

    realtype thetasquared, *xdata = nullptr, *ydata = nullptr, *vdata = nullptr;

    void evaluate(realtype * const result, const realtype xt, const realtype yt, const Tree::Node & root)
    {
	const Tree::Node * stack[15 * 4 * 2];

	int stackentry = 0, maxentry = 0;

	stack[0] = &root;
	*result = 0;
	while(stackentry > -1)
	{
	    const Tree::Node * const node = stack[stackentry--];

	    //realtype tmp[2];

	    const realtype r2 = pow(xt - node->xcom(), 2) + pow(yt - node->ycom(), 2);

	    if (node->r * node->r < thetasquared * r2)
		*result += potential_e2p(node->mass, xt - node->xcom(), yt - node->ycom(), node->rexpansions, node->iexpansions);
	    else
	    {
		if (node->leaf)
		{
		    const int s = node->s;

		    *result += potential_p2p(&xdata[s], &ydata[s], &vdata[s], node->e - s, xt, yt);
		}
		else
		{
		    for(int c = 0; c < 4; ++c)
			stack[++stackentry] = (Tree::Node *)node->children[c];

		    maxentry = std::max(maxentry, stackentry);
		}
	    }
	}
    }
}

using namespace EvaluatePotential;

extern "C"
__attribute__ ((visibility ("default")))
void treecode_potential_solve(const realtype theta,
			const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
			const realtype * const xdst, const realtype * const ydst, const int ndst, realtype * const vdst)
{
    thetasquared = theta * theta;

    Tree::Node root;
    
    //CUDA_CHECK(cudaMemset(device_root, 0, sizeof(device_root)));
    
    const double t0 = omp_get_wtime();
    Tree::build(xsrc, ysrc, vsrc, nsrc, &root,  32 * 16, ORDER); //before: 64
    const double t1 = omp_get_wtime();
    
    
    xdata = Tree::xdata;
    ydata = Tree::ydata;
    vdata = Tree::vdata;

#pragma omp parallel for schedule(static,1)
    for(int i = 0; i < ndst; ++i)
	evaluate(vdst + i, xdst[i], ydst[i], root);

    Tree::dispose();
        
    const double t2 = omp_get_wtime();

#ifdef _INSTRUMENTATION_
    printf("UPWARD: %.2f ms EVAL: %.2f ms (%.1f %%)\n", (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t1) / (t2 - t0) * 100);
#endif
}


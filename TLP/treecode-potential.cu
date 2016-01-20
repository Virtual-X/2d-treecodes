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
#include <algorithm>

#include "cuda-common.h"

typedef REAL realtype; 

//#include "treecode-potential.h"

#include "potential-kernels.h"
#include "upward.h"

#define _INSTRUMENTATION_

#define STACKSIZE (LMAX * 4)

namespace EvaluatePotential
{
    __constant__ int nnodes;
    __constant__ Tree::Node * nodes;
    __constant__ realtype * expansions, *xdata, *ydata, *vdata;
  
    __global__ void evaluate(const realtype * const xts, const realtype * const yts, const realtype thetasquared, realtype * const results, const int ndst)
    {
	assert(blockDim.x == 32);
	
	const int tid = threadIdx.x;
	const bool master = tid == 0;
	
	const int gid = threadIdx.y + blockDim.y * blockIdx.x;

	if (gid >= ndst)
	    return;
	
	const realtype xt = xts[gid];
	const realtype yt = yts[gid];

	extern __shared__ int ary[];

	volatile int * stack = ary + STACKSIZE * threadIdx.y;

	int stackentry = 0, maxentry = 0;

	if (master)
	    stack[0] = 0;
	
	realtype result = 0;

	while(stackentry > -1)
	{
	    const int nodeid = stack[stackentry--];
	    assert(nodeid < nnodes);

	    const Tree::Node node = *(nodes + nodeid);
	
	    if (node.e - node.s == 0)
	       	continue;

	    const realtype rx = xt - node.xcom;
	    const realtype ry = yt - node.ycom;
	    const realtype r2 = rx * rx + ry * ry;

	    if (node.r * node.r < thetasquared * r2)
	    {
		if (master)
		{
		    const realtype * rxp = expansions + ORDER * (0 + 2 * nodeid);
		    const realtype * ixp = expansions + ORDER * (1 + 2 * nodeid);
		    
		    result += potential_e2p(node.mass, xt - node.xcom, yt - node.ycom, rxp, ixp);
		}
	    }
	    else
	    {
		if (!node.state.innernode)
		{
		    const int s = node.s;

		    result += potential_p2p(&xdata[s], &ydata[s], &vdata[s], node.e - s, xt, yt);
		}
		else
		{
		    if (master)   
			for(int c = 0; c < 4; ++c)
			    stack[++stackentry] = node.state.children[c];
		    else
		    	stackentry += 4;
		    
		    maxentry = max(maxentry, stackentry);
		    assert(maxentry < STACKSIZE);
		}
	    }
	}

	if (master)
	    results[gid] = result;
    }
    
    void reference_evaluate(realtype * const result, const realtype xt, const realtype yt, realtype thetasquared)
    {
	int stack[LMAX * 4 * 2];

	int stackentry = 0, maxentry = 0;

	stack[0] = 0;
	*result = 0;
	while(stackentry > -1)
	{
	    const int nodeid = stack[stackentry--];
	    
	    const Tree::Node * const node = Tree::host_nodes + nodeid;

	    if (node->e - node->s == 0)
		continue;
	    
	    const realtype r2 = pow(xt - node->xcom, 2) + pow(yt - node->ycom, 2);

	    if (node->r * node->r < thetasquared * r2)
	    {
		const realtype * rxp = Tree::host_expansions + ORDER * (0 + 2 * nodeid);
		const realtype * ixp = Tree::host_expansions + ORDER * (1 + 2 * nodeid);
		
		*result += reference_potential_e2p(node->mass, xt - node->xcom, yt - node->ycom, rxp, ixp);
	    }
	    else
		if (!node->state.innernode)
		{
		    const int s = node->s;
		    
		    *result += reference_potential_p2p(&Tree::host_xdata[s], &Tree::host_ydata[s], &Tree::host_vdata[s], node->e - s, xt, yt);
		}
		else
		{
		    for(int c = 0; c < 4; ++c)
			stack[++stackentry] = node->state.children[c];
		    
		    maxentry = std::max(maxentry, stackentry);
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
    const realtype thetasquared = theta * theta;

    realtype * device_xdst, *device_ydst, *device_results;
    CUDA_CHECK(cudaMalloc(&device_xdst, sizeof(realtype) * ndst));
    CUDA_CHECK(cudaMalloc(&device_ydst, sizeof(realtype) * ndst));
    CUDA_CHECK(cudaMalloc(&device_results, sizeof(realtype) * ndst));
    
    CUDA_CHECK(cudaMemcpyAsync(device_xdst, xdst, sizeof(realtype) * ndst, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(device_ydst, ydst, sizeof(realtype) * ndst, cudaMemcpyHostToDevice));
    
    const double t0 = omp_get_wtime();
    Tree::build(xsrc, ysrc, vsrc, nsrc,  32 * 16); //before: 64
    const double t1 = omp_get_wtime();

#if 1
    CUDA_CHECK(cudaMemcpyToSymbolAsync(xdata, &Tree::device_xdata, sizeof(Tree::device_xdata)));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(ydata, &Tree::device_ydata, sizeof(Tree::device_ydata)));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(vdata, &Tree::device_vdata, sizeof(Tree::device_vdata)));

    CUDA_CHECK(cudaMemcpyToSymbolAsync(nodes, &Tree::device_nodes, sizeof(Tree::device_nodes)));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(nnodes, &Tree::nnodes, sizeof(Tree::nnodes)));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(expansions, &Tree::device_expansions, sizeof(Tree::device_expansions)));

    const int yblocksize = 8;
    evaluate<<<(ndst + yblocksize - 1) / yblocksize, dim3(32, yblocksize), STACKSIZE * sizeof(int)* yblocksize>>>(
	device_xdst, device_ydst, thetasquared, device_results, ndst);
    CUDA_CHECK(cudaPeekAtLastError());
        
    CUDA_CHECK(cudaMemcpyAsync(vdst, device_results, sizeof(realtype) * ndst, cudaMemcpyDeviceToHost));
    
#else
#pragma omp parallel for schedule(static,1)
    for(int i = 0; i < ndst; ++i)
	reference_evaluate(vdst + i, xdst[i], ydst[i], thetasquared);
#endif
    
    Tree::dispose();

    const double t2 = omp_get_wtime();

#ifdef _INSTRUMENTATION_
    printf("UPWARD: %.2f ms EVAL: %.2f ms (%.1f %%)\n", (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t1) / (t2 - t0) * 100);
#endif

    CUDA_CHECK(cudaFree(device_xdst));
    CUDA_CHECK(cudaFree(device_ydst));
    CUDA_CHECK(cudaFree(device_results));
    
}


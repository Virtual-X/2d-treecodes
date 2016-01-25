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

#include "potential-kernels.h"
#include "upward.h"

#define _INSTRUMENTATION_

#define STACKSIZE (LMAX * 4)
#define _NO_THREADCOOP_

struct SharedBuffers
{
#ifndef _NO_THREADCOOP_
    realtype scratch[2][32];
    int buffered_e2ps[8];
#else
    int buffered_e2ps[32];
#endif
    int stack[LMAX * 4];
};

namespace EvaluatePotential
{
#ifndef NDEBUG
    __constant__ int nnodes;
#endif
    __constant__ Tree::Node * nodes;
    __constant__ realtype * expansions, *xdata, *ydata, *vdata;



__device__ __forceinline__ double potential_e2pAZZ(const double mass,
				const double rz,
				const double iz,
				const double * __restrict__ const rxp,
				const double * __restrict__ const ixp,
				double * const scratch) // size of 32 * 2 * sizeof(double)
{
    //asm volatile ("L_AZZ:");
    
    const int tid = threadIdx.x;
    assert(tid < 32 && blockDim.x == 32);

    const int mask = tid & 0x3;
    const int base = tid & ~0x3;
    
    const double r2 = rz * rz + iz * iz;

    if (mask == 0)
    {	
	scratch[base + 0] = rz / r2;
    	scratch[32 + base + 0] = -iz / r2;

    	scratch[base + 1] = scratch[base + 0] * scratch[base + 0] - scratch[32 + base + 0] * scratch[32 + base + 0];
    	scratch[32 + base + 1] = 2 * scratch[32 + base + 0] * scratch[base + 0];
    }

    if (mask < 2)
    {
        scratch[base + mask + 2] = scratch[base + mask] * scratch[base + 1] - scratch[32 + base + mask] * scratch[32 + base + 1];
	scratch[32 + base + mask + 2] = scratch[base + mask] * scratch[32 + base + 1] + scratch[32 + base + mask] * scratch[base + 1];
    }

    const double rinvz_4 = scratch[base + 3];
    const double iinvz_4 = scratch[32 + base + 3];

    double rprod = scratch[tid];
    double iprod = scratch[32 + tid];

    double rsum = 0, rtmp, itmp;
    
    
    
    rsum += rxp[0 + mask] * rprod - ixp[0 + mask] * iprod;
        
    rtmp = rinvz_4 * rprod - iinvz_4 * iprod;
    itmp = rinvz_4 * iprod + iinvz_4 * rprod;
    rprod = rtmp;
    iprod = itmp;
    
    rsum += rxp[4 + mask] * rprod - ixp[4 + mask] * iprod;
        
    rtmp = rinvz_4 * rprod - iinvz_4 * iprod;
    itmp = rinvz_4 * iprod + iinvz_4 * rprod;
    rprod = rtmp;
    iprod = itmp;
    
    rsum += rxp[8 + mask] * rprod - ixp[8 + mask] * iprod;
        

    if (mask == 0)
	rsum +=  mass * log(r2) / 2;

    //asm volatile ("; //AZZ END");
    return  rsum;
}

    

    __global__ void   //  __launch_bounds__(128, 12)
    evaluate(const realtype * const xts, const realtype * const yts, const realtype thetasquared, realtype * const results, const int ndst)
    {
	assert(blockDim.x == 32);
	
	const int tid = threadIdx.x;
	const bool master = tid == 0;
	
	const int gid = threadIdx.y + blockDim.y * blockIdx.x;

	if (gid >= ndst)
	    return;
	
	const realtype xt = xts[gid];
	const realtype yt = yts[gid];

	extern __shared__ SharedBuffers ary[];
#ifndef _NO_THREADCOOP_
	realtype * scratch = ary[threadIdx.y].scratch[0];
#endif
	int * stack = ary[threadIdx.y].stack;
	int * buffered_e2ps = ary[threadIdx.y].buffered_e2ps;
	int counter_e2ps = 0;
	
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
		assert(counter_e2ps < 8);

		if (master)
		    buffered_e2ps[counter_e2ps] = nodeid;

		counter_e2ps++;

#ifndef _NO_THREADCOOP_
		if (counter_e2ps == 8)
		{
		    counter_e2ps = 0;

		    const int mynodeid = buffered_e2ps[tid / 4];
		    assert(mynodeid < nnodes);
	    
		    const Tree::Node * mynode = nodes + mynodeid;

		    result += potential_e2p(mynode->mass, xt - mynode->xcom, yt - mynode->ycom, 
					    expansions + ORDER * (0 + 2 * mynodeid), 
					    expansions + ORDER * (1 + 2 * mynodeid), scratch);
		}
#else
		if (counter_e2ps == 32)
		{
		    counter_e2ps = 0;

		    const int mynodeid = buffered_e2ps[tid];
		    assert(mynodeid < nnodes);
	    
		    const Tree::Node * mynode = nodes + mynodeid;

		    result += potential_e2p_individual(mynode->mass, xt - mynode->xcom, yt - mynode->ycom, 
					    expansions + ORDER * (0 + 2 * mynodeid), 
					    expansions + ORDER * (1 + 2 * mynodeid));
		}
#endif
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

#ifndef _NO_THREADCOOP_
  	if (tid / 4 < counter_e2ps)
	{
	    const int mynodeid = buffered_e2ps[tid / 4];
	    assert(mynodeid < nnodes);
	    
	    const Tree::Node * mynode = nodes + mynodeid;

	    result += potential_e2p(mynode->mass, xt - mynode->xcom, yt - mynode->ycom, 
				    expansions + ORDER * (0 + 2 * mynodeid), 
				    expansions + ORDER * (1 + 2 * mynodeid), scratch);
	}
#else
	if (tid < counter_e2ps)
	{
	    const int mynodeid = buffered_e2ps[tid];
	    assert(mynodeid < nnodes);
	    
	    const Tree::Node * mynode = nodes + mynodeid;

	    result += potential_e2p_individual(mynode->mass, xt - mynode->xcom, yt - mynode->ycom, 
					       expansions + ORDER * (0 + 2 * mynodeid), 
					       expansions + ORDER * (1 + 2 * mynodeid));
	}
#endif

	result += __shfl_xor(result, 16 );
	result += __shfl_xor(result, 8 );
	result += __shfl_xor(result, 4 );
	result += __shfl_xor(result, 2 );
	result += __shfl_xor(result, 1 );

	if (master)
	    results[gid] = result;
    }

#ifndef NDEBUG
    void reference_evaluate(realtype * const result, const realtype xt, const realtype yt, realtype thetasquared)
    {
	int stack[LMAX * 4 * 2];

	int stackentry = 0, maxentry = 0;

	stack[0] = 0;
	*result = 0;
	//int slast = 0, elast = 0;
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
		    //rintf("%d %d   %d %d\n", slast, s, elast, node->e);
		
		    //assert(s >= slast && node->e >= elast);
		    //slast = s;
		    //elast = node->e;
		}
		else
		{
		    for(int c = 0; c < 4; ++c)
			stack[++stackentry] = node->state.children[c];
		    
		    maxentry = std::max(maxentry, stackentry);
		}
	}
    }
#endif
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
#ifndef NDEBUG
    CUDA_CHECK(cudaMemcpyToSymbolAsync(nnodes, &Tree::nnodes, sizeof(Tree::nnodes)));
#endif
    CUDA_CHECK(cudaMemcpyToSymbolAsync(expansions, &Tree::device_expansions, sizeof(Tree::device_expansions)));

    const int yblocksize = 4;
    evaluate<<<(ndst + yblocksize - 1) / yblocksize, dim3(32, yblocksize),
	sizeof(SharedBuffers) * yblocksize>>>(
	    device_xdst, device_ydst, thetasquared, device_results, ndst);
    CUDA_CHECK(cudaPeekAtLastError());
         
    CUDA_CHECK(cudaMemcpyAsync(vdst, device_results, sizeof(realtype) * ndst, cudaMemcpyDeviceToHost));
    
#else
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


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

#include <cstdio>
#include <cassert>

#include "cuda-common.h"
#include "force-kernels.h"
#include "upward.h"

#define ACCESS(x) __ldg(&(x)) 

namespace EvaluateForce
{
     struct SharedBuffers
    {
	realtype scratch[64];
	int buffered_e2ps[8];
	int stack[LMAX * 3];
    };

#ifndef NDEBUG
    __constant__ int nnodes;
#endif
    __constant__ Tree::Node * nodes;
    __constant__ realtype * expansions, *xdata, *ydata, *vdata;

    /*__global__ void  __launch_bounds__(128, 16)
    evaluate(const realtype * const xts, const realtype * const yts, const realtype thetasquared, 
	     realtype * const xforce, realtype * const yforce, const int ndst)
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

	realtype * scratch = ary[threadIdx.y].scratch;

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

	    const Tree::Node * node = nodes + nodeid;
	    const realtype nodemass = ACCESS(node->mass);
	
	    if (nodemass == 0)
	       	continue;

	    const realtype xcom = ACCESS(node->xcom);
	    const realtype ycom = ACCESS(node->ycom);
	    const realtype r = ACCESS(node->r);
	    const realtype rx = xt - xcom;
	    const realtype ry = yt - ycom;
	    const realtype r2 = rx * rx + ry * ry;

	    if (r * r < thetasquared * r2)
 	    {
		if (master)
		    buffered_e2ps[counter_e2ps] = nodeid;

		counter_e2ps++;

		if (counter_e2ps == 8)
		{
		    counter_e2ps = 0;

		    const int mynodeid = buffered_e2ps[tid / 4];
		    assert(mynodeid < nnodes);
	    
		    const Tree::Node * mynode = nodes + mynodeid;

		    result += force_e2p(ACCESS(mynode->mass), xt - ACCESS(mynode->xcom), yt - ACCESS(mynode->ycom), 
					expansions + ORDER * (0 + 2 * mynodeid), 
					expansions + ORDER * (1 + 2 * mynodeid), scratch);
		}
	    }
	    else 
	    {
		if (!node->state.innernode)
		{
		    const int s = node->s;

		    result += potential_p2p(&xdata[s], &ydata[s], &vdata[s], node->e - s, xt, yt);
		}
		else
		{
		    if (master)   
		    {
			const int childbase = ACCESS(node->state.childbase);

			for(int c = 0; c < 4; ++c) 
			    stack[++stackentry] = childbase + c;
		    }
		    else
			stackentry += 4;
			    
		    maxentry = max(maxentry, stackentry);
		    assert(maxentry < LMAX * 3);
		}
	    }
	}

  	if (tid / 4 < counter_e2ps)
	{
	    const int mynodeid = buffered_e2ps[tid / 4];
	    assert(mynodeid < nnodes);
	    
	    const Tree::Node * mynode = nodes + mynodeid;

	    result += force_e2p(ACCESS(mynode->mass), xt - ACCESS(mynode->xcom), yt - ACCESS(mynode->ycom), 
				expansions + ORDER * (0 + 2 * mynodeid), 
				expansions + ORDER * (1 + 2 * mynodeid), scratch);
	}

	result += __shfl_xor(result, 16 );
	result += __shfl_xor(result, 8 );
	result += __shfl_xor(result, 4 );
	result += __shfl_xor(result, 2 );
	result += __shfl_xor(result, 1 );

	if (master)
	    results[gid] = result;
	    }*/


}

void reference_evaluate(realtype * const xforce, realtype * const yforce, 
			const realtype xt, const realtype yt, realtype thetasquared);

using namespace EvaluateForce;
   
extern "C"
__attribute__ ((visibility ("default")))
void treecode_force_mrag_solve(const realtype theta,
			       const realtype * const xsrc,
			       const realtype * const ysrc,
			       const realtype * const vsrc,
			       const int nsrc,
			       const realtype * const x0s,
			       const realtype * const y0s,
			       const realtype * const hs,
			       const int nblocks,
			       realtype * const xforce,
			       realtype * const yforce)
{
    const realtype thetasquared = theta * theta;
    
    realtype * device_x0s, *device_y0s, *device_hs, *device_xforce, *device_yforce;
    
    const int ndst = nblocks * BLOCKSIZE * BLOCKSIZE;

    CUDA_CHECK(cudaMalloc(&device_x0s, sizeof(realtype) * ndst));
    CUDA_CHECK(cudaMalloc(&device_y0s, sizeof(realtype) * ndst));
    CUDA_CHECK(cudaMalloc(&device_hs, sizeof(realtype) * ndst));

    CUDA_CHECK(cudaMalloc(&device_xforce, sizeof(realtype) * ndst));
    CUDA_CHECK(cudaMalloc(&device_yforce, sizeof(realtype) * ndst));
    
    CUDA_CHECK(cudaMemcpyAsync(device_x0s, x0s, sizeof(realtype) * nblocks, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(device_y0s, y0s, sizeof(realtype) * nblocks, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(device_hs, hs, sizeof(realtype) * nblocks, cudaMemcpyHostToDevice));
    
    Tree::build(xsrc, ysrc, vsrc, nsrc, 512);
    //CUDA_CHECK(cudaDeviceSynchronize());
#if 0
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
	sizeof(SharedBuffers) * yblocksize>>>(device_xdst, device_ydst, thetasquared, device_xforce, device_yforce, ndst);
    CUDA_CHECK(cudaPeekAtLastError());
         
    CUDA_CHECK(cudaMemcpyAsync(xdst, device_xforce, sizeof(realtype) * ndst, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(ydst, device_yforce, sizeof(realtype) * ndst, cudaMemcpyDeviceToHost));
#else
    for(int c = 0, b = 0; b < nblocks; ++b)
	for(int iy = 0; iy < BLOCKSIZE; ++iy)
	    for(int ix = 0; ix < BLOCKSIZE; ++ix, ++c)
	    {
		const realtype xdst = x0s[b] + hs[b] * ix;
		const realtype ydst = y0s[b] + hs[b] * iy;

		reference_evaluate(xforce + c, yforce + c, xdst, ydst, thetasquared);
	    }
#endif
    
    Tree::dispose();

    CUDA_CHECK(cudaFree(device_x0s));
    CUDA_CHECK(cudaFree(device_y0s));   
    CUDA_CHECK(cudaFree(device_hs));
    CUDA_CHECK(cudaFree(device_xforce));
    CUDA_CHECK(cudaFree(device_yforce));
}

void reference_evaluate(realtype * const xforce, realtype * const yforce, 
			const realtype xt, const realtype yt, realtype thetasquared)
{
    const double eps = 10 * __DBL_EPSILON__;
    int stack[LMAX * 3];

    int stackentry = 0, maxentry = 0;
    
    stack[0] = 0;

    *xforce = 0;
    *yforce = 0;
	
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

	    const realtype rz = xt - node->xcom;
	    const realtype iz = yt - node->ycom;

	    const realtype rinvz_1 = rz / r2;
	    const realtype iinvz_1 = -iz / r2;

	    realtype rsum = node->mass * rinvz_1, isum = node->mass * iinvz_1;
	    realtype rprod = rinvz_1, iprod = iinvz_1;

	    for(int j = 0; j < ORDER; ++j)
	    {
		const realtype rtmp = rprod * rinvz_1 - iprod * iinvz_1;
		const realtype itmp = rprod * iinvz_1 + iprod * rinvz_1;
		
		rprod = rtmp;
		iprod = itmp;	

		rsum -= (j + 1) * (rxp[j] * rprod - ixp[j] * iprod);
		isum -= (j + 1) * (rxp[j] * iprod + ixp[j] * rprod);
	    }
	    
	    *xforce += rsum;
	    *yforce += -isum;
	}
	else
	    if (!node->state.innernode)
	    {
		const int s = node->s;
		    
		realtype xsum = 0, ysum = 0;
		for(int i = s; i < node->e; ++i)
		{
		    const realtype xr = xt - Tree::host_xdata[i];
		    const realtype yr = yt - Tree::host_ydata[i];
		    const realtype factor = Tree::host_vdata[i] / (xr * xr + yr * yr + eps);

		    xsum += xr * factor;
		    ysum += yr * factor;
		}
    
		*xforce += xsum;
		*yforce += ysum;
	    }
	    else
	    {
		for(int c = 0; c < 4; ++c)
		    stack[++stackentry] = node->state.childbase + c;
		    
		if (maxentry < stackentry)
		    maxentry = stackentry;
	    }
    }
}

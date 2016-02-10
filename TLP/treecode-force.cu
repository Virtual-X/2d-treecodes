/*
 *  TLP/treecode-force.cu
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
#include "force-downward-kernels.h"
#include "upward.h"

#define ACCESS(x) __ldg(&(x)) 

namespace EvaluateForce
{
    enum { ntiles1d = BLOCKSIZE / 8 };

    struct SharedBuffers
    {
	int stack[LMAX * 3];
	int e2lbuffer[32];
	realtype localxp[2 * (ORDER + 1)];
    };
    
#ifndef NDEBUG
    __constant__ int nnodes;
#endif
    __constant__ Tree::Node * nodes;
    __constant__ realtype * expansions, *xdata, *ydata, *vdata;

    __global__ void  __launch_bounds__(128, 16)
    evaluate(const realtype theta,
	     const realtype thetasquared,
	     const realtype * const x0s,
	     const realtype * const y0s,
	     const realtype * const hs,
	     realtype * const xforce, realtype * const yforce)
    {
	assert(blockDim.x == 32);
	
	const int tid = threadIdx.x;
	const bool master = tid == 0;
	
	const int blockid = blockIdx.y; 
	const int tileid = threadIdx.y + 4 * blockIdx.x;
	const int ix0 = 8 * (tileid % ntiles1d);
	const int iy0 = 8 * (tileid / ntiles1d);
	const int tx = tid % 8;
	const int ty = tid / 8;
	assert(ix0 + tx < BLOCKSIZE && iy0 + ty < BLOCKSIZE);

	const realtype h = hs[blockid];
	const realtype x0 = x0s[blockid] + h * ix0;
	const realtype y0 = y0s[blockid] + h * iy0;
	const realtype rbrick = 1.4142135623730951 * h * (8 - 1) * 0.5;
	const realtype xbrick = x0 + 3.5 * h;
	const realtype ybrick = y0 + 3.5 * h;

	extern __shared__ SharedBuffers ary[];

	int * const stack = ary[threadIdx.y].stack;	
	int stackentry = 0, maxentry = 0;

	int * const e2lbuf = ary[threadIdx.y].e2lbuffer;
	int e2lcount = 0;

	realtype * const lxp = ary[threadIdx.y].localxp;

	for(int i = tid; i < 2 * (ORDER + 1); i += 32)
	    lxp[i] = 0;

	if (master)
	    stack[0] = 0;
	
	realtype xsum0 = 0, xsum1 = 0, ysum0 = 0, ysum1 = 0;
	
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
	    
	    const realtype xdistance = xbrick - xcom;
	    const realtype ydistance = ybrick - ycom;
	    const realtype distance = sqrt(xdistance * xdistance + ydistance * ydistance);
	    const bool localexpansion_converges = (distance / r - 1) > (1 / theta) && rbrick <= r;

	    if (localexpansion_converges)
	    {
		if (master)
		    e2lbuf[e2lcount] = nodeid;

		if (++e2lcount == 32)
		{
		    e2lcount = 0;

		    const int mynodeid = e2lbuf[tid];
		    const Tree::Node * mynode = nodes + mynodeid;
		    const realtype myxcom = ACCESS(mynode->xcom);
		    const realtype myycom = ACCESS(mynode->ycom);
		    const realtype mymass = ACCESS(mynode->mass);

		    const realtype * rxp = expansions + ORDER * (0 + 2 * mynodeid);
                    const realtype * ixp = expansions + ORDER * (1 + 2 * mynodeid);

		    force_downward_e2l(myxcom - xbrick, myycom - ybrick, mymass, 
				       rxp, ixp, lxp, lxp + ORDER + 1);
		}
	    }
	    else
	    {
		const double xt = max(x0, min(x0 + 7 * h, xcom));
		const double yt = max(y0, min(y0 + 7 * h, ycom));
		
		const realtype rx = xt - xcom;
		const realtype ry = yt - ycom;
		const realtype r2 = rx * rx + ry * ry;
	    
		if (r * r < thetasquared * r2)
		{
		    const realtype mass = ACCESS(node->mass);
		    const realtype * rxp = expansions + ORDER * (0 + 2 * nodeid);
		    const realtype * ixp = expansions + ORDER * (1 + 2 * nodeid);
		    
		    force_e2p(mass, rxp, ixp, x0 + tx * h - xcom, y0 + ty * h - ycom, xsum0, ysum0);
		    force_e2p(mass, rxp, ixp, x0 + tx * h - xcom, y0 + (ty + 4) * h - ycom, xsum1, ysum1);
		}
		else 
		{
		    if (!node->state.innernode)
		    {
			const int s = node->s;
			const int count = node->e - s;
			
			force_p2p(xdata + s, ydata + s, vdata + s, count, x0 + tx * h, y0 + ty * h, xsum0, ysum0);
			force_p2p(xdata + s, ydata + s, vdata + s, count, x0 + tx * h, y0 + (ty + 4) * h, xsum1, ysum1);
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
	}

	if (tid < e2lcount)
	{
	    const int mynodeid = e2lbuf[tid];
	    const Tree::Node * mynode = nodes + mynodeid;

	    const realtype myxcom = mynode->xcom;
	    const realtype myycom = mynode->ycom;
	    const realtype mymass = mynode->mass;
	    
	    const realtype * rxp = expansions + ORDER * (0 + 2 * mynodeid);
	    const realtype * ixp = expansions + ORDER * (1 + 2 * mynodeid);
	    
	    force_downward_e2l(myxcom - xbrick, myycom - ybrick, mymass,
			       rxp, ixp, lxp, lxp + ORDER + 1);
	}

	force_downward_l2p(x0 + tx * h - xbrick, y0 + ty * h - ybrick, lxp, lxp + ORDER + 1, xsum0, ysum0);
	force_downward_l2p(x0 + tx * h - xbrick, y0 + (ty + 4) * h - ybrick, lxp, lxp + ORDER + 1, xsum1, ysum1);

	const int entry = (ix0 + tx) + BLOCKSIZE * (iy0 + ty) + BLOCKSIZE * BLOCKSIZE * blockid;
	assert(entry + 4 * BLOCKSIZE < gridDim.y * BLOCKSIZE * BLOCKSIZE);

	xforce[entry] = xsum0;
	yforce[entry] = ysum0;
	xforce[entry + 4 * BLOCKSIZE] = xsum1;
	yforce[entry + 4 * BLOCKSIZE] = ysum1;
    }
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
    const int xgridsize = (ntiles1d * ntiles1d) / yblocksize;
    assert((ntiles1d * ntiles1d) % yblocksize == 0);

    CUDA_CHECK(cudaMemset(device_xforce, 0xff, sizeof(realtype) * ndst));
    CUDA_CHECK(cudaMemset(device_yforce, 0xff, sizeof(realtype) * ndst));

    evaluate<<<dim3(xgridsize, nblocks), dim3(32, yblocksize), sizeof(SharedBuffers) * yblocksize>>>(
	theta, thetasquared, device_x0s, device_y0s, device_hs, device_xforce, device_yforce);

    CUDA_CHECK(cudaPeekAtLastError());
  
    CUDA_CHECK(cudaMemcpyAsync(xforce, device_xforce, sizeof(realtype) * ndst, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(yforce, device_yforce, sizeof(realtype) * ndst, cudaMemcpyDeviceToHost));
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

#ifndef NDEBUG

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
#endif
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

#include <omp.h>
#include <algorithm>
#include <limits>
#include <utility>

#include "upward.h"
#include "sort-sources.h"
#include "upward-kernels.h"
#include "cuda-common.h"

#define WARPSIZE 32
#define MFENCE
//__threadfence()
#define NQUEUES 4
#define LQSIZE 16

namespace Tree
{
    texture<int, cudaTextureType1D> texKeys;

    __device__ int lower_bound(int s, int e, const int val)
    {
	int c = e - s;

	if (tex1Dfetch(texKeys, s) >= val)
	    return 0;

	if (tex1Dfetch(texKeys, e - 1) < val)
	    return e;

	while (c)
	{
	    int candidate_s = s, candidate_e = e;

	    const float h = (e - s) * 1.f/ WARPSIZE;
	    const int i = min(e - 1, (int)(s + threadIdx.x * h + 0.499999f));

	    const bool isless = tex1Dfetch(texKeys, i) < val;
	    candidate_s = isless ? i : s;
	    candidate_e = isless ? e : i;

#pragma unroll
	    for(int mask = WARPSIZE / 2 ; mask > 0 ; mask >>= 1)
	    {
		candidate_s = max(candidate_s, __shfl_xor(candidate_s, mask));
		candidate_e = min(candidate_e, __shfl_xor(candidate_e, mask));
	    }

	    s = candidate_s;
	    e = candidate_e;
	    c = min(c / 32, e - s);
	}

	return s + 1;
    }

    __device__ int upper_bound(int s, int e, const int val)
    {
	int c = e - s;

	if (tex1Dfetch(texKeys, s) > val)
	    return 0;

	if (tex1Dfetch(texKeys, e - 1) <= val)
	    return e;

	while (c)
	{
	    int candidate_s = s, candidate_e = e;

	    const float h = (e - s) * 1.f / WARPSIZE;
	    const int i = min(e - 1, (int)(s + threadIdx.x * h + 0.499999f));

	    const bool isless = tex1Dfetch(texKeys, i) <= val;
	    candidate_s = isless ? i : s;
	    candidate_e = isless ? e : i;

#pragma unroll
	    for(int mask = WARPSIZE / 2 ; mask > 0 ; mask >>= 1)
	    {
		candidate_s = max(candidate_s, __shfl_xor(candidate_s, mask));
		candidate_e = min(candidate_e, __shfl_xor(candidate_e, mask));
	    }

	    s = candidate_s;
	    e = candidate_e;
	    c = min(c / 32, e - s);
	}

	return s + 1;
    }

    __constant__ realtype *xsorted, *ysorted, *vsorted;

    struct NodeHelper
    {
	int x, y, l, mask, parent, validchildren;
	realtype w, wx, wy;

	__device__ void setup(int x, int y, int l, int mask, int parent)
	    {
		this->x = x;
		this->y = y;
		this->l = l;
	
		this->mask = mask;
		this->parent = parent;
		this->validchildren = 0;
	    }
    };

//TREE info
    __constant__ int bufsize;
    __device__ int currnnodes;
    
    __constant__ Node * bufnodes;
    __constant__ NodeHelper * bufhelpers;
    __constant__ realtype * bufexpansion;

    __device__ void process_leaf(const int nodeid, realtype extent)
    {
	const int tid = threadIdx.x;
	const bool master = tid == 0;

	Node * node = bufnodes + nodeid;
	NodeHelper * helper = bufhelpers + nodeid;

	const int s = node->s;
	const int e = node->e;

	realtype msum = 0, wsum = 0, wxsum = 0, wysum = 0;

	for(int t = s + tid; t < e; t += WARPSIZE)
	{
	    const realtype x = xsorted[t];
	    const realtype y = ysorted[t];
	    const realtype m = vsorted[t];
	    const realtype w = fabs(m);

	    msum += m;
	    wsum += w;
	    wxsum += x * w;
	    wysum += y * w;
	}

#pragma unroll
	for(int mask = WARPSIZE / 2 ; mask > 0 ; mask >>= 1)
	{
	    msum += __shfl_xor(msum, mask);
	    wsum += __shfl_xor(wsum, mask);
	    wxsum += __shfl_xor(wxsum, mask);
	    wysum += __shfl_xor(wysum, mask);
	}

	const realtype xcom = wsum ? wxsum / wsum : 0;
	const realtype ycom = wsum ? wysum / wsum : 0;

	upward_p2e(xcom, ycom,
		   xsorted + s, ysorted + s, vsorted + s, e - s,
		   bufexpansion + ORDER * (2 * nodeid + 0),
		   bufexpansion + ORDER * (2 * nodeid + 1));

	realtype r2 = 0;
	for(int i = s + tid; i < e; i += WARPSIZE)
	{
	    const realtype xr = xsorted[i] - xcom;
	    const realtype yr = ysorted[i] - ycom;

	    r2 = max(r2, xr * xr + yr * yr);
	}

#pragma unroll
	for(int mask = WARPSIZE / 2 ; mask > 0 ; mask >>= 1)
	{
	    const realtype other_r2 = __shfl_xor(r2, mask);
	    r2 = max(r2, other_r2);
	}

	if (master)
	{
	    helper->w = wsum;
	    helper->wx = wxsum;
	    helper->wy = wysum;

	    node->mass = msum;
	    node->r = sqrt(r2);
	    node->xcom = xcom;
	    node->ycom = ycom;
	    	    
	    MFENCE;
	}

	while(helper->parent >= 0)
	{
	    Node * parent = bufnodes + helper->parent;
	    NodeHelper * parenthelper = bufhelpers + helper->parent;
	    
	    bool e2e = false;

	    if (master)
		e2e = 3 == atomicAdd(&parenthelper->validchildren, 1);

	    e2e = __shfl(e2e, 0);

	    if (e2e)
	    {
		realtype xcom_parent, ycom_parent;

		if (master)
		{
		    realtype msum = 0, wsum = 0, wxsum = 0, wysum = 0;
		    for(int c = 0; c < 4; ++c)
		    {
			const int childid = parent->state.children[c];

			const Node * child = bufnodes + childid;
			msum += child->mass;

			const NodeHelper * childhelper = bufhelpers + childid;
			wsum += childhelper->w;
			wxsum += childhelper->wx;
			wysum += childhelper->wy;
		    }

		    parent->mass = msum;
		    parenthelper->w = wsum;
		    parenthelper->wx = wxsum;
		    parenthelper->wy = wysum;

		    assert(wsum);
		    xcom_parent = wxsum / wsum;
		    ycom_parent = wysum / wsum;

		    realtype rr = 0;
		    for(int c = 0; c < 4; ++c)
		    {
			const int childid = parent->state.children[c];
			const Node * child = bufnodes + childid;
			const NodeHelper * childhelper = bufhelpers + childid;

			if (childhelper->w)
			{
			    const realtype rx = xcom_parent - child->xcom;
			    const realtype ry = ycom_parent - child->ycom;

			    rr = max(rr, child->r + sqrt(rx * rx + ry * ry));
			}
		    }

		    parent->r = min(rr, 1.4143f * extent / (1 << parenthelper->l));
		    parent->xcom = xcom_parent;
		    parent->ycom = ycom_parent;
		}

		xcom_parent = __shfl(xcom_parent, 0);
		ycom_parent = __shfl(ycom_parent, 0);

		if (tid < 4)
		{
		    const int childid = parent->state.children[tid];
		    const Node * chd = bufnodes + childid;

		    upward_e2e(chd->xcom - xcom_parent, chd->ycom - ycom_parent, chd->mass,
			       bufexpansion + ORDER * (2 * childid + 0),
			       bufexpansion + ORDER * (2 * childid + 1),
			       bufexpansion + ORDER * (2 * helper->parent + 0),
			       bufexpansion + ORDER * (2 * helper->parent + 1));
		}

		if (master)
		    MFENCE;
	    }
	    else
		break;

	    node = parent;
	    helper = parenthelper;
	}
    }

//QUEUE info
    __constant__ int queuesize, * queues[NQUEUES];
    __device__  int qlock[NQUEUES], qhead[NQUEUES], qtail[NQUEUES], qtailnext[NQUEUES], qitems;
    __device__ bool qgood;

    __global__ void setup(const int nsrc)
    {
	currnnodes = 1;
	bufnodes[0].setup(0, nsrc);
	bufhelpers[0].setup(0, 0, 0, 0, -1);

	for(int i = 0; i < NQUEUES; ++i)
	{
	    qlock[i] = 1;
	    qhead[i] = 0;
	    qtail[i] = 0;
	    qtailnext[i] = 0;
	}

	const int qid = 0;
	queues[qid][0] = 0;
	qtail[qid] = 1;
	qtailnext[qid] = 1;

	qitems = 1;
	qgood = true;
    }

    __global__ void build_tree(const int LEAF_MAXCOUNT, const double extent)
    {
	assert(blockDim.x == warpSize && WARPSIZE == warpSize);

#if LQSIZE > 0
	__shared__ int ltasks[LQSIZE];

	{
	    const int tid2d = threadIdx.x + blockDim.x * threadIdx.y;

	    for(int i = tid2d; i < LQSIZE; i += blockDim.x * blockDim.y)
		ltasks[i] = -1;

	    __syncthreads();
	}
#endif

	const int tid = threadIdx.x;
	const bool master = tid == 0;

	int currid = -1;

	int iteration = -1;

	while(qitems && qgood)
	{
	    const int qid = (++iteration + blockIdx.x) % NQUEUES;

	    if (currid == -1)
	    {
		if (master)
		{
#if LQSIZE > 0
		    //get a task from the local pool if possible
		    for(int i = 0; i < LQSIZE && currid == -1; ++i)
			currid = atomicExch(ltasks + i, -1);
#endif

		    //or take one task from the global queues (if any)
		    if (currid == -1)
			if (atomicCAS(&qlock[qid], 1, 0)) 
			{
			    const int currhead = qhead[qid];

			    if (currhead < qtail[qid])
			    {
				const int entry = currhead % queuesize;

				currid = queues[qid][entry];

				qhead[qid] = currhead + 1;

				MFENCE;
			    }

			    qlock[qid] = 1;
			}
		}

		currid = __shfl(currid, 0);
	    }

	    if (currid >= 0)
	    {
		Node * node = bufnodes + currid;
		NodeHelper * helper = bufhelpers + currid;
		
		const int s = node->s;
		const int e = node->e;
		const int l = helper->l;

		const bool leaf = e - s <= LEAF_MAXCOUNT || l + 1 > LMAX;

		if (leaf)
		{
		    process_leaf(currid, extent);

		    if (master)
			atomicSub(&qitems, 1);

		    currid = -1;
		}
		else
		{
		    if (master) //children allocation
		    {
			const int bufbase = atomicAdd(&currnnodes, 4);

			if (bufbase + 4 > bufsize)
			{
			    qgood = false;
			    break;
			}

			for(int c = 0; c < 4; ++c)
			    node->state.children[c] = bufbase + c;
		    }

		    const int mask = helper->mask;
		    const int x = helper->x;
		    const int y = helper->y;

		    for(int c = 0; c < 4; ++c)
		    {
			const int shift = 2 * (LMAX - l - 1);

			const int key1 = mask | (c << shift);
			const int key2 = key1 + (1 << shift) - 1;

			const size_t indexmin = c == 0 ? s : lower_bound(s, e, key1);
			const size_t indexsup = c == 3 ? e : upper_bound(s, e, key2);

			if (master)
			{
			    const int childid = node->state.children[c];
			    Node * child = bufnodes + childid;
			    NodeHelper * childhelper = bufhelpers + childid;

			    child->setup(indexmin, indexsup);
			    childhelper->setup((x << 1) + (c & 1), (y << 1) + (c >> 1), l + 1, key1, currid);
			}
		    }

		    if (master) //enqueue new tasks
		    {
			bool placed_locally = false;

			const int localtask = node->state.children[2];
			
#if (LQSIZE > 0)
			//try to put a task in the local queue
			for(int i = 0; i < LQSIZE && !placed_locally; ++i)
			    placed_locally = atomicCAS(ltasks + i, -1, localtask) == -1;
#endif
			const int ngtasks = 3 - placed_locally;

			const int base = atomicAdd(&qtailnext[qid], ngtasks);

			if (base + ngtasks - qhead[qid] >= queuesize)
			{
			    qgood = false;
			    break;
			}
			else
			{
			    for(int c = 0; c < ngtasks; ++c)
				queues[qid][(base + c) % queuesize] = node->state.children[c];

			    atomicAdd(&qitems, 3);

			    MFENCE;

			    atomicAdd(&qtail[qid], ngtasks);
			}

			currid = node->state.children[3];
		    }
		}

		currid = __shfl(currid, 0);
	    }
	}
    }

    struct BuildResult
    {
	int ntreenodes, queuesize, nqueueitems;
	bool good;
    };

    __global__ void conclude(BuildResult * result)
    {
	result->ntreenodes = currnnodes;
	result->queuesize = qtail - qhead;
	result->nqueueitems = qitems;
	result->good = qgood;
    }

    realtype xmin, ymin, extent, *host_xdata, *host_ydata, *host_vdata;
    
    realtype * host_expansions = NULL, *device_expansions = NULL;
    Node * host_nodes = NULL;
    int nnodes = 0;
    
    realtype *device_xdata = NULL, *device_ydata = NULL, *device_vdata = NULL;
    Node * device_nodes = NULL;
    NodeHelper * device_helpers = NULL;
    int * device_keys = NULL;

    cudaStream_t stream = 0;
    cudaEvent_t evstart, evstop;

    int * device_queue;
    BuildResult * device_diag;
}

namespace TreeCheck
{
    void verify_all(const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc, const int LEAF_MAXCOUNT);
}

void Tree::build(const realtype * const xsrc,
		 const realtype * const ysrc,
		 const realtype * const vsrc,
		 const int nsrc,
		 const int LEAF_MAXCOUNT)
{
    //CUDA_CHECK(cudaFuncSetCacheConfig(build_tree, cudaFuncCachePreferL1) );
    
    texKeys.channelDesc = cudaCreateChannelDesc<int>();
    texKeys.filterMode = cudaFilterModePoint;
    texKeys.mipmapFilterMode = cudaFilterModePoint;
    texKeys.normalized = 0;

    CUDA_CHECK(cudaEventCreate(&evstart));
    CUDA_CHECK(cudaEventCreate(&evstop));
    
    int nsmxs = -1;
    CUDA_CHECK(cudaDeviceGetAttribute (&nsmxs, cudaDevAttrMultiProcessorCount, 0));
    printf("i have found %d smxs\n", nsmxs);
   
    const int device_queuesize = 8e4;
    const int device_bufsize = 8e4;

    CUDA_CHECK(cudaMalloc(&device_queue, sizeof(*device_queue) * device_queuesize * NQUEUES));
    CUDA_CHECK(cudaMalloc(&device_nodes, sizeof(*device_nodes) * device_bufsize));
    CUDA_CHECK(cudaMalloc(&device_helpers, sizeof(*device_helpers) * device_bufsize));    
    CUDA_CHECK(cudaMalloc(&device_expansions, sizeof(realtype) * ORDER * 2 * device_bufsize));
    CUDA_CHECK(cudaMalloc(&device_xdata, sizeof(realtype) * nsrc));
    CUDA_CHECK(cudaMalloc(&device_ydata, sizeof(realtype) * nsrc));
    CUDA_CHECK(cudaMalloc(&device_vdata, sizeof(realtype) * nsrc));
    CUDA_CHECK(cudaMalloc(&device_keys, sizeof(int) * nsrc));
    
    CUDA_CHECK(cudaMallocHost(&device_diag, sizeof(*device_diag)));
   
    size_t textureoffset = 0;
    CUDA_CHECK(cudaBindTexture(&textureoffset, &texKeys, device_keys, &texKeys.channelDesc, sizeof(int) * nsrc));
    assert(textureoffset == 0);

    CUDA_CHECK(cudaPeekAtLastError());

    CUDA_CHECK(cudaMemcpyToSymbolAsync(xsorted, &device_xdata, sizeof(device_xdata)));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(ysorted, &device_ydata, sizeof(device_ydata)));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(vsorted, &device_vdata, sizeof(device_vdata)));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(bufsize, &device_bufsize, sizeof(device_bufsize)));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(bufnodes, &device_nodes, sizeof(device_nodes)));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(bufhelpers, &device_helpers, sizeof(device_helpers)));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(bufexpansion, &device_expansions, sizeof(device_expansions)));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(queuesize, &device_queuesize, sizeof(device_queuesize)));

    {
	int * ptrs[NQUEUES];
	for(int i = 0; i < NQUEUES; ++i)
	    ptrs[i] = device_queue + device_queuesize * i;

	CUDA_CHECK(cudaMemcpyToSymbolAsync(queues, &ptrs, sizeof(ptrs)));
    }

    CUDA_CHECK(cudaMemcpyAsync(device_xdata, xsrc, sizeof(realtype) * nsrc, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(device_ydata, ysrc, sizeof(realtype) * nsrc, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(device_vdata, vsrc, sizeof(realtype) * nsrc, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(evstart));
    
    sort_sources(stream, device_xdata, device_ydata, device_vdata, nsrc, device_keys, &xmin, &ymin, &extent);

    setup<<<1, 1>>>(nsrc);

    const int ysize = 16;
    build_tree<<<nsmxs * 2, dim3(32, ysize), sizeof(realtype) * 4 * 4 * ORDER * ysize>>>(LEAF_MAXCOUNT, extent);
    CUDA_CHECK(cudaPeekAtLastError());

    conclude<<<1, 1>>>(device_diag);
    CUDA_CHECK(cudaPeekAtLastError());

    CUDA_CHECK(cudaEventRecord(evstop));
 
#ifndef NDEBUG
    TreeCheck::verify_all(xsrc, ysrc, vsrc, nsrc, LEAF_MAXCOUNT);
#endif
} 

void Tree::dispose()
{   
    CUDA_CHECK(cudaEventSynchronize(evstop));

    float timems;
    CUDA_CHECK(cudaEventElapsedTime(&timems, evstart,evstop  ));
    printf("\x1B[33mtimems: %f\x1b[0m\n", timems);

    printf("device has found %d nodes, and max queue size was %d, outstanding items %d, queue is good: %d\n",
	   device_diag->ntreenodes, device_diag->queuesize, device_diag->nqueueitems, device_diag->good);

    CUDA_CHECK(cudaFree(device_xdata));
    CUDA_CHECK(cudaFree(device_ydata));
    CUDA_CHECK(cudaFree(device_vdata));
    CUDA_CHECK(cudaFree(device_nodes));
    CUDA_CHECK(cudaFree(device_expansions));
    CUDA_CHECK(cudaFree(device_keys));
    CUDA_CHECK(cudaFree(device_helpers));
    CUDA_CHECK(cudaEventDestroy(evstart));
    CUDA_CHECK(cudaEventDestroy(evstop));
    CUDA_CHECK(cudaFree(device_queue));
    CUDA_CHECK(cudaFreeHost(device_diag));

#ifndef NDEBUG
    CUDA_CHECK(cudaFreeHost(host_nodes));
    CUDA_CHECK(cudaFreeHost(host_expansions));
    free(host_xdata);
    free(host_ydata);
    free(host_vdata);
#endif
}

namespace TreeCheck
{   
    int LEAF_MAXCOUNT;

    int * debug_keys = NULL;

    struct DebugNode
    {
	int x, y, l, s, e;
	bool leaf;
	realtype w, wx, wy, mass, r;

	DebugNode * children[4];

	void setup(int x, int y, int l, int s, int e, bool leaf)
	    {
		this->x = x;
		this->y = y;
		this->l = l;
		this->s = s;
		this->e = e;
		this->leaf = leaf;

	    }

	realtype xcom() const { return wx / w; }
	realtype ycom() const { return wy / w; }

	DebugNode() 
	    { 
		for (int i = 0; i < 4; ++i) 
		    children[i] = nullptr; 

		w = wx = wy = mass = r = 0; 
	    }
		
	typedef realtype alignedvec[ORDER] __attribute__ ((aligned (32)));

	alignedvec rexpansions;
	alignedvec iexpansions;

	void allocate_children()
	    {
		for(int i = 0; i < 4; ++i)
		    children[i] = new DebugNode;
	    }
	
	realtype * rexp(){return &rexpansions[0];} 
	realtype * iexp(){return &iexpansions[0];}
	
	void p2e(const realtype * __restrict__ const xsources,
		 const realtype * __restrict__ const ysources,
		 const realtype * __restrict__ const vsources,
		 const double x0, const double y0, const double h)
	    {
		reference_upward_p2e(xsources, ysources, vsources, e - s,
				     x0, y0, h, &mass, &w, &wx, &wy, &r,
				     rexpansions, iexpansions);
	    }
	void e2e()
	    {
		realtype srcmass[4], rx[4], ry[4];
		realtype * chldrxp[4], *chldixp[4];

		for(int c = 0; c < 4; ++c)
		{
		    DebugNode * chd = children[c];

		    srcmass[c] = chd->mass;
		    rx[c] = chd->xcom() - xcom();
		    ry[c] = chd->ycom() - ycom();
		    chldrxp[c] = chd->rexpansions;
		    chldixp[c] = chd->iexpansions;
		}

		reference_upward_e2e(rx, ry, srcmass, chldrxp, chldixp, rexpansions, iexpansions);
#ifndef NDEBUG
		{
		    for(int i = 0; i < ORDER; ++i)
			assert(!std::isnan((double)rexpansions[i]) && !std::isnan(iexpansions[i]));
		}
#endif
	    }

	~DebugNode() 
	    {
		for(int i = 0; i < 4; ++i)
		    if (children[i])
		    {
			delete children[i];

			children[i] = nullptr;
		    }
	    }
    };

    DebugNode * debugroot = NULL;

    void _build(DebugNode * const node, const int x, const int y, const int l, const int s, const int e, const int mask)
    {
	const double h = Tree::extent / (1 << l);
	const double x0 = Tree::xmin + h * x, y0 = Tree::ymin + h * y;

	assert(x < (1 << l) && y < (1 << l) && x >= 0 && y >= 0);
	//printf("node %d %d l%d\n", x, y, l);
	
	for(int i = s; i < e; ++i)
	    assert(Tree::host_xdata[i] >= x0 && Tree::host_xdata[i] < x0 + h && Tree::host_ydata[i] >= y0 && Tree::host_ydata[i] < y0 + h);
	
	node->setup(x, y, l, s, e, e - s <= LEAF_MAXCOUNT || l + 1 > LMAX);

	if (node->leaf)
	{
	    node->p2e(&Tree::host_xdata[s], &Tree::host_ydata[s], &Tree::host_vdata[s], x0, y0, h);

	    assert(node->r < 1.5 * h);
	}
	else
	{
	    node->allocate_children();

	    for(int c = 0; c < 4; ++c)
	    {
		const int shift = 2 * (LMAX - l - 1);

		const int key1 = mask | (c << shift);
		const int key2 = key1 + (1 << shift) - 1;

		const size_t indexmin = c == 0 ? s : std::lower_bound(debug_keys + s, debug_keys + e, key1) - debug_keys;
		const size_t indexsup = c == 3 ? e : std::upper_bound(debug_keys + s, debug_keys + e, key2) - debug_keys;

		DebugNode * chd = node->children[c];

		_build(chd, (x << 1) + (c & 1), (y << 1) + (c >> 1), l + 1, indexmin, indexsup, key1);
	    }

	    for(int c = 0; c < 4; ++c)
	    {
		DebugNode * chd = node->children[c];
		node->mass += chd->mass;
		node->w += chd->w;
		node->wx += chd->wx;
		node->wy += chd->wy;

		node->children[c] = chd;
	    }

	    node->r = 0;
	    for(int c = 0; c < 4; ++c)
		if (node->children[c]->w)
		    node->r = std::max(node->r,
				       node->children[c]->r +
				       sqrt(pow(node->xcom() - node->children[c]->xcom(), 2) +
					    pow(node->ycom() - node->children[c]->ycom(), 2)));

	    node->r = std::min(node->r, 1.4143 * h);

	    assert(node->r < 1.5 * h);

	    {
		realtype r = 0;

		for(int i = s; i < e; ++i)
		    r = std::max(r, pow(Tree::host_xdata[i] - node->xcom(), (realtype)2) + pow(Tree::host_ydata[i] - node->ycom(), (realtype)2));

		assert (sqrt(r) <= node->r);
	    }

	    node->e2e();
	}

	assert(node->xcom() >= x0 && node->xcom() < x0 + h && node->ycom() >= y0 && node->ycom() < y0 + h || node->e - node->s == 0);
    }

    bool verbose = false;

    int check_bits(double x, double y)
    {
	union ASD
	{
	    unsigned char c[8];
	    double d;
	};

	ASD a, b;
	a.d = x;
	b.d = y;

	int currbit = 0;
	for(int i = 0; i < 8; ++i)
	{
	    unsigned char c1 = a.c[7 - i], c2 = b.c[7 - i];

	    for(int b = 0; b < 8; ++b, ++currbit)
	    {
		if (((c1 >> b) & 1) != ((c2 >> b) & 1))
		{
		    if (verbose) printf("numbers differ from the %d most-significant bit\n", currbit);
		    return currbit;
		}
	    }
	}

	return currbit;
    }

    int check_bits(const double *a , const double *b, const int n)
    {
	if (verbose) printf("*******************************\n");
	int r = 64;

	for(int i = 0; i < n; ++i)
	{
	    if (fabs(a[i]) > 1.11e-16 || fabs(b[i]) > 1.11e-16)
	    {
		int l = check_bits(a[i], b[i]);
		const double x = a[i];
		const double y = b[i];
		if (l < 48 )
		    if (verbose) printf("strange case of %+.20e vs %+.20e relerr %e\n", x, y, (x - y) / y);
		r= min(r, l );
	    }
	}

	if (verbose) printf("********** end ***************\n");
	return r;
    }

    void check_tree (const int EXPORD, const int nodeid, realtype * allexp, Tree::Node * allnodes, Tree::Node& a, TreeCheck::DebugNode& b)
    {
	//assert(a.x == b.x);
	//assert(a.y == b.y);
	//assert(a.l == b.l);
	assert(a.s == b.s);
	assert(a.e == b.e);
	//assert(a.mask == b.mask);
	if (verbose)
	{
	    printf("<%s>", (b.leaf ? "LEAF" : "INNER"));
	    printf("ASDnode %d %d l%d s: %d e: %d. check passed..\n", b.x, b.y, b.l, b.s, b.e);
/*
  printf("a/ m-w-wx-wy-r: %.20e %.20e %.20e %.20e %.20e\n",
  a.mass, a.w, a.wx, a.wy, a.r);
  printf("b/ m-w-wx-wy-r: %.20e %.20e %.20e %.20e %.20e\n",
  b.mass, b.w, b.wx, b.wy, b.r);*/
	}
	assert(check_bits(a.mass, b.mass) >= 40);
	//assert(check_bits(a.w, b.w) >= 40);
	assert(check_bits(a.xcom, b.wx / b.w) >= 40 || b.w == 0);
	assert(check_bits(a.ycom, b.wy / b.w) >= 40 || b.w == 0);
	
	//assert(check_bits(a.wx, b.wx) >= 40 || a.w == 0);
	//assert(check_bits(a.wy, b.wy) >= 40 || a.w == 0);

	assert(check_bits(a.r, b.r) >= 32);
#ifndef NDEBUG
	{
	    const realtype * resrexp = allexp + EXPORD * (2 * nodeid + 0);
	    const realtype * resiexp = allexp + EXPORD * (2 * nodeid + 1);
	    const realtype * refrexp = b.rexp();
	    const realtype * refiexp = b.iexp();
	    assert(24 <= check_bits(resrexp, refrexp, EXPORD));
	    assert(24 <= check_bits(resiexp, refiexp, EXPORD));
	}
#endif

	if (!b.leaf)
	    for(int c = 0; c < 4; ++c)
		check_tree(EXPORD, a.state.children[c], allexp, allnodes, allnodes[a.state.children[c]], *b.children[c]);
    }

    void verify_all(const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc, const int LEAF_MAXCOUNT)
    {
	CUDA_CHECK(cudaStreamSynchronize(0));
	
	Tree::nnodes = Tree::device_diag->ntreenodes;
	const size_t expansionsbytes = sizeof(realtype) * 2 * ORDER * Tree::nnodes;
	
    
	posix_memalign((void **)&Tree::host_xdata, 32, sizeof(realtype) * nsrc);
	posix_memalign((void **)&Tree::host_ydata, 32, sizeof(realtype) * nsrc);
	posix_memalign((void **)&Tree::host_vdata, 32, sizeof(realtype) * nsrc);
	
	CUDA_CHECK(cudaMemcpy(Tree::host_xdata, Tree::device_xdata, sizeof(realtype) * nsrc, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(Tree::host_ydata, Tree::device_ydata, sizeof(realtype) * nsrc, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(Tree::host_vdata, Tree::device_vdata, sizeof(realtype) * nsrc, cudaMemcpyDeviceToHost));
	
	CUDA_CHECK(cudaMallocHost(&Tree::host_expansions, expansionsbytes));
	CUDA_CHECK(cudaMallocHost(&Tree::host_nodes, sizeof(Tree::Node) * Tree::nnodes));
	CUDA_CHECK(cudaMemcpy(Tree::host_expansions, Tree::device_expansions, expansionsbytes, cudaMemcpyDeviceToHost));

	std::vector<Tree::NodeHelper> devhelpers(Tree::nnodes);
	CUDA_CHECK(cudaMemcpy(Tree::host_nodes, Tree::device_nodes, sizeof(Tree::Node) * Tree::nnodes, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&devhelpers.front(), Tree::device_helpers, sizeof(Tree::NodeHelper) * Tree::nnodes,
			      cudaMemcpyDeviceToHost));
    
	printf("VERIFICATION _______________________________________\n");
	
	TreeCheck::LEAF_MAXCOUNT = LEAF_MAXCOUNT;

       	CUDA_CHECK(cudaMallocHost(&debug_keys, sizeof(int) * nsrc));

	CUDA_CHECK(cudaMemcpy(debug_keys, Tree::device_keys, sizeof(int) * nsrc, cudaMemcpyDeviceToHost));
	
	std::pair<int, int> * kv = NULL;
	posix_memalign((void **)&kv, 32, sizeof(*kv) * nsrc);

	//assert(truexmin == *std::min_element(xsrc, xsrc + nsrc));
	//assert(trueymin == *std::min_element(ysrc, ysrc + nsrc));

	//assert(ext0 == *std::max_element(xsrc, xsrc + nsrc) - truexmin);
	//assert(ext1 == *std::max_element(ysrc, ysrc + nsrc) - trueymin);

	for(int i = 0; i < nsrc; ++i)
	{
	    int x = floor((xsrc[i] - Tree::xmin) / Tree::extent * (1 << LMAX));
	    int y = floor((ysrc[i] - Tree::ymin) / Tree::extent * (1 << LMAX));

	    assert(x >= 0 && y >= 0);
	    assert(x < (1 << LMAX) && y < (1 << LMAX));

	    x = (x | (x << 8)) & 0x00FF00FF;
	    x = (x | (x << 4)) & 0x0F0F0F0F;
	    x = (x | (x << 2)) & 0x33333333;
	    x = (x | (x << 1)) & 0x55555555;

	    y = (y | (y << 8)) & 0x00FF00FF;
	    y = (y | (y << 4)) & 0x0F0F0F0F;
	    y = (y | (y << 2)) & 0x33333333;
	    y = (y | (y << 1)) & 0x55555555;

	    const int key = x | (y << 1);

	    kv[i].first = key;
	    kv[i].second = i;
	}

	std::sort(kv, kv + nsrc);

	for(int i = 0; i < nsrc; ++i)
	{
	    //const int key = kv[i].first;

	    const int entry = kv[i].second;
	    assert(entry >= 0 && entry < nsrc);

	    assert(Tree::host_xdata[i] == xsrc[entry]);
	    assert(Tree::host_ydata[i] == ysrc[entry]);
	    assert(Tree::host_vdata[i] == vsrc[entry]);
	}

	printf("SORTING IS GOOD\n");
	
	free(kv);

	debugroot = new DebugNode;
	
	_build(debugroot, 0, 0, 0, 0, nsrc, 0);
    
	//const int nnodes = Tree::nnodes;
	//std::vector<Tree::Node> allnodes(nnodes);

	//CUDA_CHECK(cudaMemcpy(&allnodes.front(), Tree::device_bufnodes, sizeof(Tree::DeviceNode) * allnodes.size(), cudaMemcpyDeviceToHost));

	//std::vector<realtype> allexpansions(nnodes * ORDER * 2);
	//CUDA_CHECK(cudaMemcpy(&allexpansions.front(), device_bufexpansions, sizeof(realtype) * 2 * ORDER * nnodes, cudaMemcpyDeviceToHost));

	printf("rooot xylsem: %d %d, children %d %d %d %d\n",
	       Tree::host_nodes[0].s, Tree::host_nodes[0].e,
	       Tree::host_nodes[0].state.children[0],
	       Tree::host_nodes[0].state.children[1],
	       Tree::host_nodes[0].state.children[2],
	       Tree::host_nodes[0].state.children[3]);

	//ok let's check this
	check_tree(ORDER, 0, Tree::host_expansions,Tree::host_nodes, Tree::host_nodes[0], *debugroot);

	printf("TREE IS GOOD\n");
	printf("VERIFICATION SUCCEDED.______________________________\n");
	CUDA_CHECK(cudaFreeHost(debug_keys));
    }
}
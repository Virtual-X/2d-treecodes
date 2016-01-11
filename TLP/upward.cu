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
//#include <cmath>

#include <omp.h>
//#include <parallel/algorithm>
#include <algorithm>
#include <limits>
#include <utility>

#include "upward.h"
#include "upward-kernels.h"
#include "cuda-common.h"

#define  _INSTRUMENTATION_
#ifndef _INSTRUMENTATION_
#define MYRDTSC 0
#else
#define MYRDTSC _rdtsc()
#endif

#define LMAX 15

namespace Tree
{
    int LEAF_MAXCOUNT;

    realtype ext, xmin, ymin;

    int * keys = NULL;

    realtype *xdata = NULL, *ydata = NULL, *vdata = NULL;

    Node * root = NULL;

    void _build(Node * const node, const int x, const int y, const int l, const int s, const int e, const int mask)
    {
	const int64_t startallc = MYRDTSC;

	const double h = ext / (1 << l);
	const double x0 = xmin + h * x, y0 = ymin + h * y;

	assert(x < (1 << l) && y < (1 << l) && x >= 0 && y >= 0);

#ifndef NDEBUG
	for(int i = s; i < e; ++i)
	    assert(xdata[i] >= x0 && xdata[i] < x0 + h && ydata[i] >= y0 && ydata[i] < y0 + h);
#endif

	node->setup(x, y, l, s, e, e - s <= LEAF_MAXCOUNT || l + 1 > LMAX);

	if (node->leaf)
	{
	    const int64_t startc = MYRDTSC;
	    node->p2e(&xdata[s], &ydata[s], &vdata[s], x0, y0, h);
	    node->p2ecycles = MYRDTSC - startc;

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

		const int64_t startc = MYRDTSC;
		const size_t indexmin = c == 0 ? s : std::lower_bound(keys + s, keys + e, key1) - keys;
		const size_t indexsup = c == 3 ? e : std::upper_bound(keys + s, keys + e, key2) - keys;
		node->searchcycles += MYRDTSC - startc;

		Node * chd = node->children[c];

#pragma omp task firstprivate(chd, c, x, y, l, indexmin, indexsup, key1) if (indexsup - indexmin > 5e3 && c < 3)
		//if (c < 3 && l < 8)
		{
		    _build(chd, (x << 1) + (c & 1), (y << 1) + (c >> 1), l + 1, indexmin, indexsup, key1);
		}

	    }
//#pragma omp taskyield
#pragma omp taskwait

	    const int64_t startc = MYRDTSC;

	    for(int c = 0; c < 4; ++c)
	    {
		Node * chd = node->children[c];
		node->mass += chd->mass;
		node->w += chd->w;
		node->wx += chd->wx;
		node->wy += chd->wy;

		node->children[c] = chd;
	    }

	    //realtype rcandidates[4];
	    node->r = 0;
	    for(int c = 0; c < 4; ++c)
		node->r = std::max(node->r,
				   node->children[c]->r +
				   sqrt(pow(node->xcom() - node->children[c]->xcom(), 2) +
					pow(node->ycom() - node->children[c]->ycom(), 2)));

	    node->r = std::min(node->r, 1.4143 * h);

	    assert(node->r < 1.5 * h);

#ifndef NDEBUG
	    {
		realtype r = 0;

		for(int i = s; i < e; ++i)
		    r = std::max(r, pow(xdata[i] - node->xcom(), (realtype)2) + pow(ydata[i] - node->ycom(), (realtype)2));

		assert (sqrt(r) <= node->r);
	    }
#endif

	    node->e2e();
	    node->e2ecycles = MYRDTSC - startc;
	}

#ifndef NDEBUG
	{
	    assert(node->xcom() >= x0 && node->xcom() < x0 + h && node->ycom() >= y0 && node->ycom() < y0 + h || node->e - node->s == 0);
	}
#endif

	const int64_t endallc = MYRDTSC;
	node->allcycles = endallc - startallc;
    }

    template <class ForwardIterator, class T>
    __device__ ForwardIterator lower_bound (ForwardIterator first, ForwardIterator last, const T& val)
    {
	ForwardIterator it;
	int count, step;
	count = last - first; //distance(first,last);
	while (count>0)
	{
      
	    it = first; step=count/2; it += step; //advance (it,step);
	    // printf("step: %d\n", step);
	    if (*it<val) {                 // or: if (comp(*it,val)), for version (2)
		first=++it;
		count-=step+1;
	    }
	    else count=step;
	}
	return first;
    }

    template <class ForwardIterator, class T>
    __device__ ForwardIterator upper_bound (ForwardIterator first, ForwardIterator last, const T& val)
    {
	ForwardIterator it;
	int count, step;
	count = last - first;//std::distance(first,last);
	while (count>0)
	{
	    it = first; step=count/2; it += step;//std::advance (it,step);
	    if (!(val<*it))                 // or: if (!comp(val,*it)), for version (2)
	    { first=++it; count-=step+1;  }
	    else count=step;
	}
	return first;
    }
    
    __global__ void generate_keys(const realtype * const xsrc, const realtype * const ysrc, const int n,
				  const realtype xmin, const realtype ymin, const realtype ext,
				  int * const keys)
    {
	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= n)
	    return;

	int x = floor((xsrc[gid] - xmin) / ext * (1 << LMAX));
	int y = floor((ysrc[gid] - ymin) / ext * (1 << LMAX));
	
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
	
	keys[gid] = key;
    }


    //Node * const node;//, const int x, const int y, const int l, const int s, const int e, const int mask)
    __constant__ int * sorted_keys;
    __constant__ realtype *xsorted, *ysorted, *vsorted;
   
    struct DeviceNode
    {
	int x, y, l, s, e, mask;
	int children[4];

	realtype mass, w, wx, wy;

	__host__ __device__ void setup(int x, int y, int l, int s, int e, int mask)
	    {
		this->x = x;
		this->y = y;
		this->l = l;
		this->s = s;
		this->e = e;
		this->mask = mask;
		
		for (int i = 0; i < 4; ++i) 
		    children[i] = 0;
	    }
    };
    
    __constant__ int bufsize;
    __device__ int nnodes;
    __constant__ DeviceNode * bufnodes;

    __constant__ int queuesize;
    __device__ int * queue, qlock, qhead, qtail, qtailnext, qitems;
    __device__ bool qgood;

    __global__ void setup(const int nsrc)
    {
	nnodes = 1;
	bufnodes[0].setup(0, 0, 0, 0, nsrc, false);

	queue[0] = 0;
	qlock = 1;
	qhead = 0;
	qtail = 1;
	qtailnext = 1;
	qitems = 1;
	qgood = true;
    }

    __device__ DeviceNode * bcast_ptr(DeviceNode * ptr)
    {
	union Pack
	{
	    DeviceNode * ptr;
	    int words[2];
	};

	Pack p;
	p.ptr = ptr;

	const int w0 = __shfl(p.words[0], 0);
	const int w1 = __shfl(p.words[1], 0);
	
	p.words[0] = w0;
	p.words[1] = w1;

	return p.ptr;
    }

#define WARPSIZE 32
    
    __global__ void build_tree(const int LEAF_MAXCOUNT, int * kk)
    {
	assert(blockDim.x == warpSize && WARPSIZE == warpSize);
	
	const int tid = threadIdx.x;
	const int slot = threadIdx.y;
	const bool master = tid == 0;

	//__shared__ volatile DeviceNode * volatile currnodes[4];

	DeviceNode * curr; // = currnodes[slot];
	
	while(qitems && qgood) 
	{
	    curr = NULL;
	    
	    if (master)
	    {
		if (atomicCAS(&qlock, 1, 0))
		{
		    const int currhead = qhead;
		    
		    if (currhead < qtail)
		    {
			const int entry = currhead % queuesize;
			
			//currnodes[slot] = bufnodes + queue[entry];
			curr = bufnodes + queue[entry];

			qhead = currhead + 1;
		
			__threadfence();			
		    }
		    else
			;//currnodes[slot] = NULL;

		    qlock = 1;
		}
	    }
	    	
	    //DeviceNode * const curr = currnodes[slot];
	    curr = bcast_ptr(curr);
	    
	    if (curr)
	    {
		const int s = curr->s;
		const int e = curr->e;
		const int l = curr->l;
		
		const bool leaf = e - s <= LEAF_MAXCOUNT || l + 1 > LMAX;
		
		if (leaf)
		{
		    //compute P2E here
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
		    		    
		    if (master)
		    {
			curr->mass = msum;
			curr->w = wsum;
			curr->wx = wxsum;
			curr->wy = wysum;
			
			atomicSub(&qitems, 1);
		    }
		}		
		else
		{
		    //allocate children
		    if (master)
		    {
			const int bufbase = atomicAdd(&nnodes, 4);

			if (bufbase + 4 > bufsize)
			{
			    qgood = false;
			    break;
			}
			
			for(int c = 0; c < 4; ++c)
			    curr->children[c] = bufbase + c;
		    }

		    const int mask = curr->mask;
		    const int x = curr->x;
		    const int y = curr->y;
		
		    for(int c = 0; c < 4; ++c)
		    {
			const int shift = 2 * (LMAX - l - 1);
			
			const int key1 = mask | (c << shift);
			const int key2 = key1 + (1 << shift) - 1;

			const size_t indexmin = c == 0 ? s : lower_bound(sorted_keys + s, sorted_keys + e, key1) - sorted_keys;
			const size_t indexsup = c == 3 ? e : upper_bound(sorted_keys + s, sorted_keys + e, key2) - sorted_keys;

			if (master)
			{    
			    DeviceNode * child = bufnodes + curr->children[c];
			    child->setup((x << 1) + (c & 1), (y << 1) + (c >> 1), l + 1, indexmin, indexsup, key1);
			}
		    }

		    if (master)
		    {
			const int base = atomicAdd(&qtailnext, 4);
		    //printf("base: %d\n", base);

			if (base + 4 - qhead >= queuesize)
			    qgood = false;
			else
			{
			    for(int c = 0; c < 4; ++c)
				queue[(base + c) % queuesize] = curr->children[c];
			    
			    atomicAdd(&qitems, 3);
			    
			    __threadfence();
			    
			    atomicAdd(&qtail, 4);
			}
		    }
		}
	    }
	}

	//assert(qgood);
    }

    __global__ void conclude(int * treenodes, int * queuesize)
    {
	*treenodes = nnodes;
	*queuesize = qtail - qhead;
    }
}

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
	printf(":%d: 0x%x vs 0x%x \n", i, a.c[7 - i], b.c[7 - i]);
	unsigned char c1 = a.c[7 - i], c2 = b.c[7 - i];
	for(int b = 0; b < 8; ++b, ++currbit)
	{
	    if (((c1 >> b) & 1) != ((c2 >> b) & 1))
	    {
		printf("numbers differ from the %d most-significant bit\n", currbit);
		return currbit;
	    }
	}
    }

    return currbit;
}

void check_tree (Tree::DeviceNode * allnodes, Tree::DeviceNode& a, Tree::Node& b)
	{
	    assert(a.x == b.x);
	    assert(a.y == b.y);
	    assert(a.l == b.l);
	    assert(a.s == b.s);
	    assert(a.e == b.e);
	    //assert(a.mask == b.mask);

	    printf("node %d %d l%d s: %d e: %d. check passed..\n", b.x, b.y, b.l, b.s, b.e);
	    
	    if (!b.leaf)
		for(int c = 0; c < 4; ++c)
		    check_tree(allnodes, allnodes[a.children[c]], *b.children[c]);
	    else
	    {
		
		printf("a/ m-w-wx-wy: %.20e %.20e %.20e %.20e\n",
		       a.mass, a.w, a.wx, a.wy);
		printf("b/ m-w-wx-wy: %.20e %.20e %.20e %.20e\n",
		       b.mass, b.w, b.wx, b.wy);

		
		
		assert(check_bits(a.mass, b.mass) >= 48);
		assert(check_bits(a.w, b.w) >= 48);
		assert(check_bits(a.wx, b.wx) >= 48 || a.w == 0);
		assert(check_bits(a.wy, b.wy) >= 48 || a.w == 0);
		//assert(a.mass == b.mass);
		//assert(a.w == b.w);
	    }
	}

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/pair.h>
#include <thrust/sort.h>

void Tree::build(const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
		 Node * const root, const int LEAF_MAXCOUNT)
{
    Tree::LEAF_MAXCOUNT = LEAF_MAXCOUNT;
    
    posix_memalign((void **)&xdata, 32, sizeof(*xdata) * nsrc);
    posix_memalign((void **)&ydata, 32, sizeof(*ydata) * nsrc);
    posix_memalign((void **)&vdata, 32, sizeof(*vdata) * nsrc);
    posix_memalign((void **)&keys, 32, sizeof(int) * nsrc);

    CUDA_CHECK(cudaDeviceReset());
    
    const int device_queuesize = 1e3;
    int * device_queue;
    CUDA_CHECK(cudaMalloc(&device_queue, sizeof(*device_queue) * device_queuesize));
    
    const int device_bufsize = 1e4;
    DeviceNode * device_bufnodes;
    CUDA_CHECK(cudaMalloc(&device_bufnodes, sizeof(*device_bufnodes) * device_bufsize));

    int * device_diag;
    CUDA_CHECK(cudaMallocHost(&device_diag, sizeof(int) * 2));
    
    realtype *device_xdata, *device_ydata, *device_vdata;
    
    CUDA_CHECK(cudaMalloc(&device_xdata, sizeof(realtype) * nsrc));
    CUDA_CHECK(cudaMalloc(&device_ydata, sizeof(realtype) * nsrc));
    CUDA_CHECK(cudaMalloc(&device_vdata, sizeof(realtype) * nsrc));
    
    int * device_keys;
    CUDA_CHECK(cudaMalloc(&device_keys, sizeof(int) * nsrc));
    
#ifndef NDEBUG
    CUDA_CHECK(cudaMemset(device_keys, 0xff, sizeof(int) * nsrc));
#endif

    CUDA_CHECK(cudaMemcpy(device_xdata, xsrc, sizeof(realtype) * nsrc, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_ydata, ysrc, sizeof(realtype) * nsrc, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_vdata, vsrc, sizeof(realtype) * nsrc, cudaMemcpyHostToDevice));

    thrust::pair<thrust::device_ptr<realtype>, thrust::device_ptr<realtype> > xminmax =
	thrust::minmax_element(thrust::device_pointer_cast(device_xdata), thrust::device_pointer_cast(device_xdata)  + nsrc);
    
    thrust::pair<thrust::device_ptr<realtype>, thrust::device_ptr<realtype> > yminmax =
	thrust::minmax_element(thrust::device_pointer_cast(device_ydata), thrust::device_pointer_cast(device_ydata)  + nsrc);
    
    const realtype truexmin = *xminmax.first;
    const realtype trueymin = *yminmax.first;
    
    const realtype ext0 = *xminmax.second - truexmin;
    const realtype ext1 = *yminmax.second - trueymin;

    const realtype eps = 10000 * std::numeric_limits<realtype>::epsilon();
    
    ext = std::max(ext0, ext1) * (1 + 2 * eps);
    xmin = truexmin - eps * ext;
    ymin = trueymin - eps * ext;

    generate_keys<<< (nsrc + 127)/128, 128>>>(device_xdata, device_ydata, nsrc,
					      xmin, ymin, ext, device_keys);

    CUDA_CHECK(cudaPeekAtLastError());

    thrust::sort_by_key(thrust::device_pointer_cast(device_keys),
			thrust::device_pointer_cast(device_keys + nsrc),
			thrust::make_zip_iterator(thrust::make_tuple(
						      thrust::device_pointer_cast(device_xdata),
						      thrust::device_pointer_cast(device_ydata),
						      thrust::device_pointer_cast(device_vdata)))); 

    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpyToSymbol(sorted_keys, &device_keys, sizeof(device_keys)));
    CUDA_CHECK(cudaMemcpyToSymbol(xsorted, &device_xdata, sizeof(device_xdata)));
    CUDA_CHECK(cudaMemcpyToSymbol(ysorted, &device_ydata, sizeof(device_ydata)));
    CUDA_CHECK(cudaMemcpyToSymbol(vsorted, &device_vdata, sizeof(device_vdata)));
    CUDA_CHECK(cudaMemcpyToSymbol(bufsize, &device_bufsize, sizeof(device_bufsize)));
    CUDA_CHECK(cudaMemcpyToSymbol(bufnodes, &device_bufnodes, sizeof(device_bufnodes)));
    CUDA_CHECK(cudaMemcpyToSymbol(queuesize, &device_queuesize, sizeof(device_queuesize)));
    CUDA_CHECK(cudaMemcpyToSymbol(queue, &device_queue, sizeof(device_queue)));
    
    setup<<<1, 1>>>(nsrc);
    build_tree<<<14 * 16, dim3(32, 4)>>>(LEAF_MAXCOUNT, device_keys);
    conclude<<<1, 1>>>(device_diag, device_diag + 1);

    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("device has found %d nodes, and max queue size was %d\n", device_diag[0], device_diag[1]);
    
    CUDA_CHECK(cudaMemcpy(xdata, device_xdata, sizeof(realtype) * nsrc, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ydata, device_ydata, sizeof(realtype) * nsrc, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vdata, device_vdata, sizeof(realtype) * nsrc, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaMemcpy(keys, device_keys, sizeof(int) * nsrc, cudaMemcpyDeviceToHost));
    
#ifndef NDEBUG
    std::pair<int, int> * kv = NULL;
    
    posix_memalign((void **)&kv, 32, sizeof(*kv) * nsrc);

    assert(truexmin == *std::min_element(xsrc, xsrc + nsrc));
    assert(trueymin == *std::min_element(ysrc, ysrc + nsrc));
    
    assert(ext0 == *std::max_element(xsrc, xsrc + nsrc) - truexmin);
    assert(ext1 == *std::max_element(ysrc, ysrc + nsrc) - trueymin);
    
    for(int i = 0; i < nsrc; ++i)
    {
	int x = floor((xsrc[i] - xmin) / ext * (1 << LMAX));
	int y = floor((ysrc[i] - ymin) / ext * (1 << LMAX));
	
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
	
	assert(xdata[i] == xsrc[entry]);
	assert(ydata[i] == ysrc[entry]);
	assert(vdata[i] == vsrc[entry]);
    }

    free(kv);
#endif
  
#pragma omp parallel //num_threads(24)
    {
#pragma omp single
	{ _build(root, 0, 0, 0, 0, nsrc, 0); }
    }

#ifndef NDEBUG
    const int nnodes = device_diag[0];
    std::vector<DeviceNode> allnodes(nnodes);
    CUDA_CHECK(cudaMemcpy(&allnodes.front(), device_bufnodes, sizeof(DeviceNode) * allnodes.size(), cudaMemcpyDeviceToHost));

    printf("rooot xylsem: %d %d %d %d %d 0x%x, children %d %d %d %d\n",
	   allnodes[0].x, allnodes[0].y, allnodes[0].l, allnodes[0].s, allnodes[0].e, allnodes[0].mask,
	   allnodes[0].children[0], allnodes[0].children[1], allnodes[0].children[2], allnodes[0].children[3]);
    
    //ok let's check this
    
    
    check_tree(&allnodes.front(), allnodes[0], *root);
#endif

    printf("bye!\n");
    //exit(0);

    CUDA_CHECK(cudaFree(device_xdata));
    CUDA_CHECK(cudaFree(device_ydata));
    CUDA_CHECK(cudaFree(device_vdata));
    CUDA_CHECK(cudaFree(device_keys));
    CUDA_CHECK(cudaFree(device_bufnodes));
    CUDA_CHECK(cudaFree(device_queue));
    CUDA_CHECK(cudaFreeHost(device_diag));
}

void Tree::dispose()
{
    free(xdata);
    free(ydata);
    free(vdata);
    free(keys);
}

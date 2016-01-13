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

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/pair.h>
#include <thrust/sort.h>

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

 extern __device__ void upward_p2e_order12(const realtype xcom,
 	const realtype ycom,
 	const realtype * __restrict__ const xsources,
 	const realtype * __restrict__ const ysources,
 	const realtype * __restrict__ const vsources,
 	const int nsources,
 	realtype * __restrict__ const rexpansions,
 	realtype * __restrict__ const iexpansions);

extern __device__ void upward_e2e_order12(
				const realtype  x0,
	const realtype  y0,
	const realtype  mass,
	const realtype * const rsrcxp,
	const realtype * const isrcxp,
	realtype * __restrict__ const rdstxp,
	realtype * __restrict__ const idstxp);	

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
	int x, y, l, s, e, mask, parent;
	int children[4];
	int validchildren;

	realtype mass, w, wx, wy;

	__host__ __device__ void setup(int x, int y, int l, int s, int e, int mask, int parent)
	{
		this->x = x;
		this->y = y;
		this->l = l;
		this->s = s;
		this->e = e;
		this->mask = mask;
		this->parent = parent;
		this->validchildren = 0;
		
		for (int i = 0; i < 4; ++i) 
			children[i] = 0;
	}

	__device__ realtype xcom() const { return w ? wx / w : 0; }

	__device__ realtype ycom() const { return w ? wy / w : 0; }
};

__constant__ int bufsize;
__device__ int nnodes;
__constant__ int order;
__constant__ realtype *bufexpansion;
__constant__ DeviceNode * bufnodes;

__constant__ int queuesize;
__device__ int * queue, qlock, qhead, qtail, qtailnext, qitems;
__device__ bool qgood;

__global__ void setup(const int nsrc)
{
	nnodes = 1;
	bufnodes[0].setup(0, 0, 0, 0, nsrc, 0, -1);

	queue[0] = 0;
	qlock = 1;
	qhead = 0;
	qtail = 1;
	qtailnext = 1;
	qitems = 1;
	qgood = true;
}

#define WARPSIZE 32

__global__ void build_tree(const int LEAF_MAXCOUNT, int * kk)
{
	assert(blockDim.x == warpSize && WARPSIZE == warpSize);
	
	const int tid = threadIdx.x;
	const bool master = tid == 0;

	int curr;
	
	while(qitems && qgood) 
	{
		curr = -1;

		if (master)
		{
		if (atomicCAS(&qlock, 1, 0)) //then take one task if available
		{
			const int currhead = qhead;

			if (currhead < qtail)
			{
				const int entry = currhead % queuesize;

				curr = queue[entry];

				qhead = currhead + 1;

				__threadfence();			
			}

			qlock = 1;
		}
	}

	curr = __shfl(curr, 0);

	if (curr >= 0)
	{
		DeviceNode * node = bufnodes + curr;
		
		const int s = node->s;
		const int e = node->e;
		const int l = node->l;
		
		const bool leaf = e - s <= LEAF_MAXCOUNT || l + 1 > LMAX;
		
		if (leaf)
		{
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

			upward_p2e_order12(xcom, ycom, 
				xsorted + s, ysorted + s, vsorted + s, e - s,
				bufexpansion + order * (2 * curr + 0),
				bufexpansion + order * (2 * curr + 1));
			
			if (master)
			{
				node->mass = msum;
				node->w = wsum;
				node->wx = wxsum;
				node->wy = wysum;

				atomicSub(&qitems, 1);

				__threadfence();
			}

			while(node->parent >= 0)
			{
				DeviceNode * parent = bufnodes + node->parent;

				bool e2e = false;

				if (master)
			    	e2e = 3 == atomicAdd(&parent->validchildren, 1); //then this node is in the critical path
			    
			    e2e = __shfl(e2e, 0);

			    if (e2e)
			    {
			    	realtype msum = 0, wsum = 0, wxsum = 0, wysum = 0, xcom_parent, ycom_parent;

			    	if (master)
			    	{
			    		for(int c = 0; c < 4; ++c)
			    		{
			    			const DeviceNode * child = bufnodes + parent->children[c];

			    			msum += child->mass;
			    			wsum += child->w;
			    			wxsum += child->wx;
			    			wysum += child->wy;
			    		}

			    		parent->mass = msum;
			    		parent->w = wsum;
			    		parent->wx = wxsum;
			    		parent->wy = wysum;

			    		assert(wsum);
			    		xcom_parent = wxsum / wsum;
			    		ycom_parent = wysum / wsum;
			    	}

			    	xcom_parent = __shfl(xcom_parent, 0);
			    	ycom_parent = __shfl(ycom_parent, 0);

			    	if (tid < 4)
			    	{
		    			const DeviceNode * chd = bufnodes + parent->children[tid];

			    		upward_e2e_order12(chd->xcom() - xcom_parent, chd->ycom() - ycom_parent, chd->mass, 
			    			bufexpansion + order * (2 * parent->children[tid] + 0),
			    			bufexpansion + order * (2 * parent->children[tid] + 1),
			    			bufexpansion + order * (2 * node->parent + 0),
			    			bufexpansion + order * (2 * node->parent + 1));
			    	}

			    	if (master)
			    		__threadfence();
			    }
			    else
			    	break;

			    node = parent;

			    if (master && node->parent == -1)
			    	printf("that was the root.\n");
			}
		}	
		else
		{
		    if (master) //children allocation
		    {
		    	const int bufbase = atomicAdd(&nnodes, 4);

		    	if (bufbase + 4 > bufsize)
		    	{
		    		qgood = false;
		    		break;
		    	}

		    	for(int c = 0; c < 4; ++c)
		    		node->children[c] = bufbase + c;
		    }

		    const int mask = node->mask;
		    const int x = node->x;
		    const int y = node->y;

		    for(int c = 0; c < 4; ++c)
		    {
		    	const int shift = 2 * (LMAX - l - 1);

		    	const int key1 = mask | (c << shift);
		    	const int key2 = key1 + (1 << shift) - 1;

		    	const size_t indexmin = c == 0 ? s : lower_bound(sorted_keys + s, sorted_keys + e, key1) - sorted_keys;
		    	const size_t indexsup = c == 3 ? e : upper_bound(sorted_keys + s, sorted_keys + e, key2) - sorted_keys;

		    	if (master)
		    	{    
		    		DeviceNode * child = bufnodes + node->children[c];
		    		child->setup((x << 1) + (c & 1), (y << 1) + (c >> 1), l + 1, indexmin, indexsup, key1, curr);
		    	}
		    }

		    if (master) //enqueue new tasks
		    {
		    	const int base = atomicAdd(&qtailnext, 4);

		    	if (base + 4 - qhead >= queuesize)
		    		qgood = false;
		    	else
		    	{
		    		for(int c = 0; c < 4; ++c)
		    			queue[(base + c) % queuesize] = node->children[c];

		    		atomicAdd(&qitems, 3);

		    		__threadfence();

		    		atomicAdd(&qtail, 4);
		    	}
		    }
		}
	}
}
}

__global__ void conclude(int * treenodes, int * queuesize)
{
	*treenodes = nnodes;
	*queuesize = qtail - qhead;

	const int asd = 0;//trallallero();
	
	printf("conclusion...%d\n", asd);
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
		//printf(":%d: 0x%x vs 0x%x \n", i, a.c[7 - i], b.c[7 - i]);
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

int check_bits(const double *a , const double *b, const int n)
{
	printf("******************\n");
	int r = 64;

	for(int i = 0; i < n; ++i)
	{
		if (fabs(a[i]) > 1.11e-16 || fabs(b[i]) > 1.11e-16)
		{
			int l = check_bits(a[i], b[i]);
			const double x = a[i];
			const double y = b[i];
			if (l < 48 )
				printf("strange case of %+.20e vs %+.20e relerr %e\n", x, y, (x - y) / y);
			r= min(r, l );
		}
	}

	printf("********** end ***************\n");
	return r;
}

void check_tree (const int EXPORD, const int nodeid, realtype * allexp, Tree::DeviceNode * allnodes, Tree::DeviceNode& a, Tree::Node& b)
{
	assert(a.x == b.x);
	assert(a.y == b.y);
	assert(a.l == b.l);
	assert(a.s == b.s);
	assert(a.e == b.e);
	    //assert(a.mask == b.mask);

	printf("a/ m-w-wx-wy: %.20e %.20e %.20e %.20e\n",
		a.mass, a.w, a.wx, a.wy);
	printf("b/ m-w-wx-wy: %.20e %.20e %.20e %.20e\n",
		b.mass, b.w, b.wx, b.wy);

	

	assert(check_bits(a.mass, b.mass) >= 40);
	assert(check_bits(a.w, b.w) >= 40);
	assert(check_bits(a.wx, b.wx) >= 40 || a.w == 0);
	assert(check_bits(a.wy, b.wy) >= 40 || a.w == 0);

//printf("<%s>", (b.leaf ? "LEAF" : "INNER"));
//	printf("node %d %d l%d s: %d e: %d. check passed..\n", b.x, b.y, b.l, b.s, b.e);
	
	{
			const realtype * resrexp = allexp + EXPORD * (2 * nodeid + 0);
			const realtype * resiexp = allexp + EXPORD * (2 * nodeid + 1);
			const realtype * refrexp = b.rexp();
			const realtype * refiexp = b.iexp();
			assert(24 <= check_bits(resrexp, refrexp, EXPORD));
			assert(24 <= check_bits(resiexp, refiexp, EXPORD));
		/*	printf("RRES: ");
			for(int i = 0; i < EXPORD; ++i)
				printf("%+.2e ", resrexp[i]);
			printf("\n");

			printf("RREF: ");
			for(int i = 0; i < EXPORD; ++i)
				printf("%+.2e ", refrexp[i]);
			printf("\n");

			printf("IRES: ");
			for(int i = 0; i < EXPORD; ++i)
				printf("%+.2e ", resiexp[i]);
			printf("\n");

			printf("IREF: ");
			for(int i = 0; i < EXPORD; ++i)
				printf("%+.2e ", resrexp[i]);
			printf("\n");
			*/
		//assert(a.mass == b.mass);
		//assert(a.w == b.w);
		}

	if (!b.leaf)
		for(int c = 0; c < 4; ++c)
			check_tree(EXPORD, a.children[c], allexp, allnodes, allnodes[a.children[c]], *b.children[c]);
		else;
	}



	void Tree::build(const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
		Node * const root, const int LEAF_MAXCOUNT, const int EXPANSIONORDER)
	{
		Tree::LEAF_MAXCOUNT = LEAF_MAXCOUNT;

		posix_memalign((void **)&xdata, 32, sizeof(*xdata) * nsrc);
		posix_memalign((void **)&ydata, 32, sizeof(*ydata) * nsrc);
		posix_memalign((void **)&vdata, 32, sizeof(*vdata) * nsrc);
		posix_memalign((void **)&keys, 32, sizeof(int) * nsrc);

		CUDA_CHECK(cudaDeviceReset());

		const int device_queuesize = 8e4;
		int * device_queue;
		CUDA_CHECK(cudaMalloc(&device_queue, sizeof(*device_queue) * device_queuesize));

		const int device_bufsize = 8e4;
		DeviceNode * device_bufnodes;
		CUDA_CHECK(cudaMalloc(&device_bufnodes, sizeof(*device_bufnodes) * device_bufsize));
		CUDA_CHECK(cudaMemset(device_bufnodes, 0, sizeof(*device_bufnodes) * device_bufsize));

		realtype * device_bufexpansions;
		CUDA_CHECK(cudaMalloc(&device_bufexpansions, sizeof(realtype) * EXPANSIONORDER * 2 * device_bufsize));    
		CUDA_CHECK(cudaMemset(device_bufexpansions, 0, sizeof(realtype)* EXPANSIONORDER * 2 * device_bufsize));
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
		CUDA_CHECK(cudaMemcpyToSymbol(bufexpansion, &device_bufexpansions, sizeof(device_bufexpansions)));
		CUDA_CHECK(cudaMemcpyToSymbol(order, &EXPANSIONORDER, sizeof(EXPANSIONORDER)));
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
		printf("nnodes: %d", nnodes);
		CUDA_CHECK(cudaMemcpy(&allnodes.front(), device_bufnodes, sizeof(DeviceNode) * allnodes.size(), cudaMemcpyDeviceToHost));

		std::vector<realtype> allexpansions(nnodes * EXPANSIONORDER * 2);
		CUDA_CHECK(cudaMemcpy(&allexpansions.front(), device_bufexpansions, sizeof(realtype) * 2 * EXPANSIONORDER * nnodes, cudaMemcpyDeviceToHost));
/*
		printf("nonzero entries:\n");
		for(int i = 0; i < allexpansions.size(); ++i)
			if (allexpansions[i] != 0)
				printf("ASD %d: %e\n", i, allexpansions[i]);
*/
			printf("rooot xylsem: %d %d %d %d %d 0x%x, children %d %d %d %d\n",
				allnodes[0].x, allnodes[0].y, allnodes[0].l, allnodes[0].s, allnodes[0].e, allnodes[0].mask,
				allnodes[0].children[0], allnodes[0].children[1], allnodes[0].children[2], allnodes[0].children[3]);

    //ok let's check this


			check_tree(EXPANSIONORDER, 0, &allexpansions.front(), &allnodes.front(), allnodes[0], *root);
#endif

			printf("bye!\n");
			//exit(0);

			CUDA_CHECK(cudaFree(device_xdata));
			CUDA_CHECK(cudaFree(device_ydata));
			CUDA_CHECK(cudaFree(device_vdata));
			CUDA_CHECK(cudaFree(device_keys));
			CUDA_CHECK(cudaFree(device_bufnodes));
			CUDA_CHECK(cudaFree(device_queue));
			CUDA_CHECK(cudaFree(device_bufexpansions));
			CUDA_CHECK(cudaFreeHost(device_diag));
		}

		void Tree::dispose()
		{
			free(xdata);
			free(ydata);
			free(vdata);
			free(keys);
		}

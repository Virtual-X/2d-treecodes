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
}

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/pair.h>
#include <thrust/sort.h>

void Tree::build(const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
		 Node * const root, const int LEAF_MAXCOUNT)
{
    const double t0 = omp_get_wtime();
 
    Tree::LEAF_MAXCOUNT = LEAF_MAXCOUNT;

    posix_memalign((void **)&xdata, 32, sizeof(*xdata) * nsrc);
    posix_memalign((void **)&ydata, 32, sizeof(*ydata) * nsrc);
    posix_memalign((void **)&vdata, 32, sizeof(*vdata) * nsrc);
    posix_memalign((void **)&keys, 32, sizeof(int) * nsrc);
    
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
					      xmin, ymin, ext,
					      device_keys);

    CUDA_CHECK(cudaPeekAtLastError());

    thrust::sort_by_key(thrust::device_pointer_cast(device_keys),
		      thrust::device_pointer_cast(device_keys + nsrc),
		      thrust::make_zip_iterator(thrust::make_tuple(
						    thrust::device_pointer_cast(device_xdata),
						    thrust::device_pointer_cast(device_ydata),
						    thrust::device_pointer_cast(device_vdata)))); 

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
  
  const double t5 = omp_get_wtime();

#pragma omp parallel //num_threads(24)
  {
#pragma omp single
    { _build(root, 0, 0, 0, 0, nsrc, 0); }
  }

  CUDA_CHECK(cudaFree(device_xdata));
  CUDA_CHECK(cudaFree(device_ydata));
  CUDA_CHECK(cudaFree(device_vdata));
  CUDA_CHECK(cudaFree(device_keys));
}

void Tree::dispose()
{
    free(xdata);
    free(ydata);
    free(vdata);
    free(keys);
}

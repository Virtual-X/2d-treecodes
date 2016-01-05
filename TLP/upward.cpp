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

#include <parallel/algorithm>
#include <limits>
#include <utility>

#include "upward.h"
#include "upward-kernels.h"

//#define  _INSTRUMENTATION_
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

	realtype rcandidates[4];
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
}

void Tree::build(const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
		 Node * const root, const int LEAF_MAXCOUNT)
{
  const double t0 = omp_get_wtime();
  const int isdynamic = omp_get_dynamic();
  const int maxthreads = omp_get_max_threads();

  omp_set_dynamic(0);
  omp_set_num_threads(24);

  Tree::LEAF_MAXCOUNT = LEAF_MAXCOUNT;

  posix_memalign((void **)&keys, 32, sizeof(int) * nsrc);
  posix_memalign((void **)&xdata, 32, sizeof(*xdata) * nsrc);
  posix_memalign((void **)&ydata, 32, sizeof(*ydata) * nsrc);
  posix_memalign((void **)&vdata, 32, sizeof(*vdata) * nsrc);

  std::pair<int, int> * kv = NULL;
  posix_memalign((void **)&kv, 32, sizeof(*kv) * nsrc);

  const double t1 = omp_get_wtime();
  xmin = *__gnu_parallel::min_element(xsrc, xsrc + nsrc);
  ymin = *__gnu_parallel::min_element(ysrc, ysrc + nsrc);

  const realtype ext0 = (*__gnu_parallel::max_element(xsrc, xsrc + nsrc) - xmin);
  const realtype ext1 = (*__gnu_parallel::max_element(ysrc, ysrc + nsrc) - ymin);

  const realtype eps = 10000 * std::numeric_limits<realtype>::epsilon();

  ext = std::max(ext0, ext1) * (1 + 2 * eps);
  xmin -= eps * ext;
  ymin -= eps * ext;
  const double t2 = omp_get_wtime();

#pragma omp parallel for
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

  const double t3 = omp_get_wtime();
  __gnu_parallel::sort(kv, kv + nsrc);
  const double t4 = omp_get_wtime();

#pragma omp parallel for
  for(int i = 0; i < nsrc; ++i)
    {
      keys[i] = kv[i].first;

      const int entry = kv[i].second;
      assert(entry >= 0 && entry < nsrc);

      xdata[i] = xsrc[entry];
      ydata[i] = ysrc[entry];
      vdata[i] = vsrc[entry];
    }

  const double t5 = omp_get_wtime();

#pragma omp parallel
  {
#pragma omp single
    { _build(root, 0, 0, 0, 0, nsrc, 0); }
  }

  free(kv);
  free(keys);

  omp_set_num_threads(maxthreads);
  omp_set_dynamic(isdynamic);

  const double t6 = omp_get_wtime();
#ifdef _INSTRUMENTATION_
  printf("SETUP: %.2f ms (%.1f %%) REDUCE: %.2f ms (%.1f %%) KEY: %.2f ms (%.1f %%) SORT: %.2f ms (%.1f %%) REORDER: %.2f ms (%.1f %%) TREE: %.2f ms (%.1f%%)\n",
	 (t1 - t0) * 1e3, (t1 - t0) / (t6 - t0) * 100,
	 (t2 - t1) * 1e3, (t2 - t1) / (t6 - t0) * 100,
	 (t3 - t2) * 1e3, (t3 - t2) / (t6 - t0) * 100,
	 (t4 - t3) * 1e3, (t4 - t3) / (t6 - t0) * 100,
	 (t5 - t4) * 1e3, (t5 - t4) / (t6 - t0) * 100,
	 (t6 - t5) * 1e3, (t6 - t5) / (t6 - t0) * 100);

  std::pair<int64_t, int64_t> allcycles = root->cycles(true, false);
  std::pair<int64_t, int64_t> usecycles = root->cycles(false, false);
  std::pair<int64_t, int64_t> searchcycles = root->cycles(false, true);
  std::pair<int, int> nodes = root->nodes();
  printf("TREE:\nNODES: %.2e LEAVES: %.2e (%.1f%%)\n ALL: %.2e c, CP: %.2e c (%.1f%%), WORK: %e (%.1f%%), USEFUL CP: %e (%.1f%% of WORK)\nSEARCH: %.2e (%.1f%%) CP: %.2e (%.1f%%)\n",
	 (double)nodes.first, (double)nodes.second, (double)nodes.second * 100. / nodes.first,
	 (double)allcycles.first,
	 (double)allcycles.second,(double)(allcycles.second * 100. / allcycles.first),
	 (double)usecycles.first, (double)(usecycles.first * 100. / allcycles.first),
	 (double)usecycles.second,(double)(usecycles.second * 100./ usecycles.first),
	 (double)searchcycles.first,(double)(searchcycles.first * 100./ usecycles.first),
	 (double)searchcycles.second,(double)(searchcycles.second * 100./ searchcycles.first));

#endif
}

void Tree::dispose()
{
  free(xdata);
  free(ydata);
  free(vdata);

  delete root;
}

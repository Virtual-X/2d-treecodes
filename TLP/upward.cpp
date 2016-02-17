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
#include "sort-sources.h"


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

  Tree::LEAF_MAXCOUNT = LEAF_MAXCOUNT;

  posix_memalign((void **)&keys, 32, sizeof(int) * nsrc);
  posix_memalign((void **)&xdata, 32, sizeof(*xdata) * nsrc);
  posix_memalign((void **)&ydata, 32, sizeof(*ydata) * nsrc);
  posix_memalign((void **)&vdata, 32, sizeof(*vdata) * nsrc);

  const double t1 = omp_get_wtime();

  sort_sources(xsrc, ysrc, vsrc, nsrc, keys, xdata, ydata, vdata, &xmin, &ymin, &ext);
  
  const double t5 = omp_get_wtime();

#pragma omp parallel //num_threads(24)
  {
#pragma omp single
    { _build(root, 0, 0, 0, 0, nsrc, 0); }
  }

 
  free(keys);
}

void Tree::dispose()
{
  free(xdata);
  free(ydata);
  free(vdata);

  delete root;
}

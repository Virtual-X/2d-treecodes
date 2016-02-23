/*
 *  treecode-force.cpp
 *  Part of 2d-treecodes
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
#include <cstring>
#include <tuple>
#include <algorithm>
#include <omp.h>
#include "upward.h"
#include "force-kernels.h"
#include "treecode-force.h"


namespace EvaluateForce
{
  struct TimeDistrib
  {
    int64_t p2p, e2p, e2l, l2p ;
    
    static __inline__ unsigned long long rdtsc(void)
    {
      unsigned hi, lo;
      __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
      return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
    }

    void init() { p2p = e2p = e2l = l2p ; }

    void print(const int tid)
    {
      const double tot = std::max((int64_t)1, p2p + e2p + e2l + l2p) ;
      printf("total: %.1e Mcycles p2p:%.1f%% e2p:%.1f%% e2l:%.1f%% l2p:%.1f%%\n", 
	     tot * 1e-6, p2p * 100. / tot, e2p * 100. / tot, e2l * 100. / tot, l2p * 100. / tot);
    }

  } td;

#pragma omp threadprivate(td)

    template<int size>
    struct E2LWork 
    {
	int count;
	realtype *const rdst,  *const idst;

	realtype x0s[size], y0s[size], masses[size];
	const realtype * rxps[size], *ixps[size];

	E2LWork(realtype * const rlocal, realtype * const ilocal):
	    count(0), rdst(rlocal), idst(ilocal) { }

	void _flush()
	    {
	      td.e2l -= td.rdtsc();
		downward_e2l(x0s, y0s, masses, rxps, ixps, count, rdst, idst);
	      td.e2l += td.rdtsc();

		count = 0;
	    }

	void push(const realtype x0, const realtype y0, const realtype mass,
		  const realtype * const rxp, const realtype * const ixp)
	    {
		x0s[count] = x0;
		y0s[count] = y0;
		masses[count] = mass;
		rxps[count] = rxp;
		ixps[count] = ixp;

		if (++count >= size)
		    _flush();
	    }

	void finalize()
	    {
		if (count)
		    _flush();
	    }
    };

#define TILE 8
#define BRICKSIZE (TILE * TILE)

    void evaluate(realtype * const xresultbase, realtype * const yresultbase,
		  const realtype x0, const realtype y0, const realtype h,
		  const realtype theta)
    {
	int maxentry = 0;

	int stack[LMAX * 3];

	realtype result[2 * BRICKSIZE];

	realtype rlocal[ORDER + 1], ilocal[ORDER + 1];

	E2LWork<32> e2lwork(rlocal, ilocal);

	const realtype rbrick = 1.4142135623730951 * h * (TILE - 1) * 0.5;

	for(int by = 0; by < BLOCKSIZE; by += TILE)
	    for(int bx = 0; bx < BLOCKSIZE; bx += TILE)
	    {
		const realtype x0brick = x0 + h * (bx + 0.5 * (TILE - 1));
		const realtype y0brick = y0 + h * (by + 0.5 * (TILE - 1));

		for(int i = 0; i <= ORDER; ++i)
		{
		    rlocal[i] = 0;
		    ilocal[i] = 0;
		}

		for(int i = 0; i < 2 * BRICKSIZE; ++i)
		    result[i] = 0;

		int stackentry = 0;
		stack[0] = 0;

		while(stackentry > -1)
		{
		    const int nodeid = stack[stackentry--];
		    const Tree::Node * const node = Tree::nodes + nodeid;

		    const realtype xcom = node->xcom;
		    const realtype ycom = node->ycom;

		    const realtype distance = sqrt(pow(x0brick - xcom, 2) + pow(y0brick - ycom, 2));

		    const bool localexpansion_converges = (distance / node->r - 1) > (1 / theta) && rbrick <= node->r;

		    if (localexpansion_converges)
			e2lwork.push(xcom - x0brick, ycom - y0brick, node->mass,
				     Tree::expansions + ORDER * (2 * nodeid + 0),
				     Tree::expansions + ORDER * (2 * nodeid + 1));
		    else
		    {
			const double xt = std::max(x0 + bx * h, std::min(x0 + (bx + TILE - 1) * h, xcom));
			const double yt = std::max(y0 + by * h, std::min(y0 + (by + TILE - 1) * h, ycom));

			const realtype r2 = pow(xt - xcom, 2) + pow(yt - ycom, 2);

			if (node->r * node->r < theta * theta * r2)
			{
			  td.e2p -= td.rdtsc();
			    force_e2p_8x8(node->mass, x0 + (bx + 0) * h - xcom, y0 + (by + 0) * h - ycom, h,
					  Tree::expansions + ORDER * (2 * nodeid + 0),
					  Tree::expansions + ORDER * (2 * nodeid + 1),
					  result, result + BRICKSIZE);
			    td.e2p += td.rdtsc();
			}
			else
			{
			    if (!node->state.innernode)
			    {
				const int s = node->s;
				td.p2p -= td.rdtsc();
				force_p2p_8x8(&Tree::xdata[s], &Tree::ydata[s], &Tree::vdata[s], node->e - s,
					      x0 + (bx + 0) * h, y0 + (by + 0) * h, h,
					      result, result + BRICKSIZE);
				td.p2p += td.rdtsc();
			    }
			    else
			    {
				for(int c = 0; c < 4; ++c)
				    stack[++stackentry] = node->state.childbase + c;

				maxentry = std::max(maxentry, stackentry);
			    }
			}
		    }
		}
		
		e2lwork.finalize();
		
		td.l2p -= td.rdtsc();
		downward_l2p_8x8(h * (0 - 0.5 * (TILE - 1)),
				 h * (0 - 0.5 * (TILE - 1)),
				 h, rlocal, ilocal, result, result + BRICKSIZE);
		td.l2p += td.rdtsc();


		for(int iy = 0; iy < TILE; ++iy)
		    for(int ix = 0; ix < TILE; ++ix)
			xresultbase[bx + ix + BLOCKSIZE * (by + iy)] = result[ix + TILE * iy];

		for(int iy = 0; iy < TILE; ++iy)
		    for(int ix = 0; ix < TILE; ++ix)
			yresultbase[bx + ix + BLOCKSIZE * (by + iy)] = result[BRICKSIZE + ix + TILE * iy];
	    }
    }

    extern "C"
    __attribute__ ((visibility ("default")))
    void treecode_force_mrag(const realtype theta,
			     const realtype * const xsrc,
			     const realtype * const ysrc,
			     const realtype * const vsrc,
			     const int nsrc,
			     const realtype * const x0s,
			     const realtype * const y0s,
			     const realtype * const hs,
			     const int nblocks,
			     realtype * const xdst,
			     realtype * const ydst)
    {
      const double t0 =  omp_get_wtime();
	Tree::build(xsrc, ysrc, vsrc, nsrc, 192);

	const double t1 = omp_get_wtime();
#pragma omp parallel
	{
	  td.init();

#pragma omp for schedule(dynamic,1)	
	       for(int i = 0; i < nblocks; ++i)
		 evaluate(xdst + i * BLOCKSIZE * BLOCKSIZE, ydst + i * BLOCKSIZE * BLOCKSIZE, x0s[i], y0s[i], hs[i], theta);

	       const int tid = omp_get_thread_num();
	       //printf("hello tid %d\n", tid);
	       td.print(tid);
	}

	const double t2 = omp_get_wtime();
      
	printf("UP: %.1es DW: %.1es (%.1f%%)\n", t1- t0, t2-t1, (t2 - t1) * 100. / (t2 - t0));
	Tree::dispose();
    }
}

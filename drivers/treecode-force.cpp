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

#include "treecode-force.h"
#include "upward-kernels.h"
#include "downward-kernels.h"
#include "force-kernels.h"
#include "upward.h"

namespace EvaluateForce
{
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
		downward_e2l(x0s, y0s, masses, rxps, ixps, count, rdst, idst);

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

#define BRICKSIZE 8

    void evaluate(realtype * const xresultbase, realtype * const yresultbase,
		  const realtype x0, const realtype y0, const realtype h,
		  const realtype theta)
    {
	int maxentry = 0;

	int stack[LMAX * 3];

	realtype xresult[BRICKSIZE][BRICKSIZE], yresult[BRICKSIZE][BRICKSIZE];

	realtype rlocal[ORDER + 1], ilocal[ORDER + 1];

	E2LWork<32> e2lwork(rlocal, ilocal);

	const realtype rbrick = 1.4142135623730951 * h * (BRICKSIZE - 1) * 0.5;

	for(int by = 0; by < BLOCKSIZE; by += BRICKSIZE)
	    for(int bx = 0; bx < BLOCKSIZE; bx += BRICKSIZE)
	    {
		const realtype x0brick = x0 + h * (bx + 0.5 * (BRICKSIZE - 1));
		const realtype y0brick = y0 + h * (by + 0.5 * (BRICKSIZE - 1));

		for(int i = 0; i <= ORDER; ++i)
		    rlocal[i] = ilocal[i] = 0;

		for(int iy = 0; iy < BRICKSIZE; ++iy)
		    for(int ix = 0; ix < BRICKSIZE; ++ix)
			xresult[iy][ix] = 0;

		for(int iy = 0; iy < BRICKSIZE; ++iy)
		    for(int ix = 0; ix < BRICKSIZE; ++ix)
			yresult[iy][ix] = 0;

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
			const double xt = std::max(x0 + bx * h, std::min(x0 + (bx + BRICKSIZE - 1) * h, xcom));
			const double yt = std::max(y0 + by * h, std::min(y0 + (by + BRICKSIZE - 1) * h, ycom));

			const realtype r2 = pow(xt - xcom, 2) + pow(yt - ycom, 2);

			if (node->r * node->r < theta * theta * r2)
			{
			    force_e2p_8x8(node->mass, x0 + (bx + 0) * h - xcom, y0 + (by + 0) * h - ycom, h,
					  Tree::expansions + ORDER * (2 * nodeid + 0),
					  Tree::expansions + ORDER * (2 * nodeid + 1),
					  &xresult[0][0], &yresult[0][0]);
			}
			else
			{
			    if (!node->state.innernode)
			    {
				const int s = node->s;

				force_p2p_8x8(&Tree::xdata[s], &Tree::ydata[s], &Tree::vdata[s], node->e - s,
					      x0 + (bx + 0) * h, y0 + (by + 0) * h, h,
					      &xresult[0][0], &yresult[0][0]);
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

		downward_l2p_8x8(h * (0 - 0.5 * (BRICKSIZE - 1)),
				 h * (0 - 0.5 * (BRICKSIZE - 1)),
				 h, rlocal, ilocal,
				 &xresult[0][0], &yresult[0][0]);

		for(int iy = 0; iy < BRICKSIZE; ++iy)
		    for(int ix = 0; ix < BRICKSIZE; ++ix)
			xresultbase[bx + ix + BLOCKSIZE * (by + iy)] = xresult[iy][ix];

		for(int iy = 0; iy < BRICKSIZE; ++iy)
		    for(int ix = 0; ix < BRICKSIZE; ++ix)
			yresultbase[bx + ix + BLOCKSIZE * (by + iy)] = yresult[iy][ix];
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
	Tree::build(xsrc, ysrc, vsrc, nsrc, 192);

#pragma omp parallel for schedule(dynamic,1)
	for(int i = 0; i < nblocks; ++i)
	    evaluate(xdst + i * BLOCKSIZE * BLOCKSIZE, ydst + i * BLOCKSIZE * BLOCKSIZE, x0s[i], y0s[i], hs[i], theta);

	Tree::dispose();
    }
}

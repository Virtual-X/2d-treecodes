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
#include <cstring>
#include <tuple>
#include <algorithm>

#include "treecode.h"
#include "upward-kernels.h"
#include "force-kernels.h"
#include "upward.h"

#define _INSTRUMENTATION_

#ifndef _INSTRUMENTATION_
#define MYRDTSC 0
#else
#define MYRDTSC _rdtsc()
#endif

namespace EvaluateForce
{
    struct NodeForce : Tree::NodeImplementation<ORDER> { };

    realtype *xdata = nullptr, *ydata = nullptr, *vdata = nullptr;

    struct PerfMon
    {
	int64_t e2pcalls, e2pcycles, p2pcalls, p2pcycles, p2pinteractions;
	int64_t startc, endc;

	int maxstacksize, evaluations;

	bool failed;

	void setup()
	    {
		e2pcalls = 0;
		e2pcycles = 0;
		p2pcalls = 0;
		p2pcycles = 0;
		p2pinteractions = 0;

		maxstacksize = 0;
		evaluations = 0;
		failed = false;
	    }

	double tot_cycles() { return (double)(endc - startc); }

	std::tuple<double, double, double, double> e2p(const int instructions)
	    {
		return std::make_tuple((double)(e2pcycles),
				       (double)(e2pcycles) / tot_cycles(),
				       (double)(e2pcycles) / e2pcalls,
				       (double)(e2pcalls * instructions) / e2pcycles);
	    }

	std::tuple<double, double, double, double> p2p()
	    {
		return std::make_tuple((double)(p2pcycles),
				       (double)(p2pcycles) / tot_cycles(),
				       (double)(p2pcycles) / p2pcalls,
				       (double)p2pcycles / (double)(p2pinteractions));
	    }

	std::tuple<double, double, int> traversal()
	    {
		return std::make_tuple((double)(endc - startc - p2pcycles - e2pcycles),
				       (double)(endc - startc - p2pcycles - e2pcycles) / (endc - startc),
				       maxstacksize);
	    }

    } perfmon;

#pragma omp threadprivate(perfmon)

    void evaluate(realtype * const xresult, realtype * const yresult,
		  const realtype xt, const realtype yt,
		  const NodeForce & root, const realtype thetasquared)
    {
	const NodeForce * stack[15 * 4 * 2];

	int stackentry = 0, maxentry = 0;

	stack[0] = &root;

	*xresult = 0;
	*yresult = 0;

	while(stackentry > -1)
	{
	    const NodeForce * const node = stack[stackentry--];

	    realtype tmp[2];

	    const realtype r2 = pow(xt - node->xcom(), 2) + pow(yt - node->ycom(), 2);

	    if (4 * node->r * node->r < thetasquared * r2)
	    {
		int64_t startc = MYRDTSC;
		force_e2p(node->mass, xt - node->xcom(), yt - node->ycom(), node->rexpansions, node->iexpansions, tmp, tmp + 1);
		int64_t endc = MYRDTSC;

		*xresult += tmp[0];
		*yresult += tmp[1];
#ifdef _INSTRUMENTATION_
		perfmon.e2pcycles += endc - startc;
		++perfmon.e2pcalls;
#endif
	    }
	    else
	    {
		if (node->leaf)
		{
		    const int s = node->s;

		    int64_t startc = MYRDTSC;
		    force_p2p(&xdata[s], &ydata[s], &vdata[s], node->e - s, xt, yt, tmp, tmp + 1);
		    int64_t endc = MYRDTSC;

		    *xresult += tmp[0];
		    *yresult += tmp[1];
#ifdef _INSTRUMENTATION_
		    perfmon.p2pcycles += endc - startc;
		    perfmon.p2pinteractions += node->e - s;
		    ++perfmon.p2pcalls;
#endif
		}
		else
		{
		    for(int c = 0; c < 4; ++c)
			stack[++stackentry] = (NodeForce *)node->children[c];

		    maxentry = std::max(maxentry, stackentry);
		}
	    }
	}

#ifdef _INSTRUMENTATION_
	perfmon.maxstacksize = maxentry + 1;
	++perfmon.evaluations;
	perfmon.failed = maxentry >= sizeof(stack) / sizeof(*stack);
#endif
    }

#define TILESIZE 4

    void evaluate(realtype * const xresultbase, realtype * const yresultbase,
		  const realtype x0, const realtype y0, const realtype h,
		  const NodeForce & root, const realtype thetasquared)
    {
	int maxentry = 0;

	const NodeForce * stack[15 * 4 * 2];

	for(int ty = 0; ty < BLOCKSIZE; ty += TILESIZE)
	    for(int tx = 0; tx < BLOCKSIZE; tx += TILESIZE)
	    {
		realtype xresult[TILESIZE][TILESIZE], yresult[TILESIZE][TILESIZE];

		for(int iy = 0; iy < TILESIZE; ++iy)
		    for(int ix = 0; ix < TILESIZE; ++ix)
			xresult[iy][ix] = 0;

		for(int iy = 0; iy < TILESIZE; ++iy)
		    for(int ix = 0; ix < TILESIZE; ++ix)
			yresult[iy][ix] = 0;

		int stackentry = 0;
		stack[0] = &root;

		while(stackentry > -1)
		{
		    const NodeForce * const node = stack[stackentry--];

		    const realtype xcom = node->xcom();
		    const realtype ycom = node->ycom();

		    const double xt = std::max(x0 + tx * h, std::min(x0 + (tx + TILESIZE - 1) * h, xcom));
		    const double yt = std::max(y0 + ty * h, std::min(y0 + (ty + TILESIZE - 1) * h, ycom));

		    const realtype r2 = pow(xt - xcom, 2) + pow(yt - ycom, 2);

		    if (4 * node->r * node->r < thetasquared * r2)
		    {
			int64_t startc = MYRDTSC;

			force_e2p_tiled(node->mass, x0 + tx * h - xcom, y0 + ty * h - ycom, h,
					node->rexpansions, node->iexpansions, &xresult[0][0], &yresult[0][0]);

			int64_t endc = MYRDTSC;

#ifdef _INSTRUMENTATION_
			perfmon.e2pcycles += endc - startc;
			perfmon.e2pcalls += 1;
#endif
		    }
		    else
		    {
			if (node->leaf)
			{
			    const int s = node->s;

			    int64_t startc = MYRDTSC;

			    for(int iy = 0; iy < TILESIZE; ++iy)
				for(int ix = 0; ix < TILESIZE; ++ix)
				{
				    realtype tmp[2];

				    force_p2p(&xdata[s], &ydata[s], &vdata[s], node->e - s,
					      x0 + (tx + ix) * h, y0 + (ty + iy) * h, tmp, tmp + 1);

				    xresult[iy][ix] += tmp[0];
				    yresult[iy][ix] += tmp[1];
				}

			    int64_t endc = MYRDTSC;

#ifdef _INSTRUMENTATION_
			    perfmon.p2pcycles += endc - startc;
			    perfmon.p2pinteractions += (node->e - s) * TILESIZE * TILESIZE;
			    ++perfmon.p2pcalls;
#endif
			}
			else
			{
			    for(int c = 0; c < 4; ++c)
				stack[++stackentry] = (NodeForce *)node->children[c];

			    maxentry = std::max(maxentry, stackentry);
			}
		    }
		}

		for(int iy = 0; iy < TILESIZE; ++iy)
		    for(int ix = 0; ix < TILESIZE; ++ix)
			xresultbase[tx + ix + BLOCKSIZE * (ty + iy)] = xresult[iy][ix];

		for(int iy = 0; iy < TILESIZE; ++iy)
		    for(int ix = 0; ix < TILESIZE; ++ix)
			yresultbase[tx + ix + BLOCKSIZE * (ty + iy)] = yresult[iy][ix];
	    }

#ifdef _INSTRUMENTATION_
	perfmon.maxstacksize = maxentry + 1;
	perfmon.evaluations += TILESIZE * TILESIZE;
	perfmon.failed = maxentry >= sizeof(stack) / sizeof(*stack);
#endif
    }

    void report_instrumentation(PerfMon perf[], const int N, const double t0, const double t1, const int e2pinstructions)
    {
#ifdef _INSTRUMENTATION_
	for(int i = 0; i < N; ++i)
	    if (perf[i].failed)
	    {
		printf("oops there was an overflow in the computation\n");
		abort();
	    }

	for(int i = 0; i < N; ++i)
	    printf("TID %d: tot cycles: %.3e\n", i, perf[i].tot_cycles());

	for(int i = 0; i < N; ++i)
	{
	    auto p = perf[i].e2p(e2pinstructions);

	    printf("TID %d: E2P cycles: %.3e (%.1f %%) cycles-per-call: %.1f, ipc: %.2f\n",
		   i, std::get<0>(p), std::get<1>(p) * 100., std::get<2>(p), std::get<3>(p));
	}

	for(int i = 0; i < N; ++i)
	{
	    auto p = perf[i].traversal();

	    printf("TID %d: traversal overhead: %.1f %%, max stacksize: %d \n", i, 100. * std::get<1>(p), std::get<2>(p));
	}

	for(int i = 0; i < N; ++i)
	{
	    auto p = perf[i].p2p();

	    printf("TID %d: P2P importance: %.1f %% cycles-per-interactions: %.1f, cycles-per-call: %.1f\n",
		   i, std::get<1>(p) * 100, std::get<3>(p), std::get<2>(p));
	}

	const double t2 = omp_get_wtime();
	printf("UPWARD: %.2f ms EVAL: %.2f ms (%.1f %%)\n", (t1-t0)*1e3, (t2-t1)*1e3, (t2 - t1) / (t2 - t0) * 100);
#endif
    }

    extern "C"
    void treecode_force(const realtype theta,
			const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
			const realtype * const xdst, const realtype * const ydst, const int ndst, realtype * const xresult, realtype * const yresult)
    {
	const realtype thetasquared = theta * theta;

	NodeForce root;

	const double t0 = omp_get_wtime();
	Tree::build(xsrc, ysrc, vsrc, nsrc, &root, 128);
	const double t1 = omp_get_wtime();

	xdata = Tree::xdata;
	ydata = Tree::ydata;
	vdata = Tree::vdata;

	PerfMon perf[omp_get_max_threads()];

#pragma omp parallel
	{
	    perfmon.setup();
	    perfmon.startc = MYRDTSC;

#pragma omp for schedule(static,1)
	    for(int i = 0; i < ndst; ++i)
		evaluate(xresult + i, yresult + i, xdst[i], ydst[i], root, thetasquared);

	    perfmon.endc = MYRDTSC;
#ifdef _INSTRUMENTATION_
	    perf[omp_get_thread_num()] = perfmon;
#endif
	}

	report_instrumentation(perf, sizeof(perf) / sizeof(*perf), t0, t1, 196);
    }

    extern "C"
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
	const realtype thetasquared = theta * theta;

	NodeForce root;

	const double t0 = omp_get_wtime();
	Tree::build(xsrc, ysrc, vsrc, nsrc, &root, 128);
	const double t1 = omp_get_wtime();

	xdata = Tree::xdata;
	ydata = Tree::ydata;
	vdata = Tree::vdata;

	PerfMon perf[omp_get_max_threads()];

#pragma omp parallel
	{
	    perfmon.setup();
	    perfmon.startc = MYRDTSC;

#pragma omp for schedule(dynamic,1)
	    for(int i = 0; i < nblocks; ++i)
		evaluate(xdst + i * BLOCKSIZE * BLOCKSIZE, ydst + i * BLOCKSIZE * BLOCKSIZE, x0s[i], y0s[i], hs[i], root, thetasquared);

	    perfmon.endc = MYRDTSC;
#ifdef _INSTRUMENTATION_
	    perf[omp_get_thread_num()] = perfmon;
#endif
	}

	report_instrumentation(perf, sizeof(perf) / sizeof(*perf), t0, t1, 2577);
    }
}

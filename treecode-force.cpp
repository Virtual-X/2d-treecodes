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

	std::tuple<double, double, double, double> e2p()
	    {
		return std::make_tuple((double)(e2pcycles),
				       (double)(e2pcycles) / tot_cycles(),
				       (double)(e2pcycles) / e2pcalls,
				       (double)(e2pcalls * 192) / e2pcycles);
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
		  const NodeForce & root,
		  const realtype thetasquared)
    {
	const NodeForce * stack[15 * 4 * 2];

	int stackentry = 0, maxentry = 0;

	stack[0] = &root;

	while(stackentry > -1)
	{
	    const NodeForce * const node = stack[stackentry--];

	    realtype tmp[2];

	    const realtype r2 = pow(xt - node->xcom(), 2) + pow(yt - node->ycom(), 2);

	    if (4 * node->r * node->r < thetasquared * r2)
	    {
		int64_t startc = _rdtsc();
		force_e2p(node->mass, xt - node->xcom(), yt - node->ycom(), node->rexpansions, node->iexpansions, tmp, tmp + 1);
		int64_t endc = _rdtsc();

		*xresult += tmp[0];
		*yresult += tmp[1];

		perfmon.e2pcycles += endc - startc;
		++perfmon.e2pcalls;
	    }
	    else
	    {
		if (node->leaf)
		{
		    const int s = node->s;

		    int64_t startc = _rdtsc();
		    force_p2p(&xdata[s], &ydata[s], &vdata[s], node->e - s, xt, yt, tmp, tmp + 1);
		    int64_t endc = _rdtsc();

		    *xresult += tmp[0];
		    *yresult += tmp[1];

		    perfmon.p2pcycles += endc - startc;
		    perfmon.p2pinteractions += node->e - s;
		    ++perfmon.p2pcalls;
		}
		else
		{
		    for(int c = 0; c < 4; ++c)
			stack[++stackentry] = (NodeForce *)node->children[c];

		    maxentry = std::max(maxentry, stackentry);
		}
	    }
	}

	perfmon.maxstacksize = maxentry + 1;
	++perfmon.evaluations;
	perfmon.failed = maxentry >= sizeof(stack) / sizeof(*stack);
    }

    extern "C"
    void treecode_force(const realtype theta,
			const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
			const realtype * const xdst, const realtype * const ydst, const int ndst, realtype * const xresult, realtype * const yresult)
    {
	const realtype thetasquared = theta * theta;

	NodeForce root;

	const double t0 = omp_get_wtime();
	Tree::build(xsrc, ysrc, vsrc, nsrc, xdst, ydst, ndst,  &root, 128);
    	const double t1 = omp_get_wtime();

	xdata = Tree::xdata;
	ydata = Tree::ydata;
	vdata = Tree::vdata;

	PerfMon perf[omp_get_max_threads()];

#pragma omp parallel
	{
	    perfmon.setup();
	    perfmon.startc = _rdtsc();

#pragma omp for schedule(static,1)
	    for(int i = 0; i < ndst; ++i)
		evaluate(xresult + i, yresult + i, xdst[i], ydst[i], root, thetasquared);

	    perfmon.endc = _rdtsc();

	    perf[omp_get_thread_num()] = perfmon;
	}

	for(int i = 0; i < omp_get_max_threads(); ++i)
	    if (perf[i].failed)
	    {
		printf("oops there was an overflow in the computation\n");
		abort();
	    }

	for(int i = 0; i < omp_get_max_threads(); ++i)
	    printf("TID %d: tot cycles: %.3e\n", i, perf[i].tot_cycles());

	for(int i = 0; i < omp_get_max_threads(); ++i)
	{
	    auto p = perf[i].e2p();

	    printf("TID %d: E2P cycles: %.3e (%.1f %%) cycles-per-call: %.1f, ipc: %.2f\n",
		   i, std::get<0>(p), std::get<1>(p) * 100., std::get<2>(p), std::get<3>(p));
	}

	for(int i = 0; i < omp_get_max_threads(); ++i)
	{
	    auto p = perf[i].traversal();

	    printf("TID %d: traversal overhead: %.1f %%, max stacksize: %d \n", i, 100. * std::get<1>(p), std::get<2>(p));
	}

	for(int i = 0; i < omp_get_max_threads(); ++i)
	{
	    auto p = perf[i].p2p();

	    printf("TID %d: P2P importance: %.1f %% cycles-per-interactions: %.1f, cycles-per-call: %.1f\n",
		   i, std::get<1>(p) * 100, std::get<3>(p), std::get<2>(p));
	}

	const double t2 = omp_get_wtime();
	printf("UPWARD: %.2f ms EVAL: %.2f ms (%.1f %%)\n", (t1-t0)*1e3, (t2-t1)*1e3, (t2 - t1) / (t2 - t0) * 100);
    }

}

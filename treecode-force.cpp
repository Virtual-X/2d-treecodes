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

#include "treecode.h"
#include "upward-kernels.h"
#include "force-kernels.h"
#include "upward.h"

namespace EvaluateForce
{
    struct NodeForce : Tree::NodeImplementation<ORDER> { };

    realtype thetasquared, *xdata = nullptr, *ydata = nullptr, *vdata = nullptr;

    struct PerfMon
    {
	int64_t e2pcalls, e2pcycles, p2pcalls, p2pcycles;
	int64_t startc, endc;

	void setup() { e2pcalls = e2pcycles = p2pcalls = p2pcycles = 0; }
	double tot_cycles() { return (double)(endc - startc); }
	std::tuple<double, double, double> e2p() {
	    return std::make_tuple((double)(e2pcycles), (double)e2pcycles / (double)e2pcalls, (double) e2pcalls * 192 / e2pcycles);
	}
    } perfmon;

#pragma omp threadprivate(perfmon)

    void evaluate(realtype * const xresult, realtype * const yresult, const realtype xt, const realtype yt, const NodeForce & node)
    {
	const realtype r2 = pow(xt - node.xcom(), 2) + pow(yt - node.ycom(), 2);

	if (4 * node.r * node.r < thetasquared * r2)
	{
	    int64_t startc = _rdtsc();
	    force_e2p(node.mass, xt - node.xcom(), yt - node.ycom(), node.rexpansions, node.iexpansions, xresult, yresult);
	    int64_t endc = _rdtsc();

	    perfmon.e2pcycles += endc - startc;
	    ++perfmon.e2pcalls;
	}
	else
	{
	    if (node.leaf)
	    {
		const int s = node.s;

		int64_t startc = _rdtsc();
		force_p2p(&xdata[s], &ydata[s], &vdata[s], node.e - s, xt, yt, xresult, yresult);
		int64_t endc = _rdtsc();

		perfmon.p2pcycles += endc - startc;
		++perfmon.p2pcalls;
	    }
	    else
	    {
		realtype xs[4] = {0, 0, 0, 0}, ys[4] = {0, 0, 0, 0};

		for(int c = 0; c < 4; ++c)
		{
		    NodeForce * chd = (NodeForce *)node.children[c];
		    realtype * xptr = xs + c, * yptr = ys + c;

		    evaluate(xptr, yptr, xt, yt, *chd);
		}

		*xresult = xs[0] + xs[1] + xs[2] + xs[3];
		*yresult = ys[0] + ys[1] + ys[2] + ys[3];
	    }
	}
    }


    extern "C"
    void treecode_force(const realtype theta,
			const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
			const realtype * const xdst, const realtype * const ydst, const int ndst, realtype * const xresult, realtype * const yresult)
    {
	thetasquared = theta * theta;

	NodeForce root;

	const double t0 = omp_get_wtime();
	Tree::build(xsrc, ysrc, vsrc, nsrc, xdst, ydst, ndst,  &root);
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
		evaluate(xresult + i, yresult + i, xdst[i], ydst[i], root);

	    perfmon.endc = _rdtsc();

	    perf[omp_get_thread_num()] = perfmon;
	}

	for(int i = 0; i < omp_get_max_threads(); ++i)
	    printf("TID %d: tot cycles: %.3e\n", i, perf[i].tot_cycles());

	for(int i = 0; i < omp_get_max_threads(); ++i)
	    printf("TID %d: E2P cycles: %.3e (%.1f %%) cycles-per-call: %.1f, ipc: %.2f\n", i, std::get<0>(perf[i].e2p()), 100. * perf[i].e2pcycles / perf[i].tot_cycles(), std::get<1>(perf[i].e2p()), std::get<2>(perf[i].e2p()));
	/*
	    printf("TID: %d, c:%3.e  #e2p: %d, e2p c: %.2e, IPC %.2f p2pcalls: %d IPC: %.2f\n",
		   omp_get_thread_num(), (double)(endc - startc), (double)perfmon.e2pcycle
		   perfmon.e2pcalls, 1. / perfmon.e2pcalls * (double)perfmon.e2pcycles, perfmon.e2pcalls * 192 / (double)perfmon.e2pcycles,
		   perfmon.p2pcalls, perfmon.p2pcalls * 3880 / (double)perfmon.p2pcycles);
		   }*/

	const double t2 = omp_get_wtime();
	printf("UPWARD: %.2f ms EVAL: %.2f ms (%.1f %%)\n", (t1-t0)*1e3, (t2-t1)*1e3, (t2 - t1) / (t2 - t0) * 100);

    }

}

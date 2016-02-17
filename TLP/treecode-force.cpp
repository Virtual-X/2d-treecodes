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

#include "treecode-force.h"
#include "upward-kernels.h"
#include "downward-kernels.h"
#include "force-kernels.h"
#include "upward.h"


//#define _INSTRUMENTATION_ 1
#if ORDER <= 12
#define _MIXPREC_
#endif

#ifndef _INSTRUMENTATION_
#define MYRDTSC 0
#else
#define MYRDTSC _rdtsc()
#endif

namespace EvaluateForce
{
    struct NodeForce : Tree::NodeImplementation { };

    realtype *xdata = nullptr, *ydata = nullptr, *vdata = nullptr;
    float *xdata_fp32 = nullptr, *ydata_fp32 = nullptr, *vdata_fp32 = nullptr;

    struct PerfMon
    {
      int64_t e2pcalls, e2lcalls, e2lcycles, e2pcycles, p2pcalls, p2pcycles, p2pinteractions;
	int64_t startc, endc;

	int maxstacksize, evaluations;

	bool failed;

	void setup()
	    {
	      e2lcalls = 0;
	      e2lcycles = 0;
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

      std::tuple<double, double, double, double> e2l(const int instructions)
	    {
	      return std::make_tuple((double)(e2lcycles),
				       (double)(e2lcycles) / tot_cycles(),
				     (double)(e2lcycles) / e2lcalls,
				       (double)(e2lcalls * instructions) / e2lcycles);
	    }
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
		return std::make_tuple((double)(endc - startc - p2pcycles - e2pcycles - e2lcycles),
				       (double)(endc - startc - p2pcycles - e2pcycles - e2lcycles) / (endc - startc),
				       maxstacksize);
	    }

    } perfmon;

//#pragma omp threadprivate(perfmon)
/*
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
*/
long long combi(int n,int k)
	    {
		long long ans = 1;

		k = k > n - k ? n - k : k;
		
		for(int j = 1; j <= k; j++, n--)
		{
		    if(n % j == 0)
		    {
			ans *= n / j;
		    }
		    else
			if(ans % j == 0)
			{
			    ans = ans / j * n;
			}
			else
			{
			    ans = (ans * n) / j;
			}
		}

		return ans;
	    }

#define TMP(a, b) (a)[b]
    
   void naive_downward_e2l(
	const realtype x0s[],
	const realtype y0s[],
	const realtype masses[],
	const realtype * vrxps[],
	const realtype * vixps[],
	const int nexpansions,
	realtype rlocal[],
	realtype ilocal[])
{
    for(int i = 0;  i < nexpansions; ++i)
	{
		const realtype * const rxp = vrxps[i];
		const realtype * const ixp = vixps[i];
		const realtype mass = masses[i];
		
		const realtype x0 = x0s[i];
		const realtype y0 = y0s[i];
		
		const realtype r2z0 = x0 * x0 + y0 * y0;
		realtype rinvz[100], iinvz[100], rcoeff[100], icoeff[100];
		
    		rinvz[1] = x0 / r2z0;
    		iinvz[1] = -y0 / r2z0;

		for(int j = 1; j <= ORDER; ++j)
		{
		    if (j > 1)
		    {
			TMP(rinvz, j) = TMP(rinvz,  (j - 1)) * rinvz[1] - TMP(iinvz,  (j - 1)) * iinvz[1];
			TMP(iinvz, j) = TMP(rinvz,  (j - 1)) * iinvz[1] + TMP(iinvz,  (j - 1)) * rinvz[1];
		    }
		    
		    TMP(rcoeff, j) =  rxp[ (j - 1)] * TMP(rinvz, j) - ixp[ (j - 1)] * TMP(iinvz, j);
		    TMP(icoeff, j) =  rxp[ (j - 1)] * TMP(iinvz, j) + ixp[ (j - 1)] * TMP(rinvz, j);
		}
		
		for(int l = 1; l <= ORDER; ++l)
      		{
		    const realtype prefac =  -mass / l;

		    realtype rtmp = prefac, itmp = 0;
		    
		    for(int k = 1; k <= ORDER; ++k)
		    {
			const realtype BINFAC = combi(l + k - 1, k - 1);
			const int mysign = k % 2 ? -1 : +1;
			rtmp += mysign * BINFAC * TMP(rcoeff, k);
			itmp += mysign * BINFAC * TMP(icoeff, k);
		    }
		    
		    realtype rpartial =  (rtmp ) * TMP(rinvz, l) -  (itmp) * TMP(iinvz, l);
		    realtype ipartial =  (rtmp ) * TMP(iinvz, l) +  (itmp) * TMP(rinvz, l);

			rlocal[l] += (rpartial);
			ilocal[l] += (ipartial);
		}
	}
}

    
    template<int size>
    struct E2LWork
    {
	int count = 0;
	realtype * const rdst,  * const idst;

	realtype x0s[size], y0s[size], masses[size];
	const realtype * rxps[size], *ixps[size];

	E2LWork(realtype * const rlocal, realtype * const ilocal):
	    count(0), rdst(rlocal), idst(ilocal) { }

	void _flush()
	    {
	      const int64_t startc = MYRDTSC;
	      downward_e2l(x0s, y0s, masses, rxps, ixps, count, rdst, idst);
	      
	      const int64_t endc = MYRDTSC;

#ifdef _INSTRUMENTATION_
	      perfmon.e2lcycles += endc - startc;
	      perfmon.e2lcalls += (count + 1) / 2;
#endif
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

#define TILESIZE 4
#define BRICKSIZE 8

    void evaluate(realtype * const xresultbase, realtype * const yresultbase,
		  const realtype x0, const realtype y0, const realtype h,
		  const NodeForce & root, const realtype theta)
    {
	//const bool localexp = true;
	int maxentry = 0;

	const NodeForce * stack[15 * 4 * 2];

	realtype xresult[BRICKSIZE][BRICKSIZE], yresult[BRICKSIZE][BRICKSIZE];

#ifdef _MIXPREC_
	float xresult_fp32[BRICKSIZE][BRICKSIZE], yresult_fp32[BRICKSIZE][BRICKSIZE];
#endif

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

#ifdef _MIXPREC_
		for(int iy = 0; iy < BRICKSIZE; ++iy)
		  for(int ix = 0; ix < BRICKSIZE; ++ix)
		    xresult_fp32[iy][ix] = 0;
		
		for(int iy = 0; iy < BRICKSIZE; ++iy)
		  for(int ix = 0; ix < BRICKSIZE; ++ix)
		    yresult_fp32[iy][ix] = 0;
#endif

		int stackentry = 0;
		stack[0] = &root;

		while(stackentry > -1)
		{
		    const NodeForce * const node = stack[stackentry--];

		    const realtype xcom = node->xcom();
		    const realtype ycom = node->ycom();

		    const realtype distance = sqrt(pow(x0brick - xcom, 2) + pow(y0brick - ycom, 2));

		    const bool localexpansion_converges = (distance / node->r - 1) > (1 / theta) && rbrick <= node->r;

		    if (localexpansion_converges)
			e2lwork.push(xcom - x0brick, ycom - y0brick, node->mass, node->rexpansions, node->iexpansions);
		    else
		    {
			const double xt = std::max(x0 + bx * h, std::min(x0 + (bx + BRICKSIZE - 1) * h, xcom));
			const double yt = std::max(y0 + by * h, std::min(y0 + (by + BRICKSIZE - 1) * h, ycom));

			const realtype r2 = pow(xt - xcom, 2) + pow(yt - ycom, 2);

			if (node->r * node->r < theta * theta * r2)
			{
			    int64_t startc = MYRDTSC;

			    /*  for(int ty = 0; ty < BRICKSIZE; ty += 4)
				for(int tx = 0; tx < BRICKSIZE; tx += 4)
				    reference_force_e2p_tiled(node->mass, x0 + (bx + tx) * h - xcom, y0 + (by + ty) * h - ycom, h,
				    node->rexpansions, node->iexpansions, &xresult[ty][tx], &yresult[ty][tx], BRICKSIZE);*/
			    
			    force_e2p_tiled(node->mass, x0 + (bx + 0) * h - xcom, y0 + (by + 0) * h - ycom, h,
					    node->rexpansions, node->iexpansions, &xresult[0][0],
					    &yresult[0][0]);
			    int64_t endc = MYRDTSC;

#ifdef _INSTRUMENTATION_
			    perfmon.e2pcycles += endc - startc;
			    perfmon.e2pcalls += (BRICKSIZE / TILESIZE) * (BRICKSIZE / TILESIZE);
#endif
			}
			else
			{
			    if (node->leaf)
			    {
				const int s = node->s;

				int64_t startc = MYRDTSC;

				/*for(int ty = 0; ty < BRICKSIZE; ty += 4)
				    for(int tx = 0; tx < BRICKSIZE; tx += 4)
				      {
#ifdef _MIXPREC_
					reference_force_p2p_tiled_mixprec(&xdata_fp32[s], &ydata_fp32[s], &vdata_fp32[s], node->e - s,
								(float)(x0 + (bx + tx) * h), (float)(y0 + (by + ty) * h), (float)h, 
								&xresult_fp32[ty][tx], &yresult_fp32[ty][tx], BRICKSIZE);
#else
					reference_force_p2p_tiled(&xdata[s], &ydata[s], &vdata[s], node->e - s,
							x0 + (bx + tx) * h, y0 + (by + ty) * h, h, 
							&xresult[ty][tx], &yresult[ty][tx], BRICKSIZE);
#endif
}*/
				
				force_p2p_tiled(&xdata[s], &ydata[s], &vdata[s], node->e - s,
						x0 + (bx + 0) * h, y0 + (by + 0) * h, h, 
						&xresult[0][0], &yresult[0][0]);

				int64_t endc = MYRDTSC;

#ifdef _INSTRUMENTATION_
				perfmon.p2pcycles += endc - startc;
				perfmon.p2pinteractions += (node->e - s) * BRICKSIZE * BRICKSIZE;
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
		}

		e2lwork.finalize();
/*
		for(int ty = 0; ty < BRICKSIZE; ty += 4)
		    for(int tx = 0; tx < BRICKSIZE; tx += 4)
			reference_downward_l2p_tiled(h * (tx - 0.5 * (BRICKSIZE - 1)),
					   h * (ty - 0.5 * (BRICKSIZE - 1)),
					   h, rlocal, ilocal,
					   &xresult[ty][tx], &yresult[ty][tx], BRICKSIZE);
*/

		downward_l2p_tiled(h * (0 - 0.5 * (BRICKSIZE - 1)),
				   h * (0 - 0.5 * (BRICKSIZE - 1)),
				   h, rlocal, ilocal,
				   &xresult[0][0], &yresult[0][0]);
#ifdef _MIXPREC_
		for(int iy = 0; iy < BRICKSIZE; ++iy)
		  for(int ix = 0; ix < BRICKSIZE; ++ix)
		    xresult[iy][ix] += xresult_fp32[iy][ix];
		
		for(int iy = 0; iy < BRICKSIZE; ++iy)
		  for(int ix = 0; ix < BRICKSIZE; ++ix)
		    yresult[iy][ix] += yresult_fp32[iy][ix];	
#endif

		for(int iy = 0; iy < BRICKSIZE; ++iy)
		    for(int ix = 0; ix < BRICKSIZE; ++ix)
			xresultbase[bx + ix + BLOCKSIZE * (by + iy)] = xresult[iy][ix];

		for(int iy = 0; iy < BRICKSIZE; ++iy)
		    for(int ix = 0; ix < BRICKSIZE; ++ix)
			yresultbase[bx + ix + BLOCKSIZE * (by + iy)] = yresult[iy][ix];
	    }

#ifdef _INSTRUMENTATION_
	perfmon.maxstacksize = maxentry + 1;
	perfmon.evaluations += BRICKSIZE * BRICKSIZE;
	perfmon.failed = maxentry >= sizeof(stack) / sizeof(*stack);
#endif
    }

  void report_instrumentation(PerfMon perf[], const int N, const double t0, const double t1, 
			      const int e2linstructions, const int e2pinstructions)
    {
#ifdef _INSTRUMENTATION_
#if _INSTRUMENTATION_ == 2
      for(int i = 0; i < N; ++i)
	if (perf[i].failed)
	  {
	    printf("oops there was an overflow in the computation\n");
	    abort();
	  }

      printf("EVALUATION CYCLES ===============================\n");
	for(int i = 0; i < N; ++i)
	    printf("TID %d: tot cycles: %.3e\n", i, perf[i].tot_cycles());

printf("DOWNWARD CYCLES ===============================\n");
	for(int i = 0; i < N; ++i)
	{
	    auto p = perf[i].e2l(e2linstructions);

	    printf("TID %d: E2L cycles: %.3e (%.1f %%) cycles-per-call: %.1f, ipc: %.2f\n",
		   i, std::get<0>(p), std::get<1>(p) * 100., std::get<2>(p), std::get<3>(p));
	}

printf("E2P CYCLES ===============================\n");
	for(int i = 0; i < N; ++i)
	{
	    auto p = perf[i].e2p(e2pinstructions);

	    printf("TID %d: E2P cycles: %.3e (%.1f %%) cycles-per-call: %.1f, ipc: %.2f\n",
		   i, std::get<0>(p), std::get<1>(p) * 100., std::get<2>(p), std::get<3>(p));
	}

printf("EVALUATION P2P CYCLES ===============================\n");

	for(int i = 0; i < N; ++i)
	{
	    auto p = perf[i].p2p();

	    printf("TID %d: P2P importance: %.1f %% cycles-per-interactions: %.1f, cycles-per-call: %.1f\n",
		   i, std::get<1>(p) * 100, std::get<3>(p), std::get<2>(p));
	}

printf("EVALUATION TRAVERSAL CYCLES ===============================\n");
	for(int i = 0; i < N; ++i)
	{
	    auto p = perf[i].traversal();

	    printf("TID %d: traversal overhead: %.1f %%, max stacksize: %d \n", i, 100. * std::get<1>(p), std::get<2>(p));
	}

#endif
      
	const double t2 = omp_get_wtime();
	printf("UPWARD: %.2f ms EVAL: %.2f ms (%.1f %%)\n", (t1-t0)*1e3, (t2-t1)*1e3, (t2 - t1) / (t2 - t0) * 100);
#endif
    }
/*
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

	Tree::dispose();

	//report_instrumentation(perf, sizeof(perf) / sizeof(*perf), t0, t1, 0, E2P_IC);
    }
*/
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
	NodeForce root;

	const double t0 = omp_get_wtime();
	Tree::build(xsrc, ysrc, vsrc, nsrc, &root, 192); //before: 128
	const double t1 = omp_get_wtime();

	xdata = Tree::xdata;
	ydata = Tree::ydata;
	vdata = Tree::vdata;

#ifdef _MIXPREC_
	posix_memalign((void **)&xdata_fp32, 32, sizeof(float) * nsrc);
	posix_memalign((void **)&ydata_fp32, 32, sizeof(float) * nsrc);
	posix_memalign((void **)&vdata_fp32, 32, sizeof(float) * nsrc);

	#pragma omp parallel for
	for(int i = 0; i < nsrc; ++i)
	  {
	    xdata_fp32[i] = (float)xdata[i];
	    ydata_fp32[i] = (float)ydata[i];
	    vdata_fp32[i] = (float)vdata[i];
	  }
#endif
	PerfMon perf[omp_get_max_threads()];

#pragma omp parallel
	{
	    perfmon.setup();
	    perfmon.startc = MYRDTSC;

#pragma omp for schedule(dynamic,1)
	    for(int i = 0; i < nblocks; ++i)
		evaluate(xdst + i * BLOCKSIZE * BLOCKSIZE, ydst + i * BLOCKSIZE * BLOCKSIZE, x0s[i], y0s[i], hs[i], root, theta);

	    perfmon.endc = MYRDTSC;
#ifdef _INSTRUMENTATION_
	    perf[omp_get_thread_num()] = perfmon;
#endif
	}
	
#ifdef _MIXPREC_
	free(xdata_fp32);
	free(ydata_fp32);
	free(vdata_fp32);
#endif
	Tree::dispose();

	//report_instrumentation(perf, sizeof(perf) / sizeof(*perf), t0, t1, E2L_TILED_IC, E2P_TILED_IC);
    }
}

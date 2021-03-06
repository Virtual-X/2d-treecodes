/*
 *  main.cpp
 *  Part of 2d-treecodes
 *
 *  Created and authored by Diego Rossinelli on 2015-09-25.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <unistd.h>
#include <omp.h>

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cstring>

#include <algorithm>
#include <limits>
#include <vector>

#include "2d-treecodes-force.h"
#include "2d-treecodes-potential.h"

double tol = 1e-8;

void check(const double * ref, const double * res, const int N)
{
    double linf = 0, l1 = 0, linf_rel = 0, l1_rel = 0;

    for(int i = 0; i < N; ++i)
    {
	assert(!std::isnan(ref[i]));
	assert(!std::isnan(res[i]));

	const double err = ref[i] - res[i];
	const double maxval = std::max(fabs(res[i]), fabs(ref[i]));
	const double relerr = err/std::max(1e-6, maxval);

	if (fabs(relerr) >= tol && fabs(err) >= tol)
	    printf("%d: %e ref: %e -> %e %e\n", i, res[i], ref[i], err, relerr);

	assert(fabs(relerr) < tol || fabs(err) < tol);

	l1 += fabs(err);
	l1_rel += fabs(relerr);

	linf = std::max(linf, fabs(err));
	linf_rel = std::max(linf_rel, fabs(relerr));
    }

    printf("l-infinity errors: %.03e (absolute) %.03e (relative)\n", linf, linf_rel);
    printf("       l-1 errors: %.03e (absolute) %.03e (relative)\n", l1, l1_rel);
}

void test(double theta, double tol, FILE * f = NULL, bool potential = true, bool verify = true, bool mragfile = false)
{
    int NSRC = 1e5 * (0.1 + drand48());

    if (f)
	fread(&NSRC, sizeof(int), 1, f);

    double * xsrc = new double[NSRC];
    double * ysrc = new double[NSRC];
    double * sources = new double[NSRC];

    if (f)
    {
	fread(xsrc, sizeof(double), NSRC, f);
	fread(ysrc, sizeof(double), NSRC, f);
	fread(sources, sizeof(double), NSRC, f);
    }
    else
	for(int i = 0; i < NSRC; ++i)
	{
	    xsrc[i] = (drand48() - 0.25) * 10;
	    ysrc[i] = (drand48() - 0.75) * 20;
	    sources[i] = drand48() * 55;
	}

    int NDST = 3000 * (0.1 + drand48());

    if (f)
	fread(&NDST, sizeof(int), 1, f);

    int NBLOCKS = 0;
    const int BS2 = BLOCKSIZE * BLOCKSIZE;
    double * x0s = nullptr, *y0s = nullptr, *hs = nullptr;

    if (mragfile)
    {
	NBLOCKS = NDST;
	NDST = NBLOCKS * BS2;
	x0s = new double[NBLOCKS];
	y0s = new double[NBLOCKS];
	hs = new double[NBLOCKS];
    }

    double * xdst = new double[NDST];
    double * ydst = new double[NDST];
    double * xref = new double[NDST];
    double * yref = new double[NDST];

    if (f)
    {
	if (mragfile)
	{
	    printf("mrag file!\n");
	    fread(x0s, sizeof(double), NBLOCKS, f);
	    fread(y0s, sizeof(double), NBLOCKS, f);
	    fread(hs, sizeof(double), NBLOCKS, f);

	    for(int b =0; b < NBLOCKS; b++)
		for(int iy = 0; iy < BLOCKSIZE; ++iy)
		    for(int ix = 0; ix < BLOCKSIZE; ++ix)
		    {
			xdst[ix + BLOCKSIZE * iy + BS2 * b] = x0s[b] + ix * hs[b];
			ydst[ix + BLOCKSIZE * iy + BS2 * b] = y0s[b] + iy * hs[b];
		    }
	}
	else //if (!mragfile)
	{
	    fread(xdst, sizeof(double), NDST, f);
	    fread(ydst, sizeof(double), NDST, f);
	    fread(xref, sizeof(double), NDST, f);
	}

	if (!potential)
	{
	    fread(yref, sizeof(double), NDST, f);

	    for(int i = 0; i < NDST; ++i)
	    {
		const double tmp0 = xref[i] * (2 * M_PI);
		const double tmp1 = yref[i] * (2 * M_PI);

		xref[i] = -tmp1;
		yref[i] = tmp0;
	    }
	}
    }
    else
    {
	for(int i = 0; i < NDST; ++i)
	{
	    xdst[i] = (drand48() - 0.25) * 10;
	    ydst[i] = (drand48() - 0.75) * 20;
	}
    }

    const bool mrag = mragfile || NDST % BS2 == 0;

    if (mrag && !mragfile)
    {
	NBLOCKS = NDST / BS2;
	printf("ndst: %d i have found %d blocks\n", NDST, NBLOCKS);
	assert(NDST % BS2 == 0);

	x0s = new double[NBLOCKS];
	y0s = new double[NBLOCKS];
	hs = new double[NBLOCKS];

	for(int i = 0; i < NBLOCKS; ++i)
	{
	    x0s[i] = xdst[i * BS2];
	    y0s[i] = ydst[i * BS2];
	    hs[i] = xdst[i * BS2 + 1] - xdst[i * BS2];
	}
    }

    const double eps = std::numeric_limits<double>::epsilon() * 10;

    double * xtargets = new double[NDST];
    double * ytargets = new double[NDST];

    printf("Testing %s with %d sources and %d targets (theta %.3e)...\n", (potential ? "POTENTIAL" : "FORCE"), NSRC, NDST, theta);
    const double tstart = omp_get_wtime();
    
    if (potential)
	treecode_potential(theta, xsrc, ysrc, sources, NSRC, xdst, ydst, NDST, xtargets);
    else
	if (mrag)
	{
	    printf("MRAG LAYOUT\n");
	    treecode_force_mrag(theta, xsrc, ysrc, sources, NSRC, x0s, y0s, hs, NBLOCKS, xtargets, ytargets);
	}
	else
	{
	    printf("This file format is not support for velocities.\n");
	    abort();
	}
    
    const double tend = omp_get_wtime();

    printf("\x1b[94msolved in %.2f ms\x1b[0m\n", (tend - tstart) * 1e3);

    if (verify)
    {
	const int OFFSET = 7;
	const int JUMP = 433;

	if (potential)
#pragma omp parallel for
	    for(int i = OFFSET; i < NDST; i += JUMP)
	    {
		const double xd = xdst[i];
		const double yd = ydst[i];

		double s = 0;

		for(int j = 0; j < NSRC; ++j)
		{
		    const double xr = xd - xsrc[j];
		    const double yr = yd - ysrc[j];
		    const double r = sqrt(xr * xr + yr * yr + eps);
		    s += log(r) * sources[j];
		}

		xref[i] = s;
	    }
	else
#pragma omp parallel for
	    for(int i = OFFSET; i < NDST; i += JUMP)
	    {
		const double xd = xdst[i];
		const double yd = ydst[i];

		double xs = 0, ys = 0;

		for(int j = 0; j < NSRC; ++j)
		{
		    const double xr = xd - xsrc[j];
		    const double yr = yd - ysrc[j];
		    const double factor = sources[j] / (xr * xr + yr * yr + eps);
		    xs += xr * factor;
		    ys += yr * factor;
		}

		xref[i] = xs;
		yref[i] = ys;
	    }

	if (verify)
	{
	    std::vector<double> a, b, c, d;

	    for(int i = OFFSET; i < NDST; i += JUMP)
	    {
		a.push_back(xref[i]);
		b.push_back(xtargets[i]);
		c.push_back(yref[i]);
		d.push_back(ytargets[i]);
	    }

	    check(&a[0], &b[0], a.size());

	    if (!potential)
		check(&c[0], &d[0], c.size());
	}
    }

    if (mrag)
    {
	delete [] x0s;
	delete [] y0s;
	delete [] hs;
    }

    delete [] xdst;
    delete [] ydst;

    delete [] xtargets;
    delete [] ytargets;

    delete [] xref;
    delete [] yref;

    delete [] xsrc;
    delete [] ysrc;
    delete [] sources;

    printf("TEST PASSED.\n");
}

int main(int argc, char ** argv)
{
    srand48(1451);

    double theta = 1;
    bool verify = true;

    if (argc > 1)
	theta = atof(argv[1]);

    if (argc > 2)
	tol = atof(argv[2]);

    if (argc > 3)
	verify = strcmp(argv[3], "profile") != 0;

    enum TestType { V_TEST=1, P_TEST=2, PV_TEST=3};

    auto file2test = [&] (const char * filename, bool mragfile = false, TestType testt = PV_TEST)
	{
	    if (access(filename, R_OK) == -1)
	    {
		printf("WARNING: reference file <%s> not found, i skip this test.\n", filename);
		return;
	    }
	    else
		printf("reading from <%s> ...\n", filename);

	    FILE * fin = fopen(filename, "r");

	    assert(fin && sizeof(double) == sizeof(double));

	    if (!mragfile && (testt & P_TEST))
		test(theta, tol, fin, true, verify);

	    fseek(fin, 0, SEEK_SET);

	    if (testt & V_TEST)
		test(theta, tol * 100, fin, false, verify, mragfile);
	    fclose(fin);
	};

    file2test("testDiego/diegoBinaryN400", false, P_TEST);
    file2test("testDiego/diegoBinaryN2000", false, P_TEST);
    file2test("testDiego/diegoBinaryN12000", false, P_TEST);
    
    file2test("diegoVel/velocityPoissonFishLmax6", false, V_TEST);
    file2test("diegoVel/velocityPoissonCylUnif2048", false, V_TEST);
    file2test("diegoVel/velocityPoissonFishLmax8Early", false, V_TEST);
    file2test("diegoVel/velocityPoissonFishLmax8Late", false, V_TEST);
    
    file2test("testSid/diegoSolverCylUniform", true, V_TEST);
    file2test("testSid/diegoSolverAdaptiveGrid", true, V_TEST);
    file2test("testSid/diegoVelTestsDec10", true, V_TEST);
    file2test("testSid/diegoVelTestsDec14", true, V_TEST);

#if 0
    for(int itest = 0; itest < 10; ++itest)
    {
	printf("test nr %d\n", itest);
	test(theta, tol, NULL, true, verify);
	test(theta, tol * 100, NULL, false, verify);
    }
#endif
}

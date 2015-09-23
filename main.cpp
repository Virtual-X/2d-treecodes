#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>

#include <algorithm>
#include <limits>

#include "treecode.h"

void test(FILE * f = NULL)
{
    const realtype eps = std::numeric_limits<realtype>::epsilon() * 10;
    
    int NSRC = 1e5;
    
    if (f)
	fread(&NSRC, sizeof(int), 1, f);
        
    realtype * xsrc = new realtype[NSRC];
    realtype * ysrc = new realtype[NSRC];
    realtype * sources = new realtype[NSRC];

    if (f)
    {
	fread(xsrc, sizeof(realtype), NSRC, f);
	fread(ysrc, sizeof(realtype), NSRC, f);
	fread(sources, sizeof(realtype), NSRC, f);
    }
    else
	for(int i = 0; i < NSRC; ++i)
	{
	    xsrc[i] = (drand48() - 0.25) * 10;
	    ysrc[i] = (drand48() - 0.75) * 20;
	    sources[i] = drand48() * 55;
	}
	
    int NDST = 300;
    
    if (f)
	fread(&NDST, sizeof(int), 1, f);

    printf("ndst is %d\n", NDST);
    
    realtype * xdst = new realtype[NDST];
    realtype * ydst = new realtype[NDST];
    realtype * ref = new realtype[NDST];

    if (f)
    {
	fread(xdst, sizeof(realtype), NDST, f);
	fread(ydst, sizeof(realtype), NDST, f);
	fread(ref, sizeof(realtype), NDST, f);
    }
    else
    {
	for(int i = 0; i < NDST; ++i)
	{
	    xdst[i] = (drand48() - 0.25) * 10;
	    ydst[i] = (drand48() - 0.75) * 20;
	}
    }

//#pragma omp parallel for
    for(int i = 0; i < NDST; ++i)
    {
	const realtype xd = xdst[i];
	const realtype yd = ydst[i];
	printf("xd yd: %f %f\n", xd, yd);
	realtype s = 0;
	
	for(int j = 0; j < NSRC; ++j)
	{
	    const realtype xr = xd - xsrc[j];
	    const realtype yr = yd - ysrc[j];
	    const realtype r = sqrt(xr * xr + yr * yr + eps);
	    s += log(r) * sources[j]; 
	}
	
	ref[i] = s;
    }
   
	
    realtype * targets = new realtype[NDST];

    printf("tree code started nsrc: %d ndst: %d...\n", NSRC, NDST);
    treecode_potential(1.0, xsrc, ysrc, sources, NSRC, xdst, ydst, NDST, targets);
    printf("tree coded ended.\n");

    for(int i = 0; i < NDST; ++i)
	targets[i] /= 2.0 * M_PI;
    
    const bool verbose = false;
    const double tol = 1e-8;//std::numeric_limits<realtype>::epsilon() * 10;
    for(int i = 0; i < NDST; ++i)
    {
	assert(!std::isnan(ref[i]));
	assert(!std::isnan(targets[i]));
	    
	const double err = ref[i] - targets[i];
	const double relerr = err/std::max(1e-6, std::max(fabs(targets[i]), fabs(ref[i]))); 
	    
	if (verbose) printf("+%1.1e,", relerr);
	    
	if (fabs(relerr) >= tol && fabs(err) >= tol)
	    printf("%d: %e ref: %e -> %e %e\n", i, targets[i], ref[i], err, relerr);
	    
	//assert(fabs(relerr) < tol || fabs(err) < tol);
    }
	
    delete [] xdst;
    delete [] ydst;
    delete [] targets;
    delete [] ref;
	
    delete [] xsrc;
    delete [] ysrc;
    delete [] sources;
}

int main()
{
    srand48(1451);
    
    FILE * fin = fopen("diegoBinaryN400", "r");
    assert(fin && sizeof(realtype) == sizeof(double));
    test(fin);
    fclose(fin);
    /*
    for(int itest = 0; itest < 100; ++itest)
    {
	printf("test nr %d\n", itest);
	test();
	}*/
}

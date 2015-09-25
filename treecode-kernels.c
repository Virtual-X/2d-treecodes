#include <math.h>
#include <immintrin.h>
#include "treecode.h"

#define N2SIZE 64
#define EPS (10 * __DBL_EPSILON__)

#if 1
#define mysqrt(x)  __builtin_ia32_sqrtpd256(x)
#else
#define mysqrt(x) __builtin_ia32_sqrtpd(x)
#endif

inline v4 mylog(const v4 x)
{
    v4 y;
    for(int i = 0; i < 4; ++i)
	y[i] = log(x[i]);

    return y;
}


realtype treecode_n2(const realtype * __restrict__ const _xsrc,
		     const realtype * __restrict__ const _ysrc,
		     const realtype * __restrict__ const _vsrc,
		     const int nsources, 
		     const realtype xt,
		     const realtype yt)
{
    enum { NACC = 32 }; 

    realtype s[NACC];
    for(int i = 0; i < NACC; ++i)
	s[i] = 0;

    const int nnice = NACC * (nsources / NACC);
    
    for(int i = 0; i < nnice; i += NACC)
    {	
	const realtype * const xsrc = _xsrc + i;
	const realtype * const ysrc = _ysrc + i; 
	const realtype * const vsrc = _vsrc + i;

#pragma ivdep
#pragma unroll
	for(int j = 0; j < NACC; ++j)
	{
	    const realtype xr = xt - xsrc[j];
	    const realtype yr = yt - ysrc[j];
	    
	    s[j] += log(sqrt(xr * xr + yr * yr + EPS)) * vsrc[j];
	}
    }

    realtype sum = 0;

    for(int i = nnice; i < nsources; ++i)
    {
	 const realtype xr = xt - _xsrc[i];
	 const realtype yr = yt - _ysrc[i];
	    
	 sum += log(sqrt(xr * xr + yr * yr + EPS)) * _vsrc[i];
    }

    for(int i = 0; i < NACC; ++i)
	sum += s[i];
    
    return sum;
}

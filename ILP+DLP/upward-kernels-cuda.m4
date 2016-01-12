include(unroll.m4)
divert(-1)
define(WARPSIZE, 32)
define(P2E_KERNEL, upward_p2e_order$1)
divert(0) dnl dnl dnl

ifelse(eval(WARPSIZE >= ORDER), 1, ,`
`#'if WARPSIZE < `ORDER'
`#'error `ORDER' should be lessequal than WARPSIZE
#endif')dnl dnl
#include <cstdio>

__device__ void P2E_KERNEL(ORDER)(const realtype xcom,
	const realtype ycom,
	const realtype * __restrict__ const xsources,
	const realtype * __restrict__ const ysources,
	const realtype * __restrict__ const vsources,
	const int nsources,
	realtype * __restrict__ const rexpansions,
	realtype * __restrict__ const iexpansions)
{
	
	realtype LUNROLL(n, 0, eval(ORDER - 1),`ifelse(n,0,,`,')
		TMP(rxp, n) = 0')
	LUNROLL(n, 0, eval(ORDER - 1),`,
		TMP(ixp, n) = 0');

	const int tid = threadIdx.x;

	for(int i = tid; i < nsources; i += WARPSIZE)
	{
		const realtype rprod_0 = xsources[i] - xcom; 
		const realtype iprod_0 = ysources[i] - ycom;

		const realtype src = vsources[i]; 

		TMP(rxp, 0) -= rprod_0 * src;
		TMP(ixp, 0) -= iprod_0 * src;

		LUNROLL(n, 1, eval(ORDER - 1),`
		const realtype TMP(rprod, n) = TMP(rprod, eval(n - 1)) * TMP(rprod, 0) - TMP(iprod, eval(n - 1)) * TMP(iprod, 0);
		const realtype TMP(iprod, n) = TMP(rprod, eval(n - 1)) * TMP(iprod, 0) + TMP(iprod, eval(n - 1)) * TMP(rprod, 0);

		const realtype TMP(term, n) = src / eval(n+1).f;

		TMP(rxp, n) -= TMP(rprod, n) * TMP(term, n);
		TMP(ixp, n) -= TMP(iprod, n) * TMP(term, n);
		')
	}dnl

	SEQ(`LUNROLL(n, 0, eval(ORDER - 1),`
	TMP(rxp, n) += __shfl_xor( TMP(rxp, n), L );
	TMP(ixp, n) += __shfl_xor( TMP(ixp, n), L );')
	', L, 16, 8, 4, 2, 1)

	realtype rval, ival;
	switch(tid)
	{
	LUNROLL(n, 0, eval(ORDER - 1),`
	case n:
		rval = TMP(rxp, n);
		ival = TMP(ixp, n);
		break;
		')
	default:
		break;
	}

	/*if (tid == 0)
	{
	LUNROLL(n, 0, eval(ORDER - 1),`
		rexpansions[n] = TMP(rxp, n);
		iexpansions[n] = TMP(ixp, n);
		');
	}*/
	if (tid < ORDER)
	{
		rexpansions[tid] = rval;
		iexpansions[tid] = ival;
	}
	
}

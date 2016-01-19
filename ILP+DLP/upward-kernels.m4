#include <cassert>

include(unroll.m4)
divert(-1)
define(WARPSIZE, 32)
divert(0) dnl dnl dnl

ifelse(eval(WARPSIZE >= ORDER), 1, ,`
`#'if WARPSIZE < `ORDER'
`#'error `ORDER' should be lessequal than WARPSIZE
#endif')dnl dnl
#include <cstdio>

__device__ void print_message()
{
printf("hello again from ILPDLP order %d\n", ORDER);
}

__device__ void upward_p2e(const realtype xcom,
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

	if (tid < ORDER)
	{
		rexpansions[tid] = rval;
		iexpansions[tid] = ival;
	}
	
}

define(`ARY', `$1[tid + 4 * eval($2 - 1)]')
define(`ARYFP', eval(4 * (ORDER )))

__device__ void upward_e2e(
	const realtype x0,
	const realtype y0,
	const realtype mass,
	const realtype * __restrict__ const rsrcxp,
	const realtype * __restrict__ const isrcxp,
	realtype * __restrict__ const rdstxp,
	realtype * __restrict__ const idstxp)
{
	const int tid = threadIdx.x;
	assert(tid < 4);

	const int slot = threadIdx.y;
	extern __shared__ realtype ary[];
	realtype * const rinvz = ary + ARYFP * (0 + 4 * slot);
	realtype * const iinvz = ary + ARYFP * (1 + 4 * slot);
	realtype * const rcoeff = ary + ARYFP * (2 + 4 * slot);
	realtype * const icoeff = ary + ARYFP * (3 + 4 * slot);
	
	const realtype r2z0 = x0 * x0 + y0 * y0;
	ARY(rinvz, 1) = x0 / r2z0;
	ARY(iinvz, 1) = - y0 / r2z0;
	dnl
	LUNROLL(j, 1, eval(ORDER),`
	ifelse(j, 1, , `
	ARY(rinvz, j) = ARY(rinvz, eval(j - 1)) * ARY(rinvz, 1) - ARY(iinvz, eval(j - 1)) * ARY(iinvz, 1);
	ARY(iinvz, j) = ARY(rinvz, eval(j - 1)) * ARY(iinvz, 1) + ARY(iinvz, eval(j - 1)) * ARY(rinvz, 1);')
	ARY(rcoeff, j) = rsrcxp[eval(j - 1)] * ARY(rinvz, j) - isrcxp[eval(j - 1)] * ARY(iinvz, j);
	ARY(icoeff, j) = rsrcxp[eval(j - 1)] * ARY(iinvz, j) + isrcxp[eval(j - 1)] * ARY(rinvz, j);
	')

	LUNROLL(l, 1, eval(ORDER),`
	{
		const realtype TMP(prefac, l) = ifelse(l, 1, `- mass',`mass * esyscmd(echo -1/eval(l) | bc --mathlib )');
		pushdef(`BINFAC', `BINOMIAL(eval(l - 1), eval(k - 1)).f')
		const realtype TMP(rtmp, l) = TMP(prefac, l) LUNROLL(k, 1, l,` + ifelse(BINFAC,1.f,,`BINFAC * ') ARY(rcoeff, k)');
		const realtype TMP(itmp, l) = LUNROLL(k, 1, l,` ifelse(k,1,,+) ifelse(BINFAC,1.f,,`BINFAC * ') ARY(icoeff, k)');
		popdef(`BINFAC')dnl

		const realtype TMP(invz2, l) = ARY(rinvz, l) * ARY(rinvz, l) + ARY(iinvz, l) * ARY(iinvz, l);
		const realtype TMP(invinvz2, l) = TMP(invz2, l) ? 1 / TMP(invz2, l) : 0;
		const realtype TMP(rz, l) = ARY(rinvz, l) * TMP(invinvz2, l);
		const realtype TMP(iz, l) = - ARY(iinvz, l) * TMP(invinvz2, l);

		realtype rpartial = TMP(rtmp, l) * TMP(rz, l) - TMP(itmp, l) * TMP(iz, l);
		realtype ipartial = TMP(rtmp, l) * TMP(iz, l) + TMP(itmp, l) * TMP(rz, l);

		SEQ(`
		rpartial += __shfl_xor(rpartial, L);
		ipartial += __shfl_xor(ipartial, L);', L, 2, 1)

		//realtype tmp0, tmp1;
		if (tid == 0)
		{
			rdstxp[eval(l - 1)] = rpartial;
			idstxp[eval(l - 1)] = ipartial;
		}
	}')
}

#include <cassert>

include(unroll.m4)
divert(-1)
define(WARPSIZE, 32)
divert(0) dnl dnl dnl

ifelse(eval(WARPSIZE >= ORDER), 1, ,`
`#'if WARPSIZE < `ORDER'
`#'error `ORDER' should be lessequal than WARPSIZE
`#'endif')
dnl dnl
#include <cstdio>
#define ACCESS(x) __ldg(&(x)) 

__device__ void print_message()
{
printf("hello again from ILPDLP order %d\n", ORDER);
}

define(`ARY', `$1[tid + 4 * eval($2 - 1)]')
define(`ARYFP', eval(4 * ORDER))

extern __shared__ realtype ary[];

__device__ void upward_p2e(const realtype xcom,
	const realtype ycom,
	const realtype * __restrict__ const xsources,
	const realtype * __restrict__ const ysources,
	const realtype * __restrict__ const vsources,
	const int nsources,
	realtype * __restrict__ const rexpansions,
	realtype * __restrict__ const iexpansions)
{		
	const int tid = threadIdx.x;
	const int slot = threadIdx.y;

	realtype * const rxp = ary + ARYFP * (0 + 4 * slot);
	realtype * const ixp = ary + ARYFP * (1 + 4 * slot);

	for(int i = tid; i < eval(ARYFP * 2); i += 32)
		rxp[i] = 0;

	for(int i = tid; i < nsources; i += WARPSIZE)
	{		
		const realtype rprod_0 = ACCESS(xsources[i]) - xcom; 
		const realtype iprod_0 = ACCESS(ysources[i]) - ycom;

		const realtype src = ACCESS(vsources[i]); 

		realtype rtmp = rprod_0 * src;
		realtype itmp = iprod_0 * src;
		
		WARPSUM(rtmp, itmp)

		if (tid == 0)
		{
		    rxp[0] -= rtmp;
		    ixp[0] -= itmp;
		}

		realtype rprod = rprod_0, iprod = iprod_0;
		
		LUNROLL(n, 1, eval(ORDER - 1),`
		rtmp = rprod * TMP(rprod, 0) - iprod * TMP(iprod, 0);
		itmp = rprod * TMP(iprod, 0) + iprod * TMP(rprod, 0);

		const realtype TMP(term, n) = src * (realtype)(1 / eval(n+1).);

		rprod = rtmp;
		iprod = itmp;
		
		rtmp = rprod * TMP(term, n);
		itmp = iprod * TMP(term, n);

		WARPSUM(rtmp, itmp)

		if (tid == 0)
		{
		    rxp[n] -= rtmp;
		    ixp[n] -= itmp;	
		}
		')
	}dnl

	if (tid < ORDER)
	{
	   rexpansions[tid] = rxp[tid];
	   iexpansions[tid] = ixp[tid];
	}
}



__device__ void upward_e2e(
	const realtype x0,
	const realtype y0,
	const realtype mass,
	const realtype * __restrict__ const rsrcxp,
	const realtype * __restrict__ const isrcxp,
	realtype * __restrict__ const rdstxp,
	realtype * __restrict__ const idstxp)
{
	const int slot = threadIdx.y;
	const int tid = threadIdx.x;
	assert(tid < 4);

	realtype * const rinvz = ary + ARYFP * (0 + 4 * slot);
	realtype * const iinvz = ary + ARYFP * (1 + 4 * slot);
	realtype * const rcoeff = ary + ARYFP * (2 + 4 * slot);
	realtype * const icoeff = ary + ARYFP * (3 + 4 * slot);
	
	realtype r2z0 = x0 * x0 + y0 * y0;

	if (r2z0 == 0)
	   r2z0 = 1;

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
		const realtype TMP(rtmp, l) = TMP(prefac, l) LUNROLL(k, 1, l,`
		+ ARY(rcoeff, k) ifelse(BINFAC,1.f,,`* BINFAC')');
		
		const realtype TMP(itmp, l) = LUNROLL(k, 1, l,`
		ifelse(k,1,,+)  ARY(icoeff, k) ifelse(BINFAC,1.f,,`* BINFAC')');
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

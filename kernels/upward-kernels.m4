/*
 *  upward-kernels.m4, upward-kernels.ispc
 *  Part of 2d-treecodes
 *
 *  Created and authored by Diego Rossinelli on 2015-11-25.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

include(unroll.m4)
divert(-1)
define(GANGSIZE, 8)
divert(0) dnl dnl dnl

extern "C" void upward_p2e(
       const  realtype xsources[],
       const  realtype ysources[],
       const  realtype vsources[],
       const  int nsources,
       const  realtype xcom,
       const  realtype ycom,
        realtype rexpansions[],
        realtype iexpansions[])
{
	realtype LUNROLL(n, 0, eval(ORDER - 1),`ifelse(n,0,,`,')
            TMP(rxp, n) = 0, TMP(ixp, n) = 0');

#if 0
	    foreach(i = 0 ... nsources)
	{
		const realtype rprod_0 = xsources[i] - xcom;
		const realtype iprod_0 = ysources[i] - ycom;

		const realtype src = vsources[i];

		TMP(rxp, 0) -= rprod_0 * src;
		TMP(ixp, 0) -= iprod_0 * src;

		LUNROLL(n, 1, eval(ORDER - 1),`
		const realtype TMP(rprod, n) = TMP(rprod, eval(n - 1)) * TMP(rprod, 0) - TMP(iprod, eval(n - 1)) * TMP(iprod, 0);
		const realtype TMP(iprod, n) = TMP(rprod, eval(n - 1)) * TMP(iprod, 0) + TMP(iprod, eval(n - 1)) * TMP(rprod, 0);

		const realtype TMP(term, n) = src * (realtype)(esyscmd(echo 1/eval(n + 1) | bc --mathlib ));

		TMP(rxp, n) -= TMP(rprod, n) * TMP(term, n);
		TMP(ixp, n) -= TMP(iprod, n) * TMP(term, n);
		')
	}
#else

	for(int i = 0; i < nsources; ++i)
 	{
 		const realtype rprod_0 = xsources[i] - xcom;
 		const realtype iprod_0 = ysources[i] - ycom;

 		const realtype src = vsources[i];

 		realtype rtmp = rprod_0 * src;
 		realtype itmp = iprod_0 * src;

 		TMP(rxp, 0) -= rtmp;
 		TMP(ixp, 0) -= itmp;

 		realtype rprod = rprod_0, iprod = iprod_0;

 		LUNROLL(n, 1, eval(ORDER - 1),`
 		rtmp = rprod * TMP(rprod, 0) - iprod * TMP(iprod, 0);
 		itmp = rprod * TMP(iprod, 0) + iprod * TMP(rprod, 0);

 		const realtype TMP(term, n) = src * (realtype)(1 / eval(n+1).);

 		rprod = rtmp;
 		iprod = itmp;

 		rtmp = rprod * TMP(term, n);
 		itmp = iprod * TMP(term, n);

 		TMP(rxp, n) -= rtmp;
 		TMP(ixp, n) -= itmp;
 		')
 	}
#endif
	LUNROLL(i, 0, eval(ORDER - 1), `
   	 realtype TMP(rsum, i) = TMP(rxp, i);
	 realtype TMP(isum, i) = TMP(ixp, i);')

	LUNROLL(i, 0, eval(ORDER - 1), `
      	rexpansions[i] = TMP(rsum, i);
	iexpansions[i] = TMP(isum, i);
	')
}

extern "C" void upward_e2e(
             const realtype x0s[],
             const realtype y0s[],
             const realtype masses[],
             const realtype *  vrxps[],
             const realtype *  vixps[],
             realtype rdstxp[],
             realtype idstxp[])
            {

	    for(int i = 0; i < ORDER; ++i)
	    	    rdstxp[i] = 0;

	    for(int i = 0; i < ORDER; ++i)
	    	    idstxp[i] = 0;
	    
	    for(int programIndex = 0; programIndex < 4; ++programIndex)
	    {
		const  realtype * rxps = vrxps[programIndex];
		const  realtype * ixps = vixps[programIndex];

	         const realtype x0 = x0s[programIndex];
            	 const realtype y0 = y0s[programIndex];
                 const realtype mass = masses[programIndex];

                 const realtype r2z0 = x0 * x0 + y0 * y0;
		 //const realtype inv_r2z0 = 1 / r2z0;
                 const realtype rinvz_1 =  x0 / r2z0;
                 const realtype iinvz_1 = -y0 / r2z0;
		 dnl
		 dnl
		 LUNROLL(j, 1, eval(ORDER),`
                 ifelse(j, 1, , `
                 const realtype TMP(rinvz, j) = TMP(rinvz, eval(j - 1)) * rinvz_1 - TMP(iinvz, eval(j - 1)) * iinvz_1;
                 const realtype TMP(iinvz, j) = TMP(rinvz, eval(j - 1)) * iinvz_1 + TMP(iinvz, eval(j - 1)) * rinvz_1;')

                 const realtype TMP(rcoeff, j) = rxps[eval(j - 1)] * TMP(rinvz, j) - ixps[eval(j - 1)] * TMP(iinvz, j);
                 const realtype TMP(icoeff, j) = rxps[eval(j - 1)] * TMP(iinvz, j) + ixps[eval(j - 1)] * TMP(rinvz, j);
                 ')

                  LUNROLL(l, 1, eval(ORDER),`
		  {
			const realtype TMP(prefac, l) = ifelse(l, 1, `- mass',`mass * esyscmd(echo -1/eval(l) | bc --mathlib )');

			pushdef(`BINFAC', `BINOMIAL(eval(l - 1), eval(k - 1)).f')
			const realtype TMP(rtmp, l) = TMP(prefac, l) LUNROLL(k, 1, l,`
			+ TMP(rcoeff, k) ifelse(BINFAC,1.f,,`* BINFAC')');

			const realtype TMP(itmp, l) = LUNROLL(k, 1, l,`
			ifelse(k,1,,+)  TMP(icoeff, k) ifelse(BINFAC,1.f,,`* BINFAC')');
			popdef(`BINFAC')dnl

			const realtype TMP(invz2, l) = TMP(rinvz, l) * TMP(rinvz, l) + TMP(iinvz, l) * TMP(iinvz, l);
			const realtype TMP(invinvz2, l) = TMP(invz2, l) ? 1 / TMP(invz2, l) : 0;
			const realtype TMP(rz, l) = TMP(rinvz, l) * TMP(invinvz2, l);
			const realtype TMP(iz, l) = - TMP(iinvz, l) * TMP(invinvz2, l);

			const realtype rpartial = TMP(rtmp, l) * TMP(rz, l) - TMP(itmp, l) * TMP(iz, l);
			const realtype ipartial = TMP(rtmp, l) * TMP(iz, l) + TMP(itmp, l) * TMP(rz, l);

			rdstxp[eval(l - 1)] += rpartial;
			idstxp[eval(l - 1)] += ipartial;
		}')
		}
	}

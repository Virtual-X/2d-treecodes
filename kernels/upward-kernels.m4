include(unroll.m4)
divert(-1)
define(GANGSIZE, 8)
divert(0) dnl dnl dnl

export void upward_p2e(
       const uniform realtype xsources[],
       const uniform realtype ysources[],
       const uniform realtype vsources[],
       const uniform int nsources,
       const uniform realtype x0,
       const uniform realtype y0,
       const uniform realtype h,
       uniform realtype * uniform mass,
       uniform realtype * uniform weight,
       uniform realtype * uniform xsum,
       uniform realtype * uniform ysum,
       uniform realtype * uniform radius,
       uniform realtype rexpansions[],
       uniform realtype iexpansions[])
{
	realtype msum = 0, wsum = 0, wxsum = 0, wysum = 0;
	
	foreach(i = 0 ... nsources)
	{
	    const realtype x = xsources[i];
	    const realtype y = ysources[i];
	    const realtype m = vsources[i];
	    const realtype w = abs(m);

	    msum += m;
	    wsum += w;
	    wxsum += x * w;
	    wysum += y * w;
	}

	*mass = reduce_add(msum);
	*weight = reduce_add(wsum);
	*xsum = reduce_add(wxsum);
	*ysum = reduce_add(wysum);


	   if (*weight == 0)
    	   {
      	      *weight = 1e-13;
      	      *xsum = (x0 + 0.5 * h) * *weight;
      	      *ysum = (y0 + 0.5 * h) * *weight;
    	   }

    	const realtype xcom = *xsum / *weight;
    	const realtype ycom = *ysum / *weight;

	realtype r2 = 0;
	foreach(i = 0 ... nsources)
	{
	    const realtype xr = xsources[i] - xcom;
	    const realtype yr = ysources[i] - ycom;

	    r2 = max(r2, xr * xr + yr * yr);
	}

	*radius = sqrt(reduce_max(r2));
	
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
 	foreach(i = 0 ... nsources)
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
   	uniform realtype TMP(rsum, i) = reduce_add(TMP(rxp, i));
	uniform realtype TMP(isum, i) = reduce_add(TMP(ixp, i));')
	   
	LUNROLL(i, 0, eval(ORDER - 1), `
      	rexpansions[i] = TMP(rsum, i);
	iexpansions[i] = TMP(isum, i);
	')
}


export void upward_e2e(
            uniform const realtype x0s[],
            uniform const realtype y0s[],
            uniform const realtype masses[],
            uniform const realtype * uniform vrxps[],
            uniform const realtype * uniform vixps[],
            uniform realtype rdstxp[],
            uniform realtype idstxp[])
            {
		const uniform realtype * rxps = vrxps[programIndex];
		const uniform realtype * ixps = vixps[programIndex];
		
	         const realtype x0 = x0s[programIndex];
            	 const realtype y0 = y0s[programIndex];
                 const realtype mass = masses[programIndex];

                 const realtype r2z0 = x0 * x0 + y0 * y0;
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

			rdstxp[eval(l - 1)] = reduce_add(rpartial);
			idstxp[eval(l - 1)] = reduce_add(ipartial);
		}')
	}

include(unroll.m4)
divert(-1)
define(GANGSIZE, 8)
divert(0) dnl dnl dnl

define(P2E_KERNEL, upward_p2e_order$1)


export void P2E_KERNEL(ORDER)(
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

    	const uniform realtype xcom = *xsum / *weight;
    	const uniform realtype ycom = *ysum / *weight;

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

		const realtype TMP(term, n) = src * (realtype)(esyscmd(echo -1/eval(n + 1) | bc --mathlib ));
		
		TMP(rxp, n) -= TMP(rprod, n) * TMP(term, n);
		TMP(ixp, n) -= TMP(iprod, n) * TMP(term, n);	
		')
	}dnl

	LUNROLL(i, 0, eval(ORDER - 1), `
	rexpansions[i] = reduce_add(TMP(rxp, i));
	iexpansions[i] = reduce_add(TMP(ixp, i));
	')
}

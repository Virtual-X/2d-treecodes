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

	if (programIndex == 0)
	{
	   if (*weight == 0)
    	   {
      	      *weight = 1e-13;
      	      *xsum = (x0 + 0.5 * h) * *weight;
      	      *ysum = (y0 + 0.5 * h) * *weight;
    	   }
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
	}dnl

	LUNROLL(i, 0, eval(ORDER - 1), `
	{
	   uniform realtype rsum = reduce_add(TMP(rxp, i));
	   uniform realtype isum = reduce_add(TMP(ixp, i));
	   
	   if (programIndex == 0)
	   {
	      rexpansions[i] = rsum;
	      iexpansions[i] = isum;
	   }
	}')
}

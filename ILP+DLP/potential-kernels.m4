include(unroll.m4)

#define EPS (10 * __DBL_EPSILON__)
define(NACC, 4)
define(GSIZE, 4)

export uniform realtype potential_p2p(
   uniform const realtype * uniform _xsrc,
   uniform const realtype * uniform _ysrc,
   uniform const realtype * uniform _vsrc,
   uniform const int nsources,
   uniform const realtype xt,
   uniform const realtype yt)
   {
   const realtype eps = EPS;
   
   realtype LUNROLL(`i', 0, eval(NACC - 1), `
   ifelse(i,0,,`,') TMP(s,i) = 0') ;
   uniform const int nnice = eval(4 * NACC) * (nsources / eval(4 * NACC));

      for(uniform int i = 0; i < nnice; i += eval(4 * NACC))
      {
	const uniform realtype * uniform const xsrc = _xsrc + i;
      	const uniform realtype * uniform const ysrc = _ysrc + i;
      	const uniform realtype * uniform const vsrc = _vsrc + i;

	LUNROLL(j, 0, eval(NACC - 1), `
	const realtype TMP(xr, j) = xt - xsrc[programIndex + eval(j * GSIZE)];
	const realtype TMP(yr, j) = yt - ysrc[programIndex + eval(j * GSIZE)];
	TMP(s, j) += log((float)(TMP(xr, j) * TMP(xr, j) + TMP(yr, j) * TMP(yr, j) + EPS)) * vsrc[programIndex + eval(j * GSIZE)];
	')
    	}

	REDUCE(`+=', LUNROLL(`i', 0, eval(NACC - 1), `ifelse(i,0,,`,') TMP(s,i)')) 
	
	foreach( i = nnice ... nsources)
	//	   for(int i = 0; i <nnice; ++i)
    	{
	    const realtype xr = xt - _xsrc[i];
      	    const realtype yr = yt - _ysrc[i];

      	    s_0 += log(xr * xr + yr * yr + eps) * _vsrc[i];
    	}
    
	return reduce_add(s_0) * 0.5;
   }

   
export uniform realtype potential_e2p(
      uniform const realtype rzs[],
      uniform const realtype izs[],
      uniform const realtype masses[],
      uniform const realtype * const uniform rxps[],
      uniform const realtype * const uniform ixps[],
      uniform const int ndst)
  {
	realtype result = 0;
	
	foreach(i = 0 ... ndst)
	//for(int i = 0; i < ndst; ++i)
	{
	   const realtype rz = rzs[i];
	   const realtype iz = izs[i];
	   const realtype r2 = rz * rz + iz * iz;
	   const realtype rinvz_1 = rz / r2;
	   const realtype iinvz_1 = -iz / r2;

	   LUNROLL(j, 2, ORDER, `
     	   const realtype TMP(rinvz, j) = TMP(rinvz, eval(j - 1)) * rinvz_1 - TMP(iinvz, eval(j - 1)) * iinvz_1;
     	   const realtype TMP(iinvz, j) = TMP(rinvz, eval(j - 1)) * iinvz_1 + TMP(iinvz, eval(j - 1)) * rinvz_1;')

	   LUNROLL(j, 1, ORDER, `
	   realtype TMP(rsum, eval(j - 1)) = rxps[i][eval(j - 1)] * TMP(rinvz, j) - ixps[i][eval(j - 1)] * TMP(iinvz, j);')
 	   REDUCE(`+=', LUNROLL(i, 0, eval(ORDER - 1),`ifelse(i,0,,`,')TMP(rsum,i)'))

	   result += masses[i] * log(r2) / 2 + TMP(rsum, 0);
	 }

	 return reduce_add(result);
  }

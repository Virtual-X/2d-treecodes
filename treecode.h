#pragma once

typedef double realtype;

#ifdef __cplusplus
extern "C"
{
#endif

    realtype treecode_n2(const realtype * __restrict__ const xsources,
			 const realtype * __restrict__ const ysources,
			 const realtype * __restrict__ const sources,
			 const int nsources, 
			 const realtype _xt,
			 const realtype _yt);

    void treecode_p2e(const realtype * __restrict__ const xsources,
		      const realtype * __restrict__ const ysources,
		      const realtype * __restrict__ const sources,
		      const int nsources,
		      const realtype x0,
		      const realtype y0,
		      const realtype h,
		      realtype * const mass,
		      realtype * const weight,
		      realtype * const xsum,
		      realtype * const ysum,
		      realtype * const radius,
		      realtype * __restrict__ const rexpansions,
		      realtype * __restrict__ const iexpansions);

    typedef realtype V4 __attribute__ ((vector_size (sizeof(realtype) * 4)));

    void treecode_e2e(const V4 srcmass, const V4 rx, const V4 ry,
		      const V4 * __restrict__ const rsrcxp,
		      const V4 * __restrict__ const isrcxp,
		      realtype * __restrict__ const rdstxp,
		      realtype * __restrict__ const idstxp);

    realtype treecode_e2p(const realtype mass,
			  const realtype rx,
			  const realtype ry,
			  const realtype * __restrict__ const rxp,
			  const realtype * __restrict__ const ixp);
    
    void treecode_potential(const realtype theta,
			    const realtype * const xsources,
			    const realtype * const ysources,
			    const realtype * const sourcevalues,
			    const int nsources, 
			    const realtype * const xtargets,
			    const realtype * const ytargets,
			    const int ntargets,
			    realtype * const targetvalues);
#ifdef __cplusplus
}
#endif

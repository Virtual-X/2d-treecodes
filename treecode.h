#pragma once

typedef double realtype;

#ifdef __cplusplus
extern "C"
{
#endif
    typedef double v4 __attribute__ ((aligned(sizeof(realtype))))  __attribute__ ((vector_size (sizeof(realtype) * 4)));

realtype treecode_n2(const realtype * __restrict__ const xsources,
		     const realtype * __restrict__ const ysources,
		     const realtype * __restrict__ const sources,
		     const int nsources, 
		     const realtype _xt,
		     const realtype _yt);
    
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

/*
*  upward-kernels.mc
*  Part of MRAG/2d-treecode-potential
*
*  Created and authored by Diego Rossinelli on 2015-11-25.
*  Copyright 2015. All rights reserved.
*
*  Users are NOT authorized
*  to employ the present software for their own publications
*  before getting a written permission from the author of this file.
*/

include(unroll.m4)

#include <pmmintrin.h>
#include <math.h>

#if defined(__INTEL_COMPILER)
inline __m128d operator+(__m128d a, __m128d b){ return _mm_add_pd(a, b); }
inline __m128d operator/(__m128d a, __m128d b){ return _mm_div_pd(a, b); }
inline __m128d operator*(__m128d a, __m128d b){ return _mm_mul_pd(a, b); }
inline __m128d operator-(__m128d a, __m128d b){ return _mm_sub_pd(a, b); }
inline __m128d operator += (__m128d& a, __m128d b){ return a = _mm_add_pd(a, b); }
inline __m128d operator -= (__m128d& a, __m128d b){ return a = _mm_sub_pd(a, b); }
#endif

#define EPS (10 * __DBL_EPSILON__)
#define MAX(a,b) (((a)>(b))?(a):(b))

define(P2E_KERNEL, reference_upward_p2e_order$1)
define(E2E_KERNEL, reference_upward_e2e_order$1)

#ifdef __cplusplus
extern "C"
#endif
void P2E_KERNEL(ORDER)(const realtype * __restrict__ const xsources,
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
  realtype * __restrict__ const iexpansions)
  {
    realtype m = 0, w = 0, wx = 0, wy = 0;

    for(int i = 0; i < nsources; ++i)
    {
      const realtype sv = sources[i];
      const realtype av = fabs(sources[i]);

      m += sv;
      w += av;
      wx += xsources[i] * av;
      wy += ysources[i] * av;
    }

    *mass = m;
    *weight = w;
    *xsum = wx;
    *ysum = wy;

    if (w == 0)
    {
      *weight = w = 1e-13;
      *xsum = wx = (x0 + 0.5 * h) * w;
      *ysum = wy = (y0 + 0.5 * h) * w;
    }

    const realtype xcom = wx / w;
    const realtype ycom = wy / w;

    const __m128d xcom2 = _mm_set1_pd(xcom);
    const __m128d ycom2 = _mm_set1_pd(ycom);

    const int nnice8 = 8 * (nsources / 8);

    __m128d LUNROLL(c, 0, 3, `ifelse(c,0,,`,')
    TMP(r22,c) = _mm_setzero_pd()');

    for(int i = 0; i < nnice8; i += 8)
    {
      LUNROLL(c, 0, 3, `
        const __m128d TMP(xr, c) = _mm_loadu_pd(xsources + i + eval(c * 2)) - xcom2;')
        LUNROLL(c, 0, 3, `
          const __m128d TMP(yr, c) = _mm_loadu_pd(ysources + i + eval(c * 2)) - ycom2;')
          LUNROLL(c, 0, 3, `
            TMP(r22, c) = _mm_max_pd(TMP(r22, c),  TMP(xr, c) * TMP(xr, c) + TMP(yr, c) * TMP(yr, c));')
          }

          LUNROLL(c, 1, 3, `
            TMP(r22,0) = _mm_max_pd(TMP(r22, 0), TMP(r22, c));');
            realtype r22a[2];
            _mm_storeu_pd(r22a, TMP(r22, 0));

            realtype r2 = MAX(r22a[0], r22a[1]);

            for(int i = nnice8; i < nsources; ++i)
            {
              const realtype xr = xsources[i] - xcom;
              const realtype yr = ysources[i] - ycom;

              r2 = MAX(r2, xr * xr + yr * yr);
            }

            *radius = sqrt(r2);

            __m128d LUNROLL(n, 0, eval(ORDER - 1),`ifelse(n,0,,`,')
            TMP(rxp, n) = _mm_setzero_pd()')
            LUNROLL(n, 0, eval(ORDER - 1),`,
            TMP(ixp, n) = _mm_setzero_pd()');

            const int nnice = 2 * (nsources / 2);
            for(int i = 0; i < nnice; i += 2)
            {
              const __m128d rrp = _mm_loadu_pd(xsources + i) - xcom2;
              const __m128d irp = _mm_loadu_pd(ysources + i) - ycom2;
              const __m128d srcs = _mm_loadu_pd(sources + i);

              __m128d rprod = rrp, iprod = irp;

              TMP(rxp, 0) -= rprod * srcs;
              TMP(ixp, 0) -= iprod * srcs;

              LUNROLL(n, 1, eval(ORDER - 1),`
              const __m128d TMP(rnewprod, n) = rprod * rrp - iprod * irp;
              const __m128d TMP(inewprod, n) = rprod * irp + iprod * rrp;

              rprod = TMP(rnewprod, n);
              iprod = TMP(inewprod, n);

              const __m128d TMP(term, n) = srcs * _mm_set1_pd(esyscmd(echo 1. / eval(n+1) | bc -l));

              TMP(rxp, n) -= rprod * TMP(term, n);
              TMP(ixp, n) -= iprod * TMP(term, n);')
            }

            for(int i = nnice; i < nsources; ++i)
            {
              const __m128d rrp = _mm_set_pd(RLUNROLL(c, 1, 0, `(i + c < nsources) ? xsources[i + c] : 0 ifelse(c,0,,`,')')) - xcom2;
              const __m128d irp = _mm_set_pd(RLUNROLL(c, 1, 0, `(i + c < nsources) ? ysources[i + c] : 0 ifelse(c,0,,`,')')) - ycom2;
              const __m128d srcs = _mm_set_pd(RLUNROLL(c, 1, 0, `(i + c < nsources) ? sources[i + c] : 0 ifelse(c,0,,`,')'));

              __m128d rprod = rrp, iprod = irp;

              TMP(rxp, 0) -= rprod * srcs;
              TMP(ixp, 0) -= iprod * srcs;

              LUNROLL(n, 1, eval(ORDER - 1),`
              const __m128d TMP(rnewprod, n) = rprod * rrp - iprod * irp;
              const __m128d TMP(inewprod, n) = rprod * irp + iprod * rrp;

              rprod = TMP(rnewprod, n);
              iprod = TMP(inewprod, n);

              const __m128d TMP(term, n) = srcs / _mm_set1_pd(eval(n+1).f);

              TMP(rxp, n) -= rprod * TMP(term, n);
              TMP(ixp, n) -= iprod * TMP(term, n);')
            }

            LUNROLL(n, 0, eval(ORDER - 1),`
            _mm_store_sd(rexpansions + n, _mm_hadd_pd(TMP(rxp, n), TMP(rxp, n)));')
            LUNROLL(n, 0, eval(ORDER - 1),`
            _mm_store_sd(iexpansions + n, _mm_hadd_pd(TMP(ixp, n), TMP(ixp, n)));')
          }

          typedef realtype V4 __attribute__ ((vector_size (sizeof(realtype) * 4)));

          #ifdef __cplusplus
          extern "C"
          #endif
          void E2E_KERNEL(ORDER)(
            const realtype * const x0s,
            const realtype * const y0s,
            const realtype * const masses,
            const realtype * __restrict__ const * vrexpansions,
            const realtype * __restrict__ const * viexpansions,
            realtype * __restrict__ const rdstxp,
            realtype * __restrict__ const idstxp)
            {
              LUNROLL(pass, 0, 1,`
                {
                  const __m128d x0 = _mm_loadu_pd(x0s + eval(2 * pass));
                  const __m128d y0 = _mm_loadu_pd(y0s + eval(2 * pass));
                  const __m128d mass = _mm_loadu_pd(masses + eval(2 * pass));

                  LUNROLL(j, 0, 1, `
                    const realtype * __restrict__ const TMP(rxp, j) = vrexpansions[eval(2 * pass) + j];')
                    LUNROLL(j, 0, 1, `
                      const realtype * __restrict__ const TMP(ixp, j) = viexpansions[eval(2 * pass) + j];')

                      const __m128d r2z0 = x0 * x0 + y0 * y0;
                      const __m128d rinvz_1 = x0 / r2z0;
                      const __m128d iinvz_1 = _mm_setzero_pd() - y0 / r2z0;
                      dnl
                      LUNROLL(j, 1, eval(ORDER),`
                      ifelse(j, 1, , `
                        const __m128d TMP(rinvz, j) = TMP(rinvz, eval(j - 1)) * rinvz_1 - TMP(iinvz, eval(j - 1)) * iinvz_1;
                        const __m128d TMP(iinvz, j) = TMP(rinvz, eval(j - 1)) * iinvz_1 + TMP(iinvz, eval(j - 1)) * rinvz_1;')

                        const __m128d TMP(curr_rxp, j) = _mm_set_pd(RLUNROLL(c, 1, 0,
                          `TMP(rxp, c)[eval(j - 1)]ifelse(c,0,,`, ')'));
                          const __m128d TMP(curr_ixp, j) = _mm_set_pd(RLUNROLL(c, 1, 0,
                            `TMP(ixp, c)[eval(j - 1)]ifelse(c,0,,`, ')'));

                            const __m128d TMP(rcoeff, j) = TMP(curr_rxp, j) * TMP(rinvz, j) - TMP(curr_ixp, j) * TMP(iinvz, j);
                            const __m128d TMP(icoeff, j) = TMP(curr_rxp, j) * TMP(iinvz, j) + TMP(curr_ixp, j) * TMP(rinvz, j);
                            ')

                            LUNROLL(l, 1, eval(ORDER),`
                            {
                              const __m128d TMP(prefac, l) = ifelse(l,1,
                                `_mm_setzero_pd() - mass',
                                `mass * _mm_set1_pd(esyscmd(echo -1/eval(l) | bc --mathlib ))');

                                pushdef(`BINFAC', `BINOMIAL(eval(l - 1), eval(k - 1)).f')

                                const __m128d TMP(rtmp, l) = TMP(prefac, l) LUNROLL(k, 1, l,` +
                                  ifelse(BINFAC,1.f,,`_mm_set1_pd(BINFAC) * ') TMP(rcoeff, k)');

                                  const __m128d TMP(itmp, l) = LUNROLL(k, 1, l,` ifelse(k,1,,+)
                                  ifelse(BINFAC,1.f,,`_mm_set1_pd(BINFAC) * ') TMP(icoeff, k)');

                                  popdef(`BINFAC')

                                  const __m128d TMP(invz2, l) = TMP(rinvz, l) * TMP(rinvz, l) + TMP(iinvz, l) * TMP(iinvz, l);
                                  const __m128d TMP(invinvz2, l) = _mm_and_pd(_mm_set1_pd(1) / TMP(invz2, l),
                                  _mm_cmpnle_pd(TMP(invz2, l), _mm_setzero_pd()));

                                  const __m128d TMP(rz, l) = TMP(rinvz, l) * TMP(invinvz2, l);
                                  const __m128d TMP(iz, l) = _mm_setzero_pd() - TMP(iinvz, l) * TMP(invinvz2, l);

                                  __m128d rpartial = TMP(rtmp, l) * TMP(rz, l) - TMP(itmp, l) * TMP(iz, l);
                                  __m128d ipartial = TMP(rtmp, l) * TMP(iz, l) + TMP(itmp, l) * TMP(rz, l);

                                  rpartial = _mm_hadd_pd(rpartial, rpartial);
                                  ipartial = _mm_hadd_pd(ipartial, ipartial);

                                  double tmp0, tmp1;
                                  _mm_store_sd(&tmp0, rpartial);
                                  _mm_store_sd(&tmp1, ipartial);

                                  ifelse(pass,0,`
                                    rdstxp[eval(l - 1)] = tmp0;
                                    idstxp[eval(l - 1)] = tmp1;',`
                                    rdstxp[eval(l - 1)] += tmp0;
                                    idstxp[eval(l - 1)] += tmp1;')
                                  }')
                                }')
                              }

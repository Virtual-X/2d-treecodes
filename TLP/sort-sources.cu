#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include "cuda-common.h"

typedef REAL realtype;

namespace SortSources
{
    __global__ void generate_keys(const realtype * const xsrc, const realtype * const ysrc, const int n,
				  const realtype xmin, const realtype ymin, const realtype ext,
				  int * const keys)
    {
	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid >= n)
	    return;

	int x = floor((xsrc[gid] - xmin) / ext * (1 << LMAX));
	int y = floor((ysrc[gid] - ymin) / ext * (1 << LMAX));

	assert(x >= 0 && y >= 0);
	assert(x < (1 << LMAX) && y < (1 << LMAX));

	x = (x | (x << 8)) & 0x00FF00FF;
	x = (x | (x << 4)) & 0x0F0F0F0F;
	x = (x | (x << 2)) & 0x33333333;
	x = (x | (x << 1)) & 0x55555555;

	y = (y | (y << 8)) & 0x00FF00FF;
	y = (y | (y << 4)) & 0x0F0F0F0F;
	y = (y | (y << 2)) & 0x33333333;
	y = (y | (y << 1)) & 0x55555555;

	const int key = x | (y << 1);

	keys[gid] = key;
    }
}

extern "C" void sort_sources(cudaStream_t stream,
			     realtype * const device_xsrc,
			     realtype * const device_ysrc,
			     realtype * const device_vsrc,
			     const int nsrc,
			     int * const device_keys,
			     realtype * const host_xmin,
			     realtype * const host_ymin,
			     realtype * const host_extent)
{
    thrust::pair<thrust::device_ptr<realtype>, thrust::device_ptr<realtype> > xminmax =
	thrust::minmax_element(thrust::cuda::par.on(stream), thrust::device_pointer_cast(device_xsrc),
			       thrust::device_pointer_cast(device_xsrc)  + nsrc);

    thrust::pair<thrust::device_ptr<realtype>, thrust::device_ptr<realtype> > yminmax =
	thrust::minmax_element(thrust::cuda::par.on(stream), thrust::device_pointer_cast(device_ysrc),
			       thrust::device_pointer_cast(device_ysrc)  + nsrc);

    const realtype truexmin = *xminmax.first;
    const realtype trueymin = *yminmax.first;

    const realtype ext0 = *xminmax.second - truexmin;
    const realtype ext1 = *yminmax.second - trueymin;

    const realtype eps = 10000 * std::numeric_limits<realtype>::epsilon();

    *host_extent = std::max(ext0, ext1) * (1 + 2 * eps);
    *host_xmin = truexmin - eps * *host_extent;
    *host_ymin = trueymin - eps * *host_extent;

    SortSources::generate_keys<<< (nsrc + 127)/128, 128>>>(device_xsrc, device_ysrc, nsrc,
					      *host_xmin, *host_ymin, *host_extent, device_keys);

    CUDA_CHECK(cudaPeekAtLastError());

    thrust::sort_by_key(thrust::cuda::par.on(stream),
			thrust::device_pointer_cast(device_keys),
			thrust::device_pointer_cast(device_keys + nsrc),
			thrust::make_zip_iterator(thrust::make_tuple(
						      thrust::device_pointer_cast(device_xsrc),
						      thrust::device_pointer_cast(device_ysrc),
						      thrust::device_pointer_cast(device_vsrc))));  
}

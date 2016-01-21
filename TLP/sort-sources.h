extern "C" void sort_sources(cudaStream_t stream,
			     realtype * const xsrc,
			     realtype * const ysrc,
			     realtype * const vsrc,
			     const int nsrc,
			     int * const sortedkeys,
			     realtype * const xmin,
			     realtype * const ymin,
			     realtype * const extent);

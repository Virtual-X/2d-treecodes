# 2d-treecode-potential
2d fast multipole treecode implementation for multicore CPUs


This libraries implements two FMM solvers, with two function signatures:

     void treecode_potential(const double theta,
			const double * const xsources,
			const double * const ysources,
			const double * const sourcevalues,
			const int nsources,
			const double * const xtargets,
			const double * const ytargets,
			const int ntargets,
			double * const targetvalues);
			
     void treecode_force_mrag(const double theta,
     	  		 const double * const xsources,
			 const double * const ysources,
			 const double * const sourcevalues,
			 const int nsources,
			 const double * const x0s,
			 const double * const y0s,
			 const double * const hs,
			 const int nblocks,
			 double * const xresults,
			 double * const yresults);

The first function solves a scalar Poisson equation in a 2D domain.
Sources and targets are represented pointwise, it's ok if sources corresponds to target,
however the pointers cannot be aliased. Two copies of the same vectors must be supplied.
Theta is the opening criterion "c" as in the [short course on FMM][1].
This function has a complexity of O(M log(N)), where M is the number of targets, and N is the number of sources.

The second function 

[1]: https://web.njit.edu/~jiang/math614/beatson-greengard.pdf

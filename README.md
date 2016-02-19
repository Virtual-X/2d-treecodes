# 2d-treecodes
This repository is about efficient implementations of 2D Fast Multipole Methods (with open BC) for CPUs and GPUs.
Two functions are exposed to the client:

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

The second function is solving a 2-components Poisson equation in a 2D domain in O(M).
The sources are represented as pointwise particles, whereas the destination are organized into 2D blocks.
The number of destinations within a block can be choosen at compile time (default is 32x32 points).
The destination coordinates are computed by the block's origin and spacing (h represents the gridspacing between two adjacent grid points within the same block). The output is written into a contiguous array containing all the results.
Example: for a blocksize of 32x32, the result of the target point contained in block ib, located w.r.t. the block origin at ix, iy, will be written at ix + 32 * (iy + 32 * ib).
This function can be straightforwardly integrated into multiresolution flow solvers such as [MRAG][2].

The repository contains a benchmark that can be compiled.
The data necessary for the benchmark can be downloaded here as 3 different TAR+GZ packages:
* [Dataset for testing the scalar Poisson solver][3]
* [Dataset for testing the vector Poisson solver at moderate system sizes][4]
* [Dataset for testing the vector Poisson solver at larger system sizes][5]

Unpack the datasets in the top level folder.


[1]: https://web.njit.edu/~jiang/math614/beatson-greengard.pdf
[2]: http://www.sciencedirect.com/science/article/pii/S002199911500039X
[3]: https://n.ethz.ch/~diegor/testDiego.tgz
[4]: https://n.ethz.ch/~diegor/diegoVel.tgz
[5]: https://n.ethz.ch/~diegor/testSid.tgz

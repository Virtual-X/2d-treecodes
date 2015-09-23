#pragma once

typedef float realtype;

extern "C"
void treecode_potential(const realtype theta,
			const realtype * const xsources, const realtype * const ysources, const realtype * const sourcevalues,
			const int nsources, 
			const realtype * const xtargets, const realtype * const ytargets, const int ntargets,
			realtype * const targetvalues);

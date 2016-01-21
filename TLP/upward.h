/*
 *  treecode.cpp
 *  Part of MRAG/2d-treecode-potential
 *
 *  Created and authored by Diego Rossinelli on 2015-09-25.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

typedef REAL realtype;

namespace Tree
{
    struct Node
    {
	int s, e;
	
	realtype mass, xcom, ycom, r;

	union
	{
	    bool innernode;
	    int children[4];
	} state;

	
	__device__ void setup(int s, int e)
	    {
		this->s = s;
		this->e = e;
		
#pragma unroll
		for (int i = 0; i < 4; ++i)
		    state.children[i] = 0;
	    }
    };

    extern realtype *device_xdata, *device_ydata, *device_vdata, *device_expansions;
    extern Node *device_nodes;
    
#ifndef NDEBUG
    extern int nnodes;
    extern Node * host_nodes;
    extern realtype *host_xdata, *host_ydata, *host_vdata, *host_expansions;
#endif
    
    void build(const realtype * const xsrc,
	       const realtype * const ysrc,
	       const realtype * const vsrc,
	       const int nsrc,
	       const int LEAF_MAXCOUNT);

    void dispose();
};

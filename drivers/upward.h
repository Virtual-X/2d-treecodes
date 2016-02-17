/*
 *  upward.h
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

#include <cassert>
#include <cmath>

#include <parallel/algorithm>
#include <limits>
#include <utility>

#include "upward-kernels.h"

namespace Tree
{
    struct Node
    {
	int s, e;
	
	realtype mass, xcom, ycom, r;

	union
	{
	    bool innernode;
	    int childbase;
	} state;
	
	void setup(int s, int e)
	    {
		this->s = s;
		this->e = e;
		this->state.childbase = 0;
	    }
    };

    extern realtype *xdata, *ydata, *vdata, *expansions;

    extern Node *nodes;
    
    void build(const realtype * const xsrc,
	       const realtype * const ysrc,
	       const realtype * const vsrc,
	       const int nsrc,
	       const int leaf_maxcapacity);
    
    void dispose();
};

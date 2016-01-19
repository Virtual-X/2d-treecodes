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

//	bool leaf;
//	int children[4];
    };

    extern realtype *xdata, *ydata, *vdata, *host_expansions;

    extern Node * host_nodes;

    void build(const realtype * const xsrc,
	       const realtype * const ysrc,
	       const realtype * const vsrc,
	       const int nsrc,
	       const int LEAF_MAXCOUNT);

    void dispose();
};

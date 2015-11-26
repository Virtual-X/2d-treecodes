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

#include <cassert>
#include <cmath>

#include <parallel/algorithm>
#include <limits>

struct Node
{
    int x, y, l, s, e;
    bool leaf;
	
    realtype w, wx, wy, mass, r;

    Node * children[4];

    realtype expansions[2][ORDER];
	
    realtype xcom() const { return wx / w; }
    realtype ycom() const { return wy / w; }

    Node() = default;

    ~Node()
	{
	    if (!leaf)
		for(int i = 0; i < 4; ++i)
		{
		    delete children[i];
			
		    children[i] = NULL;
		}
	}
};

namespace Tree
{
    extern realtype *xdata, *ydata, *vdata;
    extern Node * root;
    
    void build(const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
	 const realtype * const xdst, const realtype * const ydst, const int ndst, realtype * const vdst);
    
    void dispose();
};

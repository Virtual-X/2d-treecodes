#include <cassert>
#include <cmath>

#include <algorithm>
#include <limits>
#include <vector>
#include <map>
#include "treecode.h"

#define LEAF_MAXCOUNT 80
#define LMAX 15



using namespace std;

namespace TreeCodeDiego
{
    const realtype eps = 10 * numeric_limits<realtype>::epsilon();

    realtype ext, xmin, ymin, thetasquared;

    vector<int> keys;
    vector<realtype> data[3];

    struct Node
    {
	int x, y, l, s, e;
	bool leaf;
	realtype w, wx, wy, mass, r;

	realtype xcom() const { return wx / w; }
	realtype ycom() const { return wy / w; }

	Node * children[4];
	
	realtype expansions[2][ORDER];

	void clr()
	    {
		for(int i = 0; i < 4; ++i)
		    children[i] = NULL;
		
		for(int c = 0; c < 2; ++c)
		    for(int i = 0; i < ORDER; ++i)
			expansions[c][i] = 0;
	    }

	void dispose()
	    {
		if (!leaf)
		    for(int i = 0; i < 4; ++i)
		    {
			children[i]->dispose();
			
			delete children[i];
			children[i] = NULL;
		    }
	    }
    };

    /*void upward(Node& node)
    {
	const int s = node.s, e = node.e;
    
	for(int i = s; i < e; ++i)
	{
	    const realtype rrp = data[0][i] - node.xcom();
	    const realtype irp = data[1][i] - node.ycom();
	    
	    realtype rprod = rrp;
	    realtype iprod = irp;
	    
	    for (int n = 0; n < ORDER; ++n)
	    {
		const realtype term = data[2][i] / (n + 1);

	
		
		node.expansions[0][n] -= rprod * term;
		node.expansions[1][n] -= iprod * term;

		const realtype rnewprod = rprod * rrp - iprod * irp;
		const realtype inewprod = rprod * irp + iprod * rrp;

		rprod = rnewprod;
		iprod = inewprod;


	    }

//	    node.mass += data[2][i];
	} 
	}*/
    
    constexpr unsigned int factorial(const int n)
    {
	return n <= 1 ? 1 : (n * factorial(n - 1));
    }

    constexpr unsigned int binomial(const int n, const int k)
    {
	return factorial(n) / (factorial(n - k) * factorial(k));
    }

    void upward(Node& dst)
    {
	for(int c = 0; c < 4; ++c)
	{
	    Node& src = *dst.children[c];
	    
	    const realtype rrp = src.xcom() - dst.xcom();
	    const realtype irp = src.ycom() - dst.ycom();
	    
	
	    assert(!::isnan(rrp));
	    assert(!::isnan(irp));
	
	    for (int j = 0; j < ORDER; ++j)
	    {
		realtype rsum = 0, isum = 0, rprod = 1, iprod = 0;
	    
		for (int k = j; k >= 0; --k)
		{
		    const realtype bterm = binomial(j, k);
		    rsum += bterm * (src.expansions[0][k] * rprod - src.expansions[1][k] * iprod);
		    isum += bterm * (src.expansions[1][k] * rprod + src.expansions[0][k] * iprod);

		    const realtype rnewprod = rprod * rrp - iprod * irp;
		    const realtype inewprod = rprod * irp + iprod * rrp;

		    rprod = rnewprod;
		    iprod = inewprod;
		}
	
		const realtype term = src.mass / (j + 1);
		rsum -= rprod * term;
		isum -= iprod * term;
	    
		dst.expansions[0][j] += rsum;
		dst.expansions[1][j] += isum;
	    }
	}
    }

    int nodeid(const int x, const int y, const int l)
    {
	return ((1 << 2 * l) - 1) / 3 + (x + (1 << l) * y);
    }

    Node * build(const int x, const int y, const int l, const int s, const int e, const int mask)
    {
	const double h = ext / (1 << l);
	const double x0 = xmin + h * x, y0 = ymin + h * y;
	
	assert(x < (1 << l) && y < (1 << l) && x >= 0 && y >= 0);
    
#ifndef NDEBUG	
	for(int i = s; i < e; ++i)
	    assert(data[0][i] >= x0 && data[0][i] < x0 + h && data[1][i] >= y0 && data[1][i] < y0 + h);
#endif

	Node & node = *new Node{x, y, l, s, e, e - s <= LEAF_MAXCOUNT || l + 1 > LMAX};
	node.clr();
    
	if (node.leaf)
	    treecode_p2e(&data[0][s], &data[1][s], &data[2][s], e - s,
			 x0, y0, h, &node.mass, &node.w, &node.wx, &node.wy, &node.r,
			 node.expansions[0], node.expansions[1]);
	else
	{
	    const vector<int>::const_iterator itbegins = keys.begin();

	    node.mass = 0;
	    node.w = 0;
	    node.wx = 0;
	    node.wy = 0;
	    
	    for(int c = 0; c < 4; ++c)
	    {
		const int shift = 2 * (LMAX - l - 1);
	    
		const int key1 = mask | (c << shift);
		const int key2 = key1 + (1 << shift) - 1;

		const int indexmin = lower_bound(itbegins + s, itbegins + e, key1) - itbegins;
		const int indexsup = upper_bound(itbegins + s, itbegins + e, key2) - itbegins;

		node.children[c] = build((x << 1) + (c & 1), (y << 1) + (c >> 1), l + 1, indexmin, indexsup, key1);

		node.mass += node.children[c]->mass;
		node.w += node.children[c]->w;
		node.wx += node.children[c]->wx;
		node.wy += node.children[c]->wy;
	    }

	     node.r = 0;
	    realtype rcandidates[4];
	    for(int c = 0; c < 4; ++c)
		rcandidates[c] =  node.children[c]->r +
		    sqrt(pow(node.xcom() - node.children[c]->xcom(), 2) +
			 pow(node.ycom() - node.children[c]->ycom(), 2));
	    
	    for(int i = 0; i < ORDER; ++i)
		assert(!::isnan((double)node.expansions[0][i]) && !::isnan(node.expansions[1][i]));

	    node.r = min(1.4143 * h,
			 max(max(rcandidates[0], rcandidates[1]),
			     max(rcandidates[2], rcandidates[3])));

	    assert(node.r < 1.5 * h);

#ifndef NDEBUG
	    {
		realtype r = 0;
		
		for(int i = s; i < e; ++i)
		r = max(r, pow(data[0][i] - node.xcom(), (realtype)2) + pow(data[1][i] - node.ycom(), (realtype)2));
	
		r = sqrt(r);
		assert (r <= node.r);
	    }
#endif	
//for(int c = 0; c < 4; ++c)
//	    {
//		upward(node);
	    V4 srcmass, rx, ry,  chldexp[2][ORDER];
	    for(int c = 0; c < 4; ++c)
	    {
		srcmass[c] = node.children[c]->mass;
		rx[c] = node.children[c]->xcom();
		ry[c] = node.children[c]->ycom();
		
		for(int i = 0; i < 2; ++i)
		    for(int j = 0; j < ORDER; ++j)
			chldexp[i][j][c] = node.children[c]->expansions[i][j];
	    }

	    rx -= node.xcom();
	    ry -= node.ycom();

	    
	    
	    treecode_e2e(srcmass, rx, ry, chldexp[0], chldexp[1],
			      node.expansions[0], node.expansions[1]);

//	    }

	   
	}

		
	{
	    	    for(int i = 0; i < ORDER; ++i)
		assert(!::isnan((double)node.expansions[0][i]) && !::isnan(node.expansions[1][i]));
	    assert(node.r < 1.5 * h);
	    assert(node.xcom() >= x0 && node.xcom() < x0 + h && node.ycom() >= y0 && node.ycom() < y0 + h || node.e - node.s == 0);
	    
	}
	
	const int entry = nodeid(x, y, l);

	return &node;
    }

    realtype evaluate(const realtype xt, const realtype yt, const Node node)
    {
	const realtype r2 = pow(xt - node.xcom(), 2) + pow(yt - node.ycom(), 2);

	if (4 * node.r * node.r < thetasquared * r2)
	{
	    realtype rz = xt - node.xcom(), iz = yt - node.ycom();
	    
	    const realtype rinvz = rz / r2;
	    const realtype iinvz = -iz / r2;

	    realtype rprod = rinvz;
	    realtype iprod = iinvz;
	    realtype rs = node.mass * log(sqrt(r2));
	    
	    for(int n = 0; n < ORDER; ++n)
	    {
		rs += rprod * node.expansions[0][n] - iprod * node.expansions[1][n];

		const realtype rnewprod = rinvz * rprod - iinvz * iprod;
		const realtype inewprod = iinvz * rprod + rinvz * iprod;
		
		rprod = rnewprod;
		iprod = inewprod;
	    }

	    return rs;
	}
	else
	{
	    

	    if (node.leaf)
	    {
		const int s = node.s;
		/*	for(int i = node.s; i < node.e; ++i)
		{
		    const realtype xr = xt - data[0][i];
		    const realtype yr = yt - data[1][i];
		    s += log(sqrt(xr * xr + yr * yr + eps)) * data[2][i];
		    }*/ 
		return treecode_n2(&data[0][s], &data[1][s], &data[2][s], node.e - s, xt, yt);
	    }
	    else
	    {
		realtype s = 0;
		
		for(int c = 0; c < 4; ++c)
		    s += evaluate(xt, yt, *node.children[c]);

		return s;
	    }
	}
    }
}

using namespace TreeCodeDiego;

void treecode_potential(const realtype theta,
			const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc, 
			const realtype * const xdst, const realtype * const ydst, const int ndst, realtype * const vdst)
{    
    keys.resize(nsrc);
    
    for(int c = 0; c < 3; ++c)
	data[c].resize(nsrc);

    thetasquared = theta * theta;
    
    xmin = *min_element(xsrc, xsrc + nsrc);
    ymin = *min_element(ysrc, ysrc + nsrc);
    
    const realtype ext0 = (*max_element(xsrc, xsrc + nsrc) - xmin);
    const realtype ext1 = (*max_element(ysrc, ysrc + nsrc) - ymin);
    
    ext = max(ext0, ext1) * (1 + 2 * eps);
    xmin -= eps * ext;
    ymin -= eps * ext;

    vector< pair<int, int> > kv(nsrc);
    
    for(int i = 0; i < nsrc; ++i)
    {
	int x = floor((xsrc[i] - xmin) / ext * (1 << LMAX));
	int y = floor((ysrc[i] - ymin) / ext * (1 << LMAX));
	
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

	kv[i].first = key;
	kv[i].second = i;
    }

    sort(kv.begin(), kv.end());
    
    for(int i = 0; i < nsrc; ++i)
    {
	keys[i] = kv[i].first;

	const int entry = kv[i].second;
	assert(entry >= 0 && entry < nsrc);
	
	data[0][i] = xsrc[entry];
	data[1][i] = ysrc[entry];
	data[2][i] = vsrc[entry];
    }

    kv.clear();

    Node * root = build(0, 0, 0, 0, nsrc, 0);

    for(int i = 0; i < ndst; ++i)
	vdst[i] = evaluate(xdst[i], ydst[i], *root);
    
    for(int c = 0; c < 3; ++c)
	data[c].clear();

    root->dispose();
}


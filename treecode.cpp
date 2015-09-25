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
		mass = w = wx = wy = 0;
		
		for(int i = 0; i < 4; ++i)
		    children[i] = NULL;
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
	{
	    treecode_p2e(&data[0][s], &data[1][s], &data[2][s], e - s,
			 x0, y0, h, &node.mass, &node.w, &node.wx, &node.wy, &node.r,
			 node.expansions[0], node.expansions[1]);
	    
	    assert(node.r < 1.5 * h);
	}
	else
	{
	    const vector<int>::const_iterator itbegins = keys.begin();
	    
	    for(int c = 0; c < 4; ++c)
	    {
		const int shift = 2 * (LMAX - l - 1);
	    
		const int key1 = mask | (c << shift);
		const int key2 = key1 + (1 << shift) - 1;

		const int indexmin = lower_bound(itbegins + s, itbegins + e, key1) - itbegins;
		const int indexsup = upper_bound(itbegins + s, itbegins + e, key2) - itbegins;

		Node * chd = build((x << 1) + (c & 1), (y << 1) + (c >> 1), l + 1, indexmin, indexsup, key1);

		node.mass += chd->mass;
		node.w += chd->w;
		node.wx += chd->wx;
		node.wy += chd->wy;

		node.children[c] = chd;
	    }

	    realtype rcandidates[4];
	    node.r = 0;
	    for(int c = 0; c < 4; ++c)
		node.r = max(node.r,
			     node.children[c]->r +
			     sqrt(pow(node.xcom() - node.children[c]->xcom(), 2) +
				  pow(node.ycom() - node.children[c]->ycom(), 2)));
	    
	    node.r = min(node.r, 1.4143 * h);
	    
	    assert(node.r < 1.5 * h);
	    
#ifndef NDEBUG
	    {
		realtype r = 0;
		
		for(int i = s; i < e; ++i)
		r = max(r, pow(data[0][i] - node.xcom(), (realtype)2) + pow(data[1][i] - node.ycom(), (realtype)2));
	
		assert (sqrt(r) <= node.r);
	    }
#endif	

	    V4 srcmass, rx, ry,  chldexp[2][ORDER];
	    for(int c = 0; c < 4; ++c)
	    {
		Node * chd = node.children[c];
		
		srcmass[c] = chd->mass;
		rx[c] = chd->xcom();
		ry[c] = chd->ycom();
		
		for(int i = 0; i < 2; ++i)
		    for(int j = 0; j < ORDER; ++j)
			chldexp[i][j][c] = chd->expansions[i][j];
	    }

	    rx -= node.xcom();
	    ry -= node.ycom();
	    
	    treecode_e2e(srcmass, rx, ry, chldexp[0], chldexp[1], node.expansions[0], node.expansions[1]);
	}

#ifndef NDEBUG	
	{
	    for(int i = 0; i < ORDER; ++i)
		assert(!::isnan((double)node.expansions[0][i]) && !::isnan(node.expansions[1][i]));
	    
	    assert(node.xcom() >= x0 && node.xcom() < x0 + h && node.ycom() >= y0 && node.ycom() < y0 + h || node.e - node.s == 0);
	}
#endif
	
	return &node;
    }

    realtype evaluate(const realtype xt, const realtype yt, const Node node)
    {
	const realtype r2 = pow(xt - node.xcom(), 2) + pow(yt - node.ycom(), 2);

	if (4 * node.r * node.r < thetasquared * r2)
	    return treecode_e2p(node.mass, xt - node.xcom(), yt - node.ycom(), node.expansions[0], node.expansions[1]);
	else
	{
	    if (node.leaf)
	    {
		const int s = node.s;
	
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



export void upward_setup(
    const uniform realtype xsources[],
    const uniform realtype ysources[],
    const uniform realtype vsources[],
    const uniform int nsources,
    uniform realtype * uniform mass,
    uniform realtype * uniform weight,
    uniform realtype * uniform xsum,
    uniform realtype * uniform ysum,
    uniform realtype * uniform radius)
{
    realtype msum = 0, wsum = 0, wxsum = 0, wysum = 0;
	
    foreach(i = 0 ... nsources)
    {
	const realtype x = xsources[i];
	const realtype y = ysources[i];
	const realtype m = vsources[i];
	const realtype w = abs(m);

	msum += m;
	wsum += w;
	wxsum += x * w;
	wysum += y * w;
    }

    *mass = reduce_add(msum);
    *weight = reduce_add(wsum);
    *xsum = reduce_add(wxsum);
    *ysum = reduce_add(wysum);

    const realtype xcom = wsum ? *xsum / *weight : 0;
    const realtype ycom = wsum ? *ysum / *weight : 0;

    realtype r2 = 0;
    foreach(i = 0 ... nsources)
    {
	const realtype xr = xsources[i] - xcom;
	const realtype yr = ysources[i] - ycom;

	r2 = max(r2, xr * xr + yr * yr);
    }

    *radius = sqrt(reduce_max(r2));
}

export uniform int lower_bound_vec(
    uniform int s,
    uniform int e,
    const uniform int val,
    uniform const int keys[])
{
    uniform int c = e - s;

    if (keys[s] >= val)
	return s;

    if (keys[e - 1] < val)
	return e;
 
    while (c)
    {
	int candidate_s = s, candidate_e = e;

	const float h = (e - s) * 1.f / programCount;
	const int i = min(e - 1, (int)(s + programIndex * h + 0.499999f));

	const bool isless = keys[i] < val;
	candidate_s = isless ? i : s;
	candidate_e = isless ? e : i;

	s = reduce_max(candidate_s);
	e = reduce_min(candidate_e);
	c = min(c / programCount, e - s);
    }

    return s + 1;
}

export uniform int upper_bound_vec
(
    uniform int s,
    uniform int e,
    const uniform int val,
    uniform const int keys[])
{
    int c = e - s;

    if (keys[s] > val)
	return s;

    if (keys[e - 1] <= val)
	return e;

    while (c)
    {
	int candidate_s = s, candidate_e = e;

	const float h = (e - s) * 1.f / programCount;
	const int i = min(e - 1, (int)(s + programIndex * h + 0.499999f));
	    
	const bool isless = keys[i] <= val;
	candidate_s = isless ? i : s;
	candidate_e = isless ? e : i;

	s = reduce_max(candidate_s);
	e = reduce_min(candidate_e);
	c = min(c / programCount, e - s);
    }

    return s + 1;
}
    

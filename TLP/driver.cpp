#include <cassert>
#include <cstdio>
#include <unistd.h>
#include <pthread.h>

typedef REAL realtype;

extern "C"
void treecode_potential_solve(const realtype theta,
			      const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
			      const realtype * const xdst, const realtype * const ydst, const int ndst, realtype * const vdst);

struct Arguments
{
    realtype theta;
    const realtype *xsrc, *ysrc, *vsrc;
    int nsrc;
    const realtype *xdst, *ydst;
    int ndst;
    realtype *vdst;
};

Arguments * shargs;


volatile int nrequests = 0, nservices = 0;
volatile bool termination_request = false;
pthread_t driver_thread;


void * driver(void *)
{
    while(!termination_request)
    {
	Arguments *consumed = NULL, *read;

	__atomic_exchange(&shargs, &consumed, &read, __ATOMIC_RELAXED);
	
	if (read != NULL)
	{
	    Arguments args = *read;   
	    
	    treecode_potential_solve(args.theta,
				     args.xsrc, args.ysrc, args.vsrc, args.nsrc,
				     args.xdst, args.ydst, args.ndst, args.vdst);
	    
	    delete read;

	    ++nservices;
	}
	else
	    sched_yield();
    }

    return NULL;
}

void place_token(Arguments * token)
{	
    bool success;

    do 
    {
	Arguments * expected = NULL; //0 is the "consumption signature"
	success = __atomic_compare_exchange_n(&shargs, &expected, token, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    } 
    while(!success);
}

extern "C"
__attribute__ ((visibility ("default")))
void treecode_potential_async(const realtype theta,
			      const realtype * const xsrc, const realtype * const ysrc, const realtype * const vsrc, const int nsrc,
			      const realtype * const xdst, const realtype * const ydst, const int ndst, realtype * const vdst)
{
    static bool initialized = false;
    
    if (!initialized)
    {
	int retval = pthread_create(&driver_thread, NULL, driver, NULL);
	assert(retval == 0);
	
	initialized = true;
    }
    
    Arguments arg = {theta, xsrc, ysrc, vsrc, nsrc, xdst, ydst, ndst, vdst};

    Arguments* p = new Arguments;
    *p = arg;
    
    place_token(p);

    ++nrequests;
}

extern "C"
__attribute__ ((visibility ("default")))
void treecode_potential_wait()
{
    while(nservices != nrequests)
	usleep(100);
}

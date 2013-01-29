/*
 * Worker 
 *
 * Author: Carlos Duelo Serrano
 */

#ifndef _THREAD_WORKER_H_
#define _THREAD_WORKER_H_

#include "lruCache.hpp"
#include "channel.hpp"
#include <lunchbox/thread.h>

class threadWorker : public lunchbox::Thread
{
	private:
		Channel	*	pipe;
		Cache	*	cache;
		visibleCube_t * visibleCubesCPU;
                visibleCube_t * visibleCubesGPU;
	public:
		threadWorker();

		~threadWorker()

		virtual bool start();

		virtual bool init();

		virtual void run();

		virtual void exit();
};

#endif

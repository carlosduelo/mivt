/*
 * Worker 
 *
 * Author: Carlos Duelo Serrano
 */

#ifndef _THREAD_WORKER_H_
#define _THREAD_WORKER_H_

#include "config.hpp"
#include "lruCache.hpp"
#include "Camera.hpp"
#include "channel.hpp"
#include <lunchbox/thread.h>

class threadWorker : public lunchbox::Thread
{
	private:
		threadID_t	id;

		Camera  *	camera;
		Channel	*	pipe;
		Cache	*	cache;
		Octree  *	octree;
		
		visibleCube_t * visibleCubesCPU;
                visibleCube_t * visibleCubesGPU;

		float	*	rays;
		int		numRays;
	public:
		threadWorker(char ** argv, int id_thread, int deviceID, Camera * p_camera, Cache * p_cache, OctreeContainer * p_octreeC);

		~threadWorker()

		virtual void run();
};

#endif

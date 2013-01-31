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
#include "Octree.hpp"
#include "rayCaster.hpp"
#include "channel.hpp"
#include <lunchbox/thread.h>

#define MAX_WORKS 100

class threadWorker : public lunchbox::Thread
{
	private:
		threadID_t	id;

		Camera  *	camera;
		Channel	*	pipe;
		Cache	*	cache;
		Octree  *	octree;
		rayCaster *	raycaster;
		
		visibleCube_t * visibleCubesCPU;
                visibleCube_t * visibleCubesGPU;

		float	*	rays;
		int		numRays;

		void resetVisibleCubes();
	public:
		threadWorker(char ** argv, int id_thread, int deviceID, Camera * p_camera, Cache * p_cache, OctreeContainer * p_octreeC, rayCaster_options_t * rCasterOptions);

		~threadWorker();

		virtual void run();
};

#endif

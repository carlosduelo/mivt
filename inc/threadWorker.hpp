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
#include <lunchbox/lock.h>

#ifdef _PROFILING_M_
	#include <sys/time.h>
#endif

#define MAX_WORKS 1

class threadWorker : public lunchbox::Thread
{
	private:
		threadID_t	id;
		int		numWorks;

		lunchbox::Lock	endFrame;
		bool		recivedEndFrame;

		Camera  *	camera;
		Channel	*	pipe;
		Cache	*	cache;
		Octree  *	octree;
		rayCaster *	raycaster;
		
		visibleCube_t * visibleCubesCPU;
                visibleCube_t * visibleCubesGPU;

		float	*	rays;
		int		numRays;
		int		maxRays;

		float	*	pixel_buffer;

		#ifdef _PROFILING_M_
		struct timeval stCP, endCP;
		int	numTiles;
		int	numIterations;
		double  completeExecution;
		double	timeTotal;
		double	timeOctree;
		double 	timeCachePush;
		double  timeCachePop;
		double	timeRayCasting;
		double  timeRefactorPB;
		#endif

		void createStructures();

		void destroyStructures();

		void resetVisibleCubes();

		void refactorPixelBuffer(int numPixels);

		void createRays(int2 tile, int numPixels);

		void createFrame(int2 tile, float * buffer);
	public:
		threadWorker(char ** argv, int id_thread, int id_local, int deviceID, Camera * p_camera, Cache * p_cache, OctreeContainer * p_octreeC, rayCaster_options_t * rCasterOptions);

		~threadWorker();

		Channel * getChannel();

		void waitFinishFrame();

		void signalFinishFrame(int secret_word);

		virtual void run();
};

#endif

/*
 * Master Thread 
 *
 * Author: Carlos Duelo Serrano
 */

#ifndef _THREAD_MASTER_H_
#define _THREAD_MASTER_H_

#define MAX_WORKERS 32

#include "threadWorker.hpp"

typedef struct
{
	// worker options
	int			numWorkers;
	int			deviceID;

	// rayCaster Options
	rayCaster_options_t  	rayCasterOptions;

	// Display Options
	camera_settings_t 	displayOptions;

	// Cache Options
	int			maxElementsCache;
	int3			cubeDim;
	int			cubeInc;
	int			levelCube;

	// Octree Options
	int			maxLevelOctree;

} initParams_masterWorker_t;

typedef struct
{
	threadWorker * 	worker;
	Channel *	pipe;	
} worker_t;

class threadMaster
{
	private:
		worker_t *		workers;
		int			numWorkers;

		Camera	*		camera;
		Cache	*		cache;
		OctreeContainer *	octree;
	public:
		threadMaster(char ** argv, initParams_masterWorker_t * initParams);

		~threadMaster();
};
#endif

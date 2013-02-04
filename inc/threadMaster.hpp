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

		// Camera options
		void	setNewDisplay(camera_settings_t * settings);

		void	increaseSampling();

		void	decreaseSampling();

		void	resetCameraPosition();

		void	Move(float3 Direction);

		void	RotateX(float Angle);

		void	RotateY(float Angle);
		
		void	RotateZ(float Angle);

		void	MoveForward(float Distance);

		void	MoveUpward(float Distance);

		void	StrafeRight(float Distance);	

		// Octree options
		void increaseLevelOctree();

		void decreaseLevelOctree();

		// Ray Casting
		void increaseStep();

		void decreaseStep();

		// Cache options

		// Frame creation

		~threadMaster();
};
#endif

/*
 * Master Thread 
 *
 * Author: Carlos Duelo Serrano
 */

#ifndef _THREAD_MASTER_H_
#define _THREAD_MASTER_H_

#define MAX_WORKERS 32
#define MAX_DEVICES 4

#include "threadWorker.hpp"

typedef struct
{
	// worker options
	int			numDevices;
	int			numWorkers[MAX_DEVICES];
	int			deviceID[MAX_DEVICES];

	// rayCaster Options
	rayCaster_options_t  	rayCasterOptions;

	// Display Options
	camera_settings_t 	displayOptions;

	// Cache Options
	int			maxElementsCache[MAX_DEVICES];
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
		int			numDevices;
		int			devicesID[MAX_DEVICES];
		int			numWorkersDevice[MAX_DEVICES];
		worker_t *		workers;
		int			numWorkers;

		Camera	*		camera;
		cache_CPU_File	*	cpuCache;
		Cache	*		cache[MAX_DEVICES];
		OctreeContainer *	octree[MAX_DEVICES];
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
		void createFrame(float * pixel_buffer);

		~threadMaster();
};
#endif

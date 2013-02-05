#include "threadMaster.hpp"
#include <exception>
#include <iostream>
#include <fstream>

threadMaster::threadMaster(char ** argv, initParams_masterWorker_t * initParams)
{
	numWorkers = 0;
	for(int i=0; i<initParams->numDevices; i++)
	{
		if (initParams->numWorkers[i] > MAX_WORKERS || initParams->numWorkers[i] < 1)
		{
			std::cerr<<"Error: it has to be 1 to 32 workers"<<std::endl;
			throw;
		}

		numWorkers += initParams->numWorkers[i];
	}

	// Camera options
	camera = new Camera(&(initParams->displayOptions));

	// Create octree and cache on the differents devices
	for(int i=0; i<initParams->numDevices; i++)
	{
		std::cerr<<"Select device "<<initParams->deviceID[i]<<": ";
		if (cudaSuccess != cudaSetDevice(initParams->deviceID[i]))
		{
			std::cerr<<"Fail"<<std::endl;
			throw;
		}
		else
			std::cerr<<"OK"<<std::endl;

		// Create octree container
		octree[i]	= new OctreeContainer(argv[0], initParams->maxLevelOctree);

		cache[i]	= new Cache(&argv[1], initParams->maxElementsCache[0], initParams->cubeDim, initParams->cubeInc, initParams->levelCube, octree[i]->getnLevels());
	}

	workers 	= new worker_t[numWorkers];
	int index	= 0;

	for(int i=0; i<initParams->numDevices; i++)
	{
		int idW 	= 1;
		for(int j=0; j<initParams->numWorkers[i]; j++)
		{
			workers[index].worker 	= new threadWorker(&argv[5], idW, initParams->deviceID[i], camera, cache[i], octree[i], &(initParams->rayCasterOptions));
			workers[index].pipe 	= workers[i].worker->getChannel();
			idW <<= 1;
			//
			workers[index].worker->start();
			
			index++;
		}
	}
}

threadMaster::~threadMaster()
{
	work_packet_t work;
	work.work_id = END; 

	for(int i=0; i<numWorkers; i++)
		workers[i].pipe->push(work);

	delete camera;
	delete cache;
	delete octree;
	for(int i=0; i<numWorkers; i++)
		delete workers[i].worker;
}

// Camera options
void	threadMaster::setNewDisplay(camera_settings_t * settings)
{
	camera->setNewDisplay(settings);
}

void	threadMaster::increaseSampling()
{
	camera->increaseSampling();

	work_packet_t work;
	work.work_id = CHANGE_ANTIALIASSING; 

	for(int i=0; i<numWorkers; i++)
		workers[i].pipe->push(work);
}

void	threadMaster::decreaseSampling()
{
	camera->decreaseSampling();

	work_packet_t work;
	work.work_id = CHANGE_ANTIALIASSING; 

	for(int i=0; i<numWorkers; i++)
		workers[i].pipe->push(work);
}

void	threadMaster::resetCameraPosition()
{
	camera->resetCameraPosition();
}

void	threadMaster::Move(float3 Direction)
{
	camera->Move(Direction);
}

void	threadMaster::RotateX(float Angle)
{
	camera->RotateX(Angle);
}

void	threadMaster::RotateY(float Angle)
{
	camera->RotateY(Angle);
}

void	threadMaster::RotateZ(float Angle)
{
	camera->RotateZ(Angle);
}

void	threadMaster::MoveForward(float Distance)
{
	camera->MoveForward(Distance);
}

void	threadMaster::MoveUpward(float Distance)
{
	camera->MoveUpward(Distance);
}
void	threadMaster::StrafeRight(float Distance)
{
	camera->StrafeRight(Distance);
}

// Octree options
void threadMaster::increaseLevelOctree()
{
	work_packet_t work;
	work.work_id = UP_LEVEL_OCTREE; 

	for(int i=0; i<numWorkers; i++)
		workers[i].pipe->push(work);
}

void threadMaster::decreaseLevelOctree()
{
	work_packet_t work;
	work.work_id = DOWN_LEVEL_OCTREE; 

	for(int i=0; i<numWorkers; i++)
		workers[i].pipe->push(work);
}

// Ray Casting
void threadMaster::increaseStep()
{
	work_packet_t work;
	work.work_id = INCREASE_STEP; 

	for(int i=0; i<numWorkers; i++)
		workers[i].pipe->push(work);
}

void threadMaster::decreaseStep()
{
	work_packet_t work;
	work.work_id = DECREASE_STEP; 

	for(int i=0; i<numWorkers; i++)
		workers[i].pipe->push(work);
}

// Cache options

// Frame creation


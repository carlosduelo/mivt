#include "threadMaster.hpp"
#include <exception>
#include <iostream>
#include <fstream>

threadMaster::threadMaster(char ** argv, initParams_masterWorker_t * initParams)
{
	if (initParams->numDevices > MAX_DEVICES)
	{
		std::cerr<<"Max number of devices are "<<MAX_DEVICES<<" not "<<initParams->numDevices<<std::endl;
		throw;
	} 


	numDevices = initParams->numDevices;
	numWorkers = 0;
	for(int i=0; i<initParams->numDevices; i++)
	{
		devicesID[i] 		= initParams->deviceID[i];
		numWorkersDevice[i]	= initParams->numWorkers[i];

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
			workers[index].pipe 	= workers[index].worker->getChannel();
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

	int index = 0;
	for(int i=0; i<numDevices; i++)
	{
		std::cerr<<"Select device "<<devicesID[i]<<": ";
		if (cudaSuccess != cudaSetDevice(devicesID[i]))
		{
			std::cerr<<"Fail"<<std::endl;
			throw;
		}
		else
			std::cerr<<"OK"<<std::endl;

		for(int j=0; j<numWorkersDevice[i]; j++)
		{
			workers[index].worker->join();
			delete workers[index].worker;
			index++;
		}

		delete cache[i];
		delete octree[i];
	}
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

	int index = 0;
	for(int i=0; i<numDevices; i++)
		for(int j=0; j<numWorkersDevice[i]; j++)
		{
			workers[index].pipe->push(work);
			index++;
		}
}

void	threadMaster::decreaseSampling()
{
	camera->decreaseSampling();

	work_packet_t work;
	work.work_id = CHANGE_ANTIALIASSING; 

	int index = 0;
	for(int i=0; i<numDevices; i++)
		for(int j=0; j<numWorkersDevice[i]; j++)
		{
			workers[index].pipe->push(work);
			index++;
		}
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

	int index = 0;
	for(int i=0; i<numDevices; i++)
		for(int j=0; j<numWorkersDevice[i]; j++)
		{
			workers[index].pipe->push(work);
			index++;
		}
}

void threadMaster::decreaseLevelOctree()
{
	work_packet_t work;
	work.work_id = DOWN_LEVEL_OCTREE; 

	int index = 0;
	for(int i=0; i<numDevices; i++)
		for(int j=0; j<numWorkersDevice[i]; j++)
		{
			workers[index].pipe->push(work);
			index++;
		}
}

// Ray Casting
void threadMaster::increaseStep()
{
	work_packet_t work;
	work.work_id = INCREASE_STEP; 

	int index = 0;
	for(int i=0; i<numDevices; i++)
		for(int j=0; j<numWorkersDevice[i]; j++)
		{
			workers[index].pipe->push(work);
			index++;
		}
}

void threadMaster::decreaseStep()
{
	work_packet_t work;
	work.work_id = DECREASE_STEP; 

	int index = 0;
	for(int i=0; i<numDevices; i++)
		for(int j=0; j<numWorkersDevice[i]; j++)
		{
			workers[index].pipe->push(work);
			index++;
		}
}

// Cache options

// Frame creation
void threadMaster::createFrame(float * pixel_buffer)
{
	int  W		= camera->getHeight();
	int  H		= camera->getWidth();
	int2 tileDim 	= camera->getTileDim();

	int  i = H/tileDim.x;
	int  j = W/tileDim.y;

	for(int ii=0; ii<i; ii++)
		for(int jj=0; jj<j; jj++)
		{
			work_packet_t work;
			work.work_id 		= NEW_TILE;
			work.tile 		= make_int2(ii,jj);
			//work.pixel_buffer 	= 2;  
		}
}

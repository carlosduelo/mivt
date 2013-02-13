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
	}

	// Setting cpu cache
	int totalElements = 0;
	for(int i=0; i<initParams->numDevices; i++)
		totalElements += initParams->maxElementsCache[i];

	cpuCache = new cache_CPU_File(&argv[1], totalElements, initParams->cubeDim, initParams->cubeInc, initParams->levelCube, octree[0]->getnLevels()); 

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

		cache[i]	= new Cache(&argv[4], cpuCache, initParams->numWorkers[i], initParams->maxElementsCache[i], initParams->cubeDim, initParams->cubeInc, initParams->levelCube, octree[i]->getnLevels());
	}

	workers 	= new worker_t[numWorkers];
	int index	= 0;

	for(int i=0; i<initParams->numDevices; i++)
	{
		int idW 	= 0;
		for(int j=0; j<initParams->numWorkers[i]; j++)
		{
			workers[index].worker 	= new threadWorker(&argv[5], idW, j, initParams->deviceID[i], camera, cache[i], octree[i], &(initParams->rayCasterOptions));
			workers[index].pipe 	= workers[index].worker->getChannel();
			idW << 1;
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
	work.pixel_buffer 	= 0;

	for(int i=0; i<numWorkers; i++)
		workers[i].pipe->pushBlock(work);

	for(int i=0; i<numWorkers; i++)
		workers[i].worker->join();

	for(int i=0; i<numWorkers; i++)
		delete workers[i].worker;

	for(int i=0; i<numDevices; i++)
	{
		delete cache[i];
		delete octree[i];
	}

	delete camera;
	delete cpuCache;
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
	work.pixel_buffer 	= 0;

	int index = 0;
	for(int i=0; i<numDevices; i++)
		for(int j=0; j<numWorkersDevice[i]; j++)
		{
			workers[index].pipe->pushBlock(work);
			index++;
		}
}

void	threadMaster::decreaseSampling()
{
	camera->decreaseSampling();

	work_packet_t work;
	work.work_id = CHANGE_ANTIALIASSING; 
	work.pixel_buffer 	= 0;

	int index = 0;
	for(int i=0; i<numDevices; i++)
		for(int j=0; j<numWorkersDevice[i]; j++)
		{
			workers[index].pipe->pushBlock(work);
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
	work.pixel_buffer 	= 0;

	int index = 0;
	for(int i=0; i<numDevices; i++)
		for(int j=0; j<numWorkersDevice[i]; j++)
		{
			workers[index].pipe->pushBlock(work);
			index++;
		}
}

void threadMaster::decreaseLevelOctree()
{
	work_packet_t work;
	work.work_id = DOWN_LEVEL_OCTREE; 
	work.pixel_buffer 	= 0;

	int index = 0;
	for(int i=0; i<numDevices; i++)
		for(int j=0; j<numWorkersDevice[i]; j++)
		{
			workers[index].pipe->pushBlock(work);
			index++;
		}
}

// Ray Casting
void threadMaster::increaseStep()
{
	work_packet_t work;
	work.work_id = INCREASE_STEP; 
	work.pixel_buffer 	= 0;

	int index = 0;
	for(int i=0; i<numDevices; i++)
		for(int j=0; j<numWorkersDevice[i]; j++)
		{
			workers[index].pipe->pushBlock(work);
			index++;
		}
}

void threadMaster::decreaseStep()
{
	work_packet_t work;
	work.work_id 		= DECREASE_STEP; 
	work.pixel_buffer 	= 0;

	int index = 0;
	for(int i=0; i<numDevices; i++)
		for(int j=0; j<numWorkersDevice[i]; j++)
		{
			workers[index].pipe->pushBlock(work);
			index++;
		}
}

// Cache options

// Frame creation
void threadMaster::createFrame(float * pixel_buffer)
{
	int  H		= camera->getHeight();
	int  W		= camera->getWidth();
	int2 tileDim 	= camera->getTileDim();

	int  i = H/tileDim.x;
	int  j = W/tileDim.y;

	work_packet_t work;
	work.work_id 		= NEW_FRAME;
	work.pixel_buffer 	= 0;
	for(int index=0; index<numWorkers; index++)
		workers[index].pipe->pushBlock(work);

	int index = 0;
	for(int ii=0; ii<i; ii++)
	{
		for(int jj=0; jj<j; jj++)
		{
			work.work_id 		= NEW_TILE;
			work.tile 		= make_int2(ii,jj);
			work.pixel_buffer 	= pixel_buffer + jj*3*tileDim.y + ii*3*W*tileDim.x;

			while(!workers[index].pipe->push(work))
			{
				index++;
				if (index == numWorkers)
					index = 0;
			}
			index++;
			if (index == numWorkers)
				index = 0;
			std::cout<<"Send Tile "<<ii<<" "<<jj<<" worker "<<index<<std::endl;
		}
	}

	work.work_id = END_FRAME;
	for(int index=0; index<numWorkers; index++)
		workers[index].pipe->pushBlock(work);

	for(int index=0; index<numWorkers; index++)
		workers[index].worker->waitFinishFrame();
}

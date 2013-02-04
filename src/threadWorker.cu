#include "threadWorker.hpp"
#include <exception>
#include <iostream>
#include <fstream>
#include <strings.h>

threadWorker::threadWorker(char ** argv, int id_thread, int deviceID, Camera * p_camera, Cache * p_cache, OctreeContainer * p_octreeC, rayCaster_options_t * rCasterOptions)
{
	// Setting id thread
	id.id 		= id_thread;
	id.deviceID 	= deviceID;
	std::cerr<<"Createing cudaStream: ";
	if (cudaSuccess != cudaStreamCreate(&id.stream))
	{
		std::cerr<<"Fail"<<std::endl;
	}
	else
		std::cerr<<"OK"<<std::endl;

	camera		= p_camera;
	cache 		= p_cache;
	pipe		= new Channel(MAX_WORKS);
	raycaster	= new rayCaster(p_octreeC->getIsosurface(), rCasterOptions);

	createStructures();

	if (strcmp(argv[0], "complete_GPU") == 0)
	{
		int2 tileDim	= camera->getTileDim();
		int maxR 	= tileDim.x * tileDim.y * camera->getMaxRayPixel(); 
		octree 		= new Octree_completeGPU(p_octreeC, maxR);
	}
	else
	{
	std::cerr<<argv[0]<<std::endl;
		std::cerr<<"Error: octree option error"<<std::endl;
		throw;
	}

}

threadWorker::~threadWorker()
{
	delete pipe;
	delete octree;
	delete raycaster;
	destroyStructures();
}

void threadWorker::destroyStructures()
{
	cudaFree(visibleCubesGPU);
	cudaFreeHost(visibleCubesCPU);
	cudaFree(rays);
}

void threadWorker::createStructures()
{
	int2 tileDim	= camera->getTileDim();
	numRays		= tileDim.x * tileDim.y * camera->getNumRayPixel();

	std::cerr<<"Allocating memory visibleCubesCPU "<<numRays*sizeof(visibleCube_t)/1024/1024 <<" MB : ";
	if (cudaSuccess != cudaHostAlloc((void**)&visibleCubesCPU, numRays*sizeof(visibleCube_t), cudaHostAllocDefault))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;
        std::cerr<<"Allocating memory visibleCubesGPU "<<numRays*sizeof(visibleCube_t)/1024/1024 <<" MB : ";
	if (cudaSuccess != cudaMalloc((void**)&visibleCubesGPU, numRays*sizeof(visibleCube_t)))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;

	resetVisibleCubes();

	// Create rays
        std::cerr<<"Allocating memory rays "<<numRays*3*sizeof(float)/1024/1024 <<" MB : ";
	if (cudaSuccess != cudaMalloc((void**)&rays, 3*numRays*sizeof(float)))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;
}

Channel * threadWorker::getChannel()
{
	return pipe;
}

void threadWorker::resetVisibleCubes()
{
	cudaMemsetAsync((void*)visibleCubesGPU, 0, numRays*sizeof(visibleCube_t), id.stream);
}

void threadWorker::run()
{
	std::cout<<"Thread: " << id.id<<" started"<<std::endl;

	bool end = false;

	while(!end)
	{
		work_packet_t work = pipe->pop();

		switch(work.work_id)
		{
			case UP_LEVEL_OCTREE:
				octree->increaseLevel();
				break;
			case DOWN_LEVEL_OCTREE:
				octree->decreaseLevel();
				break;
			case INCREASE_STEP:
				raycaster->increaseStep();
				break;
			case DECREASE_STEP:
				raycaster->decreaseStep();
				break;
			case CHANGE_ANTIALIASSING:
				destroyStructures();
				createStructures();
				break;
			case END:
				std::cout<<"End thread work"<<std::endl;
				end = true;
				break;
			case NEW_TILE:
				std::cout<<"New Tile"<<std::endl;
				break;
			default:
				std::cerr<<"Error: thread recieve a not valid command"<<std::endl;
				throw;
		}

		std::cout<<"Thread: " << id.id<<" new task done"<<std::endl;
	}
}

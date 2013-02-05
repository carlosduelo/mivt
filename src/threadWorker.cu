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
	
	std::cerr<<"Thread: " << id.id<<" started device "<<id.deviceID<<": ";
	if (cudaSuccess != cudaSetDevice(id.deviceID))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;

	std::cerr<<"Thread "<<id.id<<" on device "<<"Createing cudaStream: ";
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
	int2 tileDim	= camera->getTileDim();
	numRays		= tileDim.x * tileDim.y * camera->getNumRayPixel();
	maxRays		= tileDim.x * tileDim.y * camera->getMaxRayPixel();

	createStructures();

	if (strcmp(argv[0], "complete_GPU") == 0)
	{
		octree 		= new Octree_completeGPU(p_octreeC, maxRays);
	}
	else
	{
		std::cerr<<"Error: octree option error"<<std::endl;
		throw;
	}

}

threadWorker::~threadWorker()
{
	cudaSetDevice(id.deviceID);

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
	cudaFree(pixel_buffer);
}

void threadWorker::createStructures()
{
	std::cerr<<"Thread "<<id.id<<" on device "<<"Allocating memory visibleCubesCPU "<<maxRays*sizeof(visibleCube_t)/1024/1024 <<" MB : ";
	if (cudaSuccess != cudaHostAlloc((void**)&visibleCubesCPU, maxRays*sizeof(visibleCube_t), cudaHostAllocDefault))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;
        std::cerr<<"Thread "<<id.id<<" on device "<<"Allocating memory visibleCubesGPU "<<maxRays*sizeof(visibleCube_t)/1024/1024 <<" MB : ";
	if (cudaSuccess != cudaMalloc((void**)&visibleCubesGPU, maxRays*sizeof(visibleCube_t)))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;

	resetVisibleCubes();

	// Create rays
        std::cerr<<"Thread "<<id.id<<" on device "<<"Allocating memory rays "<<maxRays*3*sizeof(float)/1024/1024 <<" MB : ";
	if (cudaSuccess != cudaMalloc((void**)&rays, 3*maxRays*sizeof(float)))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;

	// Create pixle_buffer
        std::cerr<<"Thread "<<id.id<<" on device "<<"Allocating memory pixel_buffer "<<maxRays*3*sizeof(float)/1024/1024 <<" MB : ";
	if (cudaSuccess != cudaMalloc((void**)&pixel_buffer, 3*maxRays*sizeof(float)))
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
	cudaMemsetAsync((void*)visibleCubesGPU, 0, maxRays*sizeof(visibleCube_t), id.stream);
}

void threadWorker::run()
{
	std::cerr<<"Thread: " << id.id<<" started device "<<id.deviceID<<": ";
	if (cudaSuccess != cudaSetDevice(id.deviceID))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;

	bool end = false;

	while(!end)
	{
		work_packet_t work = pipe->pop();

		switch(work.work_id)
		{
			case UP_LEVEL_OCTREE:
			{
				std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" increase octree level"<<std::endl;
				octree->increaseLevel();
				break;
			}
			case DOWN_LEVEL_OCTREE:
			{
				std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" decrease octree level"<<std::endl;
				octree->decreaseLevel();
				break;
			}
			case INCREASE_STEP:
			{
				std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" increase step"<<std::endl;
				raycaster->increaseStep();
				break;
			}
			case DECREASE_STEP:
			{
				std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" decrease step"<<std::endl;
				raycaster->decreaseStep();
				break;
			}
			case CHANGE_ANTIALIASSING:
			{
				int2 tileDim	= camera->getTileDim();
				numRays		= tileDim.x * tileDim.y * camera->getNumRayPixel();
				std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" change supersampling"<<std::endl;
				break;
			}
			case END:
			{
				std::cout<<"Thread "<<id.id<<" on device "<<id.deviceID<<"End thread work"<<std::endl;
				end = true;
				break;
			}
			case NEW_TILE:
			{
				std::cout<<"Thread "<<id.id<<" on device "<<id.deviceID<<"New Tile"<<std::endl;
				break;
			}
			default:
			{
				std::cerr<<"Error: thread "<<id.id<<" on device "<<id.deviceID<<" recieve a not valid command"<<std::endl;
				throw;
			}
		}

		std::cout<<"Thread: " << id.id<<" on device "<<id.deviceID<<" new task done"<<std::endl;
	}
}

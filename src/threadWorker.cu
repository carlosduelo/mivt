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


	if (strcmp(argv[0], "complete_GPU") == 0)
	{
		int maxR 	= tileDim.x * tileDim.y * camera->getMaxRayPixel(); 
		octree = new Octree_completeGPU(p_octreeC, maxR);
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
	cudaFree(visibleCubesGPU);
	cudaFreeHost(visibleCubesCPU);
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
	std::cout<<"Soy el thread" << id.id<<std::endl;
}

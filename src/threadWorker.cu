#include "threadWorker.hpp"
#include <exception>
#include <iostream>
#include <fstream>
#include "strings.h"

threadWorker::threadWorker(char ** argv, int id_thread, int deviceID, Camera * p_camera, Cache * p_cache, OctreeContainer * p_octreeC)
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

	camera	= p_camera;
	cache 	= p_cache;

	if (strcmp(argv[0], "complete_GPU") == 0)
	{
		int2 tileDim 	= cache->getTileDim();
		int maxR 	= tileDim.x*tileDim.y*cache->getMaxRayPixel(); 
		octree = new Octree_completeGPU(p_octreeC, maxR);
	}
	else
	{
		std::cerr<<"Error: octree option error"<<std::endl;
		throw;
	}

}

threadWorker::~threadWorker()

virtual void threadWorker::run();

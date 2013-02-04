#include "threadMaster.hpp"
#include <exception>
#include <iostream>
#include <fstream>

threadMaster::threadMaster(char ** argv, initParams_masterWorker_t * initParams)
{
	numWorkers 	= initParams->numWorkers;

	if (numWorkers > MAX_WORKERS || numWorkers < 1)
	{
		std::cerr<<"Error: it has to be 1 to 32 workers"<<std::endl;
		throw;
	}


	// Camera options
	camera = new Camera(&(initParams->displayOptions));

	// Create octree container
	octree	= new OctreeContainer(argv[0], initParams->maxLevelOctree);

	cache	= new Cache(&argv[1], initParams->maxElementsCache, initParams->cubeDim, initParams->cubeInc, initParams->levelCube, octree->getnLevels());

	workers 	= new worker_t[numWorkers];
	int idW 	= 1;
	for(int i=0; i<numWorkers; i++)
	{
		workers[i].worker 	= new threadWorker(&argv[5], idW, initParams->deviceID, camera, cache, octree, &(initParams->rayCasterOptions));
		workers[i].pipe 	= workers[i].worker->getChannel();
		idW <<= 1;
		//
		workers[i].worker->run();
	}


}

threadMaster::~threadMaster()
{
	delete camera;
	delete cache;
	delete octree;
	for(int i=0; i<numWorkers; i++)
		delete workers[i].worker;
}

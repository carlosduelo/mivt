#include "lruCache.hpp"
#include <exception>
#include <iostream>
#include <fstream>

cache_GPU_File::cache_GPU_File(char ** argv, int p_maxElements, int3 p_cubeDim, int p_cubeInc, int p_levelCube, int p_nLevels) :
	lruCache(p_maxElements, p_cubeDim, p_cubeInc, p_levelCube, p_nLevels)
{
	// OpenFile
	fileManager = OpenFile(argv, p_levelCube, p_nLevels, p_cubeDim, make_int3(p_cubeInc,p_cubeInc,p_cubeInc));

	// Create temporate cube
	tempCube = new float[offsetCube];

	// Allocating memory
	std::cerr<<"Creating cache in GPU: "<< maxElements*offsetCube*sizeof(float)/1024/1024<<" MB"<<std::endl; 
	if (cudaSuccess != cudaMalloc((void**)&cacheData, maxElements*offsetCube*sizeof(float)))
	{
		std::cerr<<"LRUCache: Error creating gpu cache"<<std::endl;
		throw;
	}
}

cache_GPU_File::~cache_GPU_File()
{
	delete tempCube;
	delete queuePositions;
	cudaFree(cacheData);
}

float * cache_GPU_File::push_cube(index_node_t idCube, threadID_t * thread)
{
#ifdef _BUNORDER_MAP_
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
#else
	std::map<index_node_t, NodeLinkedList *>::iterator it;
#endif
	lock->set();

	float * cube = NULL;	

	// Find the cube in the CPU cache
	it = indexStored.find(idCube);
	if ( it != indexStored.end() ) // If exist
	{
		NodeLinkedList * node = it->second;
		
		unsigned pos	= node->element;
		cube 		= cacheData + pos*offsetCube;

		queuePositions->moveToLastPosition(node);
		queuePositions->addReference(node);
			
	}
	else // If not exists
	{
		index_node_t 	 removedCube = (index_node_t)0;
		NodeLinkedList * node = queuePositions->getFirstFreePosition(idCube, &removedCube);

		if (node != NULL)
		{
			indexStored.insert(std::pair<int, NodeLinkedList *>(idCube, node));
			if (removedCube!= (index_node_t)0)
				indexStored.erase(indexStored.find(removedCube));

			fileManager->readCube(idCube, tempCube);//cacheData+ pos*offsetCube);

			unsigned pos   = node->element;
			cube 	= cacheData + pos*offsetCube;

			if (cudaSuccess != cudaMemcpy((void*) cube, (void*) tempCube, offsetCube*sizeof(float), cudaMemcpyHostToDevice))
			{
				std::cerr<<"Cache GPU_File: error copying to a device"<<std::endl;
			}

			queuePositions->moveToLastPosition(node);
			queuePositions->addReference(node);
		}
		// else there is no free slot
	}

	lock->unset();

	return cube;
}

void cache_GPU_File::pop_cube(index_node_t idCube)
{

#ifdef _BUNORDER_MAP_
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
#else
	std::map<index_node_t, NodeLinkedList *>::iterator it;
#endif
	lock->set();

	// Find the cube in the CPU cache
	it = indexStored.find(idCube);
	if ( it != indexStored.end() ) // If exist remove reference
	{
		NodeLinkedList * node = it->second;
		queuePositions->removeReference(node);
	}
	else
	{
		std::cerr<<"Cache is unistable"<<std::endl;
		throw;
	}
	
	lock->unset();
}

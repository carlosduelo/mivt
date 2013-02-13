#include "lruCache.hpp"
#include <exception>
#include <iostream>
#include <fstream>

cache_GPU_CPU_File::cache_GPU_CPU_File(char ** argv, cache_CPU_File * p_cpuCache, int p_maxElements, int3 p_cubeDim, int p_cubeInc, int p_levelCube, int p_nLevels) :
	lruCache(p_maxElements, p_cubeDim, p_cubeInc, p_levelCube, p_nLevels)
{
	// OpenFile
	cpuCache =  p_cpuCache;

	// Allocating memory
	std::cerr<<"Creating cache in GPU: "<< maxElements*offsetCube*sizeof(float)/1024/1024<<" MB"<<std::endl; 
	if (cudaSuccess != cudaMalloc((void**)&cacheData, maxElements*offsetCube*sizeof(float)))
	{
		std::cerr<<"LRUCache: Error creating gpu cache"<<std::endl;
		throw;
	}
}

cache_GPU_CPU_File::~cache_GPU_CPU_File()
{
	delete queuePositions;
	cudaFree(cacheData);
}

visibleCube_t * cache_GPU_CPU_File::push_cube(visibleCube_t * cube, int octreeLevel, threadID_t * thread)
{
	index_node_t idCube = cube->id >> (3*(octreeLevel-levelCube));

#if _BUNORDER_MAP_
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
#else
	std::map<index_node_t, NodeLinkedList *>::iterator it;
#endif
		
	float * pCube = cpuCache->push_cube(cube, octreeLevel, thread);

	lock->set();
	// Find the cube in the CPU cache
	it = indexStored.find(idCube);
	if ( it != indexStored.end() ) // If exist
	{
		NodeLinkedList * node = it->second;
		
		unsigned pos	= node->element;
		cube->data 	= cacheData + pos*offsetCube;
		cube->state 	= CACHED;
		cube->cubeID 	= idCube;

		queuePositions->moveToLastPosition(node);
		queuePositions->addReference(node,thread->id);
			
	}
	else // If not exists
	{
		index_node_t 	 removedCube = (index_node_t)0;
		NodeLinkedList * node = queuePositions->getFirstFreePosition(idCube, &removedCube);

		if (node != NULL)
		{
			// search on CPU cache
			if (pCube == NULL)
			{
				cube->state 	= NOCACHED;
				cube->cubeID 	= 0;
				cube->data	= 0;
			}
			else
			{
				indexStored.insert(std::pair<int, NodeLinkedList *>(idCube, node));
				if (removedCube!= (index_node_t)0)
					indexStored.erase(indexStored.find(removedCube));

				unsigned pos   = node->element;

				cube->data 	= cacheData + pos*offsetCube;
				cube->state 	= CACHED;
				cube->cubeID 	= idCube;

				if (cudaSuccess != cudaMemcpyAsync((void*) cube->data, (void*) pCube, offsetCube*sizeof(float), cudaMemcpyHostToDevice, thread->stream))
				{
					std::cerr<<"Cache GPU_CPU_File: error copying to a device"<<std::endl;
				}

				queuePositions->moveToLastPosition(node);
				queuePositions->addReference(node,thread->id);
			}
		}
		else // there is no free slot
		{
			cube->state 	= NOCACHED;
                        cube->cubeID 	= 0;
			cube->data	= 0;
		}
	}

	lock->unset();

	return cube;
}

visibleCube_t * cache_GPU_CPU_File::pop_cube(visibleCube_t * cube, int octreeLevel, threadID_t * thread)
{
	index_node_t idCube = cube->id >> (3*(octreeLevel-levelCube));

#if _BUNORDER_MAP_
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
		queuePositions->removeReference(node,thread->id);
		cpuCache->pop_cube(cube, octreeLevel, thread);
	}
	else
	{
		std::cerr<<"Cache is unistable"<<std::endl;
		throw;
	}
	
	lock->unset();

	return cube;
}

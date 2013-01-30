#include "lruCache.hpp"
#include <Exceptions.hpp>
#include <iostream>
#include <fstream>

cache_GPU_File::cache_GPU_File(char ** argv, int p_maxElements, int3 p_cubeDim, int p_cubeInc, int p_levelCube, int p_nLevels) :
	lruCache(p_maxElements, p_cubeDim, p_cubeInc, p_levelCube, p_nLevels)
{
	// OpenFile
	fileManager = OpenFile(argv, p_levelCube, p_nLevels, p_cubeDim, make_int3(p_cubeInc,p_cubeInc,p_cubeInc));

	// Allocating memory
	std::cerr<<"Creating cache in GPU: "<< maxElements*offsetCube*sizeof(float)/1024/1024<<" MB"<<std::endl; 
	if (cudaSuccess != cudaMalloc((void**)&cacheData, maxElements*offsetCube*sizeof(float)))
	{
		std::cerr<<"LRUCache: Error creating gpu cache"<<std::endl;
		throw excepGen;
	}
	#if 0
	std::cerr<<"Creating cache in CPU: "<< maxElementsCPU*offsetCube*sizeof(float)/1024/1024<<" MB: "<<std::endl;
	if (cudaSuccess != cudaHostAlloc((void**)&cacheDataCPU, maxElementsCPU*offsetCube*sizeof(float),cudaHostAllocDefault))
	{
		std::cerr<<"LRUCache: Error creating gpu cache"<<std::endl;
		throw excepGen;
	}
	#endif
}

cache_GPU_File::~cache_GPU_File()
{
	delete queuePositions;
	cudaFree(cacheData);
}

int cache_GPU_File::getCacheLevel()
{
	return levelCube;
}

void cache_GPU_File::push_cube(visibleCube_t * cube, int octreeLevel, threadID_t * thread)
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
			indexStored.insert(std::pair<int, NodeLinkedList *>(idCube, node));
			if (removedCube!= (index_node_t)0)
				indexStored.erase(indexStored.find(removedCube));

			unsigned pos   = node->element;
			fileManager->readCube(idCube, cacheData+ pos*offsetCube);

			cube->data 	= cacheData + pos*offsetCube;
			cube->state 	= CACHED;
			cube->cubeID 	= idCube;

			queuePositions->moveToLastPosition(node);
			queuePositions->addReference(node,thread->id);
		}
		else // there is no free slot
		{
			cube->state = NOCACHED;
                        cube->cubeID = 0;
		}
	}

	lock->unset();
}

void cache_GPU_File::pop_cube(visibleCube_t * cube, int octreeLevel, threadID_t * thread)
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
	}
	
	lock->unset();
}

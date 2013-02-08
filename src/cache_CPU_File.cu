#include "lruCache.hpp"
#include <exception>
#include <iostream>
#include <fstream>

cache_CPU_File::cache_CPU_File(char ** argv, int p_maxElements, int3 p_cubeDim, int p_cubeInc, int p_levelCube, int p_nLevels)
{
	// cube size
	cubeDim 	= p_cubeDim;
	cubeInc		= make_int3(p_cubeInc,p_cubeInc,p_cubeInc);
	realcubeDim	= p_cubeDim + 2 * p_cubeInc;
	levelCube	= p_levelCube;
	nLevels		= p_nLevels;
	offsetCube	= (cubeDim.x+2*cubeInc.x)*(cubeDim.y+2*cubeInc.y)*(cubeDim.z+2*cubeInc.z);

	// Creating caches
	maxElements	= p_maxElements;
	queuePositions	= new LinkedList(maxElements);

	// OpenFile
	fileManager = OpenFile(argv, p_levelCube, p_nLevels, p_cubeDim, make_int3(p_cubeInc,p_cubeInc,p_cubeInc));

	// Allocating memory
	std::cerr<<"Creating cache in CPU: "<< maxElements*offsetCube*sizeof(float)/1024/1024<<" MB: "<<std::endl;
	if (cudaSuccess != cudaHostAlloc((void**)&cacheData, maxElements*offsetCube*sizeof(float),cudaHostAllocDefault))
	{
		std::cerr<<"LRUCache: Error creating cpu cache"<<std::endl;
		throw;
	}
}

cache_CPU_File::~cache_CPU_File()
{
	delete fileManager;
	delete queuePositions;
	cudaFreeHost(cacheData);
}

float * cache_CPU_File::push_cube(visibleCube_t * cube, int octreeLevel, threadID_t * thread)
{
	index_node_t idCube = cube->id >> (3*(octreeLevel-levelCube));

#if _BUNORDER_MAP_
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
#else
	std::map<index_node_t, NodeLinkedList *>::iterator it;
#endif
	// Find the cube in the CPU cache
	it = indexStored.find(idCube);
	if ( it != indexStored.end() ) // If exist
	{
		NodeLinkedList * node = it->second;

		queuePositions->moveToLastPosition(node);
		queuePositions->addReference(node,thread->id);

		return cacheData + it->second->element*offsetCube;
			
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

			queuePositions->moveToLastPosition(node);
			queuePositions->addReference(node,thread->id);
		
			return cacheData+ pos*offsetCube;
		}
		else // there is no free slot
		{
			return NULL; 
		}
	}
}

void cache_CPU_File::pop_cube(visibleCube_t * cube, int octreeLevel, threadID_t * thread)
{
	index_node_t idCube = cube->id >> (3*(octreeLevel-levelCube));

#if _BUNORDER_MAP_
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
#else
	std::map<index_node_t, NodeLinkedList *>::iterator it;
#endif
	// Find the cube in the CPU cache
	it = indexStored.find(idCube);
	if ( it != indexStored.end() ) // If exist remove reference
	{
		NodeLinkedList * node = it->second;
		queuePositions->removeReference(node,thread->id);
	}
	else
	{
		std::cerr<<"Cache is unistable"<<std::endl;
		throw;
	}
}

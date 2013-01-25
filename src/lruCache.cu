#include "lruCache.hpp"
#include <Exceptions.hpp>
#include <iostream>
#include <fstream>

LinkedList::LinkedList(int size)
{
	freePositions 	= size;
	memoryList 	= new NodeLinkedList[size];
	list 		= memoryList;
	last 		= &memoryList[size-1];

	for(int i=0; i<size; i++)
	{
		if (i==0)
		{
			memoryList[i].after 		= &memoryList[i+1];
			memoryList[i].before 		= 0;
			memoryList[i].element 		= i;
			memoryList[i].cubeID 		= 0;
			memoryList[i].references 	= 0;
		}
		else if (i==size-1)
		{
			memoryList[i].after 		= 0;
			memoryList[i].before 		= &memoryList[i-1];
			memoryList[i].element 		= i;
			memoryList[i].cubeID 		= 0;
			memoryList[i].references 	= 0;
		}
		else
		{
			memoryList[i].after 		= &memoryList[i+1];
			memoryList[i].before 		= &memoryList[i-1];
			memoryList[i].element 		= i;
			memoryList[i].cubeID 		= 0;
			memoryList[i].references 	= 0;
		}
	}
}

LinkedList::~LinkedList()
{
	delete[] memoryList;
}


NodeLinkedList * LinkedList::getFirstFreePosition(index_node_t newIDcube, index_node_t * removedIDcube)
{
	if (freePositions > 0)
	{
		NodeLinkedList * first = list;

		// Search first free position
		while(first->references != 0)
		{
			moveToLastPosition(first);
			first = list;
		}

		list = first->after;
		list->before = 0;
		
		first->after  = 0;
		first->before = last;
		
		last->after = first;
		
		last = first;
		*removedIDcube = last->cubeID;
		last->cubeID = newIDcube;

		return first;
	}

	return NULL;
}

NodeLinkedList * LinkedList::moveToLastPosition(NodeLinkedList * node)
{
	if (node->before == 0)
	{
		NodeLinkedList * first = list;

		list = first->after;
		list->before = 0;
		
		first->after  = 0;
		first->before = last;
		
		last->after = first;
		
		last = first;

		return first;
	}
	else if (node->after == 0)
	{
		return node;
	}
	else
	{
		node->before->after = node->after;
		node->after->before = node->before;
		
		last->after = node;
		
		node->before = last;
		node->after  = 0;
		last = node;
		
		return node;
	}
}

void	LinkedList::removeReference(NodeLinkedList * node, int ref)
{
	node->references &= ~(ref);

	if (node->references == 0)
		freePositions++;
}

void 	LinkedList::addReference(NodeLinkedList * node, int ref)
{
	if (node->references == 0)
		freePositions--;

	node->references |= ref;
}

lruCache::lruCache(char ** argv, int p_maxElementsGPU, int3 p_cubeDim, int p_cubeInc, int p_levelCube, int p_levelsOctree, int p_nLevels, int p_maxElementsCPU)
{
	// cube size
	cubeDim 	= p_cubeDim;
	cubeInc		= make_int3(p_cubeInc,p_cubeInc,p_cubeInc);
	realcubeDim	= p_cubeDim + 2 * p_cubeInc;
	levelCube	= p_levelCube;
	levelOctree	= p_levelsOctree;
	nLevels		= p_nLevels;
	offsetCube	= (cubeDim.x+2*cubeInc.x)*(cubeDim.y+2*cubeInc.y)*(cubeDim.z+2*cubeInc.z);

	// Creating caches
	maxElementsGPU 		= p_maxElementsGPU;
	maxElementsCPU 		= p_maxElementsCPU;
	queuePositionsGPU	= new LinkedList(maxElementsGPU);
	queuePositionsCPU	= new LinkedList(maxElementsCPU);

	// Allocating memory
	std::cerr<<"Creating cache in GPU: "<< maxElementsGPU*offsetCube*sizeof(float)/1024/1024<<" MB"<<std::endl; 
	if (cudaSuccess != cudaMalloc((void**)&cacheDataGPU, maxElementsGPU*offsetCube*sizeof(float)))
	{
		std::cerr<<"LRUCache: Error creating gpu cache"<<std::endl;
		throw excepGen;
	}
	std::cerr<<"Creating cache in CPU: "<< maxElementsCPU*offsetCube*sizeof(float)/1024/1024<<" MB: "<<std::endl;
	if (cudaSuccess != cudaHostAlloc((void**)&cacheDataCPU, maxElementsCPU*offsetCube*sizeof(float),cudaHostAllocDefault))
	{
		std::cerr<<"LRUCache: Error creating gpu cache"<<std::endl;
		throw excepGen;
	}

	// Open File
	fileManager = OpenFile(argv, levelCube, nLevels, cubeDim, cubeInc);

	// Creating mutex needed to synchronization
	if(pthread_mutex_init(&mutex, NULL))
    	{
        	std::cerr<<"Unable to initialize a mutex"<<std::endl;
        	throw excepGen;
    	}	
}

lruCache::~lruCache()
{
	pthread_mutex_destroy(&mutex);

	delete queuePositionsGPU;
	delete queuePositionsCPU;
	delete fileManager;
	cudaFree(cacheDataGPU);
	cudaFreeHost(cacheDataCPU);
}

void lruCache::push(visibleCube_t * visibleCubes, int num, threadID_t * thread)
{
	// For each visible cube push into the cache
	for(int i=0; i<num; i++)
	{
		pthread_mutex_lock(&mutex);

		push_cube(&visibleCubes[i], thread);

		pthread_mutex_unlock(&mutex);
	}
}

void lruCache::pop(visibleCube_t * visibleCubes, int num, threadID_t * thread)
{
	// For each visible cube pop out the cache
	for(int i=0; i<num; i++)
	{
		pthread_mutex_lock(&mutex);

		pop_cube(&visibleCubes[i], thread);

		pthread_mutex_unlock(&mutex);
	}
}

void lruCache::push_cube(visibleCube_t * cube, threadID_t * thread)
{
	return;
}

void lruCache::pop_cube(visibleCube_t * cube, threadID_t * thread)
{
	index_node_t idCube = cube->id >> (3*(levelOctree-levelCube));

	#if _BUNORDER_MAP_
		boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
	#else
		std::map<index_node_t, NodeLinkedList *>::iterator it;
	#endif

	// Find the cube in the CPU cache
	it = indexStoredCPU.find(idCube);

	if ( it != indexStoredCPU.end() ) // If exist remove reference
	{
		NodeLinkedList * node = it->second;
		queuePositionsCPU->removeReference(node,thread->id);
	}

	// Find the cube in the GPU cache
	it = indexStoredGPU.find(idCube);

	if ( it != indexStoredGPU.end() ) // If exist remove reference
	{
		NodeLinkedList * node = it->second;
		queuePositionsGPU->removeReference(node, thread->id);
	}
}

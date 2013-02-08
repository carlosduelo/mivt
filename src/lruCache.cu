#include "lruCache.hpp"
#include <exception>
#include <iostream>
#include <fstream>
#include <strings.h>

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
	if (node->references > 0)
	{
		node->references &= ~(ref);

		if (node->references == 0)
			freePositions++;
	}
}

void 	LinkedList::addReference(NodeLinkedList * node, int ref)
{
	if (node->references == 0)
		freePositions--;

	node->references |= ref;
}


lruCache::lruCache(int p_maxElements, int3 p_cubeDim, int p_cubeInc, int p_levelCube, int p_nLevels)
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

	// Creating mutex needed to synchronization
	lock = new lunchbox::Lock();
}

int lruCache::getCacheLevel()
{
	return levelCube;
}

int3 lruCache::getCubeDim()
{
	return cubeDim;
}

int3 lruCache::getCubeInc()
{
	return cubeInc;
}

Cache::Cache(char ** argv, cache_CPU_File * p_cpuCache, int p_numWorkers, int p_maxElements, int3 p_cubeDim, int p_cubeInc, int p_levelCube, int p_nLevels)
{
	#if _BUNORDER_MAP_
		insertedCubes = new boost::unordered_map<index_node_t, visibleCube_t *>[p_numWorkers];
	#else
		insertedCubes = new std::map<index_node_t, visibleCube_t*>[p_numWorkers];
	#endif

	if (strcmp(argv[0], "GPU_FILE") == 0)
	{
		cache = new cache_GPU_File(&argv[1], p_maxElements, p_cubeDim, p_cubeInc, p_levelCube, p_nLevels);
	}
	else if (strcmp(argv[0], "GPU_CPU_FILE") == 0)
	{
		cache = new cache_GPU_CPU_File(&argv[1], p_cpuCache, p_maxElements, p_cubeDim, p_cubeInc, p_levelCube, p_nLevels);
	}
	else
	{
		std::cerr<<"Error: cache options error"<<std::endl;
		throw;
	}
}

Cache::~Cache()
{
	delete 	cache;
	delete[] insertedCubes;
}

int Cache::getCacheLevel()
{
	return cache->getCacheLevel();
}

int3 Cache::getCubeDim()
{
	return cache->getCubeDim();
}

int3 Cache::getCubeInc()
{
	return cache->getCubeInc();
}

void Cache::push(visibleCube_t * visibleCubes, int num, int octreeLevel, threadID_t * thread)
{
	#if 1
	#if _BUNORDER_MAP_
		boost::unordered_map<index_node_t, visibleCube_t *>::iterator it;
	#else
		std::map<index_node_t, visibleCube_t *>::iterator it;
	#endif

	// For each visible cube push into the cache
	for(int i=0; i<num; i++)
	{
		if (visibleCubes[i].state == NOCACHED || visibleCubes[i].state == CUBE)
		{
			it = insertedCubes[thread->id_global].find(visibleCubes[i].id >> (3*(octreeLevel-cache->getCacheLevel())));
			if (it == insertedCubes[thread->id_global].end()) // If does not exist, do not push again
			{
				visibleCube_t * result = cache->push_cube(&visibleCubes[i], octreeLevel, thread);
				insertedCubes[thread->id_global].insert(std::pair<int, visibleCube_t *>(visibleCubes[i].id >> (3*(octreeLevel-cache->getCacheLevel())), result));	
			}
			else
			{
				visibleCubes[i].cubeID 	= it->second->cubeID;  
				visibleCubes[i].state 	= it->second->state;
				visibleCubes[i].data	= it->second->data;

			}
		}
	}
	#else
	#if _BUNORDER_MAP_
                boost::unordered_map<index_node_t, visibleCube_t *>  insertedCubes;
		boost::unordered_map<index_node_t, visibleCube_t *>::iterator it;
	#else
                std::map<index_node_t, visibleCube_t* >  insertedCubes;
		std::map<index_node_t, visibleCube_t *>::iterator it;
	#endif

	// For each visible cube push into the cache
	for(int i=0; i<num; i++)
	{
		if (visibleCubes[i].state == NOCACHED || visibleCubes[i].state == CUBE)
		{
			it = insertedCubes.find(visibleCubes[i].id >> (3*(octreeLevel-cache->getCacheLevel())));
			if (it == insertedCubes.end()) // If does not exist, do not push again
			{
				visibleCube_t * result = cache->push_cube(&visibleCubes[i], octreeLevel, thread);
				insertedCubes.insert(std::pair<int, visibleCube_t *>(visibleCubes[i].id >> (3*(octreeLevel-cache->getCacheLevel())), result));	
			}
			else
			{
				visibleCubes[i].cubeID 	= it->second->cubeID;  
				visibleCubes[i].state 	= it->second->state;
				visibleCubes[i].data	= it->second->data;

			}
		}
	}
	#endif
}

void Cache::pop(visibleCube_t * visibleCubes, int num, int octreeLevel, threadID_t * thread)
{
#if 1
	#if _BUNORDER_MAP_
		boost::unordered_map<index_node_t, visibleCube_t *>::iterator it = insertedCubes[thread->id_global].begin();
	#else
		std::map<index_node_t, visibleCube_t *>::iterator it = insertedCubes[thread->id_global].begin();
	#endif

	while(it != insertedCubes[thread->id_global].begin())
	{
		if (it->second->state == CACHED)
			cache->pop_cube(it->second, octreeLevel, thread);

		it++;
	}
	insertedCubes[thread->id_global].clear();
#else
	#if _BUNORDER_MAP_
                boost::unordered_map<index_node_t, visibleCube_t *>  insertedCubes;
		boost::unordered_map<index_node_t, visibleCube_t *>::iterator it;
	#else
                std::map<index_node_t, visibleCube_t* >  insertedCubes;
		std::map<index_node_t, visibleCube_t *>::iterator it;
	#endif
	// For each visible cube pop out the cache
	for(int i=0; i<num; i++)
	{
		if (visibleCubes[i].state == PAINTED)
		{
			it = insertedCubes.find(visibleCubes[i].id >> (3*(octreeLevel-cache->getCacheLevel())));
			if (it == insertedCubes.end()) // If does not exist, do not pop again
			{
				cache->pop_cube(&visibleCubes[i], octreeLevel, thread);
				insertedCubes.insert(std::pair<int, visibleCube_t *>(visibleCubes[i].id >> (3*(octreeLevel-cache->getCacheLevel())), NULL));	
			}
		}
	}
#endif
}


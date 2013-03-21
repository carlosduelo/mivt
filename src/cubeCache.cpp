/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include "cubeCache.h"

namespace eqMivt
{


bool cubeCache::init(cubeCacheCPU * p_cpuCache, int p_numWorkers, int p_maxElements)
{
#ifdef _BUNORDER_MAP_
	insertedCubes = new boost::unordered_map<index_node_t, cacheElement_t >[p_numWorkers];
#else
	insertedCubes = new std::map<index_node_t, cacheElement_t >[p_numWorkers];
#endif

	return cache.init(p_cpuCache, p_maxElements);	

}

cubeCache::~cubeCache()
{
	delete insertedCubes;
}

void cubeCache::push(visibleCube_t * visibleCubes, int num, int octreeLevel, threadID_t * thread)
{
#ifdef _BUNORDER_MAP_
	boost::unordered_map<index_node_t, cacheElement_t>::iterator it;
#else
	std::map<index_node_t, cacheElement_t>::iterator it;
#endif

	// For each visible cube push into the cache
	for(int i=0; i<num; i++)
	{
		if (visibleCubes[i].state == NOCACHED || visibleCubes[i].state == CUBE)
		{
			index_node_t idCube = visibleCubes[i].id >> (3*(octreeLevel - cache.getLevelCube()));

			it = insertedCubes[thread->id_local].find(idCube);
			if (it == insertedCubes[thread->id_local].end()) // If does not exist, do not push again
			{
				float * cubeData = cache.push_cube(idCube, thread);

				visibleCubes[i].cubeID  = idCube;
				visibleCubes[i].state   = cubeData == 0 ? NOCACHED : CACHED;
				visibleCubes[i].data    = cubeData;

				cacheElement_t newCube;
				newCube.cubeID = idCube;
				newCube.state = cubeData == 0 ? NOCACHED : CACHED;
				newCube.data = cubeData;

				insertedCubes[thread->id_local].insert(std::pair<index_node_t, cacheElement_t>(idCube, newCube));
			}
			else
			{
				visibleCubes[i].cubeID  = it->second.cubeID;
				visibleCubes[i].state   = it->second.state;
				visibleCubes[i].data    = it->second.data;

			}
		}
	}

}

void cubeCache::pop(visibleCube_t * visibleCubes, int num, int octreeLevel, threadID_t * thread)
{
#ifdef _BUNORDER_MAP_
	boost::unordered_map<index_node_t, cacheElement_t>::iterator it;
#else
	std::map<index_node_t, cacheElement_t>::iterator it;
#endif

	it = insertedCubes[thread->id_local].begin();

	while(it != insertedCubes[thread->id_local].end())
	{
		if (it->second.state == CACHED)
		{
			cache.pop_cube(it->second.cubeID);
		}
		it++;
	}

	insertedCubes[thread->id_local].clear();

}

}

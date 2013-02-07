/*
 * Cube cache
 *
 * Author: Carlos Duelo Serrano
 */

#ifndef _LRU_CACHE_H_
#define _LRU_CACHE_H_

#include "config.hpp"
#include "fileUtil.hpp"
#include "cutil_math.h"
#include <lunchbox/lock.h>

#if _BUNORDER_MAP_
	#include <boost/unordered_map.hpp>
#else
	#include <map>
#endif

class NodeLinkedList
{
	public:
		NodeLinkedList * 	after;
		NodeLinkedList * 	before;
		unsigned int	 	element;
		index_node_t 	 	cubeID;
		int			references;
};

class LinkedList
{
	private:
		NodeLinkedList * 	list;
		NodeLinkedList * 	last;
		NodeLinkedList * 	memoryList;
		int			freePositions;
	public:
		LinkedList(int size);
		~LinkedList();

		/* pop_front and push_last */
		NodeLinkedList * 	getFirstFreePosition(index_node_t newIDcube, index_node_t * removedIDcube);

		NodeLinkedList * 	moveToLastPosition(NodeLinkedList * node);	

		void 			removeReference(NodeLinkedList * node, int ref);
		void 			addReference(NodeLinkedList * node, int ref);
};

class lruCache
{
	protected:		
		lunchbox::Lock	* lock;

		int3	cubeDim;
		int3	cubeInc;
		int3	realcubeDim;
		int	offsetCube;
		int	levelCube;
		int	nLevels;

		#if _BUNORDER_MAP_
			boost::unordered_map<index_node_t, NodeLinkedList *> indexStored;
		#else
			std::map<index_node_t, NodeLinkedList *> indexStored;
		#endif

		LinkedList	*	queuePositions;

		int			maxElements;
		float		*	cacheData;


	public:
		lruCache(int p_maxElements, int3 p_cubeDim, int p_cubeInc, int p_levelCube, int p_nLevels);

		virtual ~lruCache() {}

		int3 getCubeDim();

		int3 getCubeInc();

		int getCacheLevel();

		virtual visibleCube_t * push_cube(visibleCube_t * cube, int octreeLevel, threadID_t * thread) = 0;

                virtual visibleCube_t * pop_cube(visibleCube_t * cube, int octreeLevel, threadID_t * thread) = 0;		
};

class cache_GPU_File : public lruCache
{
	private:
		// Acces to file
		float		*	tempCube;
		FileManager	*	fileManager;
	public:
		cache_GPU_File(char ** argv, int p_maxElements, int3 p_cubeDim, int p_cubeInc, int p_levelCube, int p_nLevels);

		~cache_GPU_File();

		visibleCube_t * push_cube(visibleCube_t * cube, int octreeLevel, threadID_t * thread);

                visibleCube_t * pop_cube(visibleCube_t * cube, int octreeLevel, threadID_t * thread);
};


class Cache
{
	private:
		lruCache * cache; 

	public:
		Cache(char ** argv, int p_maxElements, int3 p_cubeDim, int p_cubeInc, int p_levelCube, int p_nLevels);
		
		~Cache();

		int getCacheLevel();

		int3 getCubeDim();

		int3 getCubeInc();

		void push(visibleCube_t * visibleCubes, int num, int octreeLevel, threadID_t * thread);

		void pop(visibleCube_t * visibleCubes, int num, int octreeLevel, threadID_t * thread);
};
#endif

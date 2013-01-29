/*
 * Cube cache
 *
 * Author: Carlos Duelo Serrano
 */

#ifndef _LRU_CACHE_H_
#define _LRU_CACHE_H_

#include "config.hpp"
#include "FileManager.hpp"
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
		int	levelOctree;
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
		lruCache(int p_maxElements, int3 p_cubeDim, int p_cubeInc, int p_levelCube, int p_levelsOctree, int p_nLevels);

		virtual ~lruCache() {};

		virtual void push_cube(visibleCube_t * cube, threadID_t * thread);

                virtual void pop_cube(visibleCube_t * cube, threadID_t * thread);		
};

class cache_GPU_File : public lruCache
{
	private:
		// Acces to file
		FileManager	*	fileManager;
	public:
		cache_GPU_File(char ** argv, int p_maxElements, int3 p_cubeDim, int p_cubeInc, int p_levelCube, int p_levelsOctree, int p_nLevels);

		~cache_GPU_File();

		void push_cube(visibleCube_t * cube, threadID_t * thread);

                void pop_cube(visibleCube_t * cube, threadID_t * thread);
};


class Cache
{
	private:
		lruCache * caches; 
	public:
		Cache(char ** argv, int p_maxElements, int3 p_cubeDim, int p_cubeInc, int p_levelCube, int p_levelsOctree, int p_nLevels);
		
		~Cache();

		void push(visibleCube_t * visibleCubes, int num, threadID_t * thread);

		void pop(visibleCube_t * visibleCubes, int num, threadID_t * thread);
};
#endif

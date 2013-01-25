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
#include <pthread.h>

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
	private:
		// Synchronization
		pthread_mutex_t mutex;

		int3	cubeDim;
		int3	cubeInc;
		int3	realcubeDim;
		int	offsetCube;
		int	levelCube;
		int	levelOctree;
		int	nLevels;

		#if _BUNORDER_MAP_
			boost::unordered_map<index_node_t, NodeLinkedList *> indexStoredCPU;
			boost::unordered_map<index_node_t, NodeLinkedList *> indexStoredGPU;
		#else
			std::map<index_node_t, NodeLinkedList *> indexStoredCPU;
			std::map<index_node_t, NodeLinkedList *> indexStoredGPU;
		#endif

		LinkedList	*	queuePositionsCPU;
		LinkedList	*	queuePositionsGPU;

		int			maxElementsGPU;
		int			maxElementsCPU;
		float		*	cacheDataGPU;
		float		*	cacheDataCPU;

		// Acces to file
		FileManager	*	fileManager;

		// Methods
		void push_cube(visibleCube_t * cube, threadID_t * thread);
		void pop_cube(visibleCube_t * cube, threadID_t * thread);
	public:
		lruCache(char ** argv, int p_maxElementsGPU, int3 p_cubeDim, int p_cubeInc, int p_levelCube, int p_levelsOctree, int p_nLevels, int p_maxElementsCPU);
		
		~lruCache();

		void push(visibleCube_t * visibleCubes, int num, threadID_t * thread);

		void pop(visibleCube_t * visibleCubes, int num, threadID_t * thread);
};
#endif

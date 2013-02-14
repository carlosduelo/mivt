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

	lock = new lunchbox::Lock();

	// Allocating memory
	std::cerr<<"Creating cache in CPU: "<< maxElements*offsetCube*sizeof(float)/1024/1024<<" MB: "<<std::endl;
	if (cudaSuccess != cudaHostAlloc((void**)&cacheData, maxElements*offsetCube*sizeof(float),cudaHostAllocDefault))
	{
		std::cerr<<"LRUCache: Error creating cpu cache"<<std::endl;
		throw;
	}

	#ifdef _PROFILING_M_
                access = 0;
                missR = 0;
                missN = 0;
                hits = 0;
		pops = 0;
                timeAccess = 0.0;
                timeMiss = 0.0;
                timeHits = 0.0;
		timePop = 0.0;
	#endif
}

cache_CPU_File::~cache_CPU_File()
{
	delete fileManager;
	delete queuePositions;
	cudaFreeHost(cacheData);
	delete lock;

	#ifdef _PROFILING_M_
	std::cout<<"CPU CACHE SUMMARY:"<<std::endl;
	std::cout<<" Access: "		<<access<<" time spend "<<timeAccess<<" seconds "<<"average time "<<timeAccess/access<<" seconds"<<std::endl;
	std::cout<<" Hits: "		<<hits<<" time spend "<<timeHits<<" seconds "<<"average time "<<timeHits/hits<<" seconds"<<std::endl;
	std::cout<<" Miss Read: "	<<missR<<" time spend "<<timeMiss<<" seconds "<<"average time "<<timeMiss/missR<<" seconds"<<std::endl;
	std::cout<<" Miss Empty: "	<<missN<<" time spend "<<0<<" seconds "<<"average time "<<0/missN<<" seconds"<<std::endl;
	std::cout<<" Pops:"		<<pops<<" time spend "<<timePop<<" seconds "<<"average time "<<timePop/pops<<" seconds"<<std::endl;
	#endif
}

float * cache_CPU_File::push_cube(visibleCube_t * cube, int octreeLevel, threadID_t * thread)
{
	index_node_t idCube = cube->id >> (3*(octreeLevel-levelCube));

#ifdef _BUNORDER_MAP_
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
#else
	std::map<index_node_t, NodeLinkedList *>::iterator it;
#endif
	lock->set();
	
#ifdef _PROFILING_M_
	struct timeval st, end;
	gettimeofday(&st, NULL);
	access++;
#endif

	// Find the cube in the CPU cache
	it = indexStored.find(idCube);
	if ( it != indexStored.end() ) // If exist
	{
#ifdef _PROFILING_M_
	struct timeval stOP, endOP;
	gettimeofday(&stOP, NULL);
#endif
		NodeLinkedList * node = it->second;

		queuePositions->moveToLastPosition(node);
		queuePositions->addReference(node,thread->id);

#ifdef _PROFILING_M_
	hits++;
	gettimeofday(&end, NULL);
	timeAccess += ((end.tv_sec  - st.tv_sec) * 1000000u + end.tv_usec - st.tv_usec) / 1.e6;
	gettimeofday(&endOP, NULL);
	timeHits += ((endOP.tv_sec  - stOP.tv_sec) * 1000000u + endOP.tv_usec - stOP.tv_usec) / 1.e6;
#endif
		lock->unset();
		return cacheData + it->second->element*offsetCube;
			
	}
	else // If not exists
	{
#ifdef _PROFILING_M_
	struct timeval stOP, endOP;
	gettimeofday(&stOP, NULL);
#endif
		index_node_t 	 removedCube = (index_node_t)0;
		NodeLinkedList * node = queuePositions->getFirstFreePosition(idCube, &removedCube);

		if (node != NULL)
		{
			indexStored.insert(std::pair<int, NodeLinkedList *>(idCube, node));
			if (removedCube!= (index_node_t)0)
				indexStored.erase(indexStored.find(removedCube));

			unsigned pos   = node->element;

			queuePositions->moveToLastPosition(node);
			queuePositions->addReference(node,thread->id);

			fileManager->readCube(idCube, cacheData+ pos*offsetCube);

			lock->unset();
		
#ifdef _PROFILING_M_
	missR++;
	gettimeofday(&end, NULL);
	timeAccess += ((end.tv_sec  - st.tv_sec) * 1000000u + end.tv_usec - st.tv_usec) / 1.e6;
	gettimeofday(&endOP, NULL);
	timeMiss += ((endOP.tv_sec  - stOP.tv_sec) * 1000000u + endOP.tv_usec - stOP.tv_usec) / 1.e6;
#endif

			return cacheData+ pos*offsetCube;
		}
		else // there is no free slot
		{
#ifdef _PROFILING_M_
	missN++;
	gettimeofday(&end, NULL);
	timeAccess += ((end.tv_sec  - st.tv_sec) * 1000000u + end.tv_usec - st.tv_usec) / 1.e6;
	gettimeofday(&endOP, NULL);
	timeMiss += ((endOP.tv_sec  - stOP.tv_sec) * 1000000u + endOP.tv_usec - stOP.tv_usec) / 1.e6;
#endif
			lock->unset();
			return NULL; 
		}
	}
}

void cache_CPU_File::pop_cube(visibleCube_t * cube, int octreeLevel, threadID_t * thread)
{
	index_node_t idCube = cube->id >> (3*(octreeLevel-levelCube));

#ifdef _BUNORDER_MAP_
	boost::unordered_map<index_node_t, NodeLinkedList *>::iterator it;
#else
	std::map<index_node_t, NodeLinkedList *>::iterator it;
#endif
	lock->set();
#ifdef _PROFILING_M_
	struct timeval stOP, endOP;
	gettimeofday(&stOP, NULL);
	pops++;
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
		lock->unset();
		std::cerr<<"Cache is unistable"<<std::endl;
		throw;
	}
#ifdef _PROFILING_M_
	gettimeofday(&endOP, NULL);
	timePop += ((endOP.tv_sec  - stOP.tv_sec) * 1000000u + endOP.tv_usec - stOP.tv_usec) / 1.e6;
#endif
	lock->unset();
}

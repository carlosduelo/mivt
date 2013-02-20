#include "lruCache.hpp"
#include <exception>
#include <iostream>
#include <fstream>

typedef struct
{
	cache_CPU_File *	cpuCache;
	index_node_t 		idCube;
} callback_struct_t;

void unLockCPUCube(cudaStream_t stream, cudaError_t status, void *data)
{
	callback_struct_t * packet = (callback_struct_t*) data;

	packet->cpuCache->pop_cube(packet->idCube);

	delete packet;
}


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

cache_GPU_CPU_File::~cache_GPU_CPU_File()
{
	delete queuePositions;
	cudaFree(cacheData);

	#ifdef _PROFILING_M_
	std::cout<<"GPU CACHE SUMMARY:"<<std::endl;
	std::cout<<" Access: "		<<access<<" time spend "<<timeAccess<<" seconds "<<"average time "<<timeAccess/access<<" seconds"<<std::endl;
	std::cout<<" Hits: "		<<hits<<" time spend "<<timeHits<<" seconds "<<"average time "<<timeHits/hits<<" seconds"<<std::endl;
	std::cout<<" Miss Read: "	<<missR<<" time spend "<<timeMiss<<" seconds "<<"average time "<<timeMiss/missR<<" seconds"<<std::endl;
	std::cout<<" Miss Empty: "	<<missN<<" time spend "<<0<<" seconds "<<"average time "<<0/missN<<" seconds"<<std::endl;
	std::cout<<" Pops:"		<<pops<<" time spend "<<timePop<<" seconds "<<"average time "<<timePop/pops<<" seconds"<<std::endl;
	#endif
}

float * cache_GPU_CPU_File::push_cube(index_node_t idCube, threadID_t * thread)
{
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

	float * cube = 0;

	// Find the cube in the GPU cache
	it = indexStored.find(idCube);
	if ( it != indexStored.end() ) // If exist
	{
#ifdef _PROFILING_M_
	struct timeval stOP, endOP;
	gettimeofday(&stOP, NULL);
#endif
		NodeLinkedList * node = it->second;
		
		unsigned pos	= node->element;
		cube 	= cacheData + pos*offsetCube;

		queuePositions->moveToLastPosition(node);
		queuePositions->addReference(node);
#ifdef _PROFILING_M_
	hits++;
	gettimeofday(&endOP, NULL);
	timeHits += ((endOP.tv_sec  - stOP.tv_sec) * 1000000u + endOP.tv_usec - stOP.tv_usec) / 1.e6;
#endif
			
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
			// Search in cpu cache and check as locked
			float * pCube = cpuCache->push_cube(idCube);

			// search on CPU cache
			if (pCube != NULL)
			{
				indexStored.insert(std::pair<int, NodeLinkedList *>(idCube, node));
				if (removedCube!= (index_node_t)0)
					indexStored.erase(indexStored.find(removedCube));

				queuePositions->moveToLastPosition(node);
				queuePositions->addReference(node);

				unsigned pos   = node->element;
				cube 	= cacheData + pos*offsetCube;

				if (cudaSuccess != cudaMemcpyAsync((void*) cube, (void*) pCube, offsetCube*sizeof(float), cudaMemcpyHostToDevice, thread->stream))
				{
					std::cerr<<"Cache GPU_CPU_File: error copying to a device"<<std::endl;
				}

				// Unlock the cube on cpu cache

				callback_struct_t * callBackData = new callback_struct_t;
				callBackData->cpuCache = cpuCache;
				callBackData->idCube = idCube;

				if ( cudaSuccess != cudaStreamAddCallback(thread->stream, unLockCPUCube, (void*)callBackData, 0))
        			{
			                std::cerr<<"Error making cudaCallback"<<std::endl;
                			throw;
			        }

			}

#ifdef _PROFILING_M_
	missR++;
	gettimeofday(&endOP, NULL);
	timeMiss += ((endOP.tv_sec  - stOP.tv_sec) * 1000000u + endOP.tv_usec - stOP.tv_usec) / 1.e6;
#endif
		}
		else // there is no free slot
		{
#ifdef _PROFILING_M_
	missN++;
	gettimeofday(&endOP, NULL);
	timeMiss += ((endOP.tv_sec  - stOP.tv_sec) * 1000000u + endOP.tv_usec - stOP.tv_usec) / 1.e6;
#endif
		}
	}
#ifdef _PROFILING_M_
	gettimeofday(&end, NULL);
	timeAccess += ((end.tv_sec  - st.tv_sec) * 1000000u + end.tv_usec - st.tv_usec) / 1.e6;
#endif

	lock->unset();

	return cube;
}

void cache_GPU_CPU_File::pop_cube(index_node_t idCube)
{
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
		queuePositions->removeReference(node);
	}
	else
	{
		std::cerr<<"Cache is unistable"<<std::endl;
		throw;
	}
#ifdef _PROFILING_M_
	gettimeofday(&endOP, NULL);
	timePop += ((endOP.tv_sec  - stOP.tv_sec) * 1000000u + endOP.tv_usec - stOP.tv_usec) / 1.e6;
#endif
	
	lock->unset();
}

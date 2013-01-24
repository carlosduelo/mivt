#include "lruCache.hpp"
#include <iostream>
#include <fstream>

LinkedList::LinkedList(int size)
{
	memoryList = new NodeLinkedList[size];
	list = memoryList;
	last = &memoryList[size-1];
	for(int i=0; i<size; i++)
	{
		if (i==0)
		{
			memoryList[i].after = &memoryList[i+1];
			memoryList[i].before = 0;
			memoryList[i].element = i;
			memoryList[i].cubeID = 0;
		}
		else if (i==size-1)
		{
			memoryList[i].after = 0;
			memoryList[i].before = &memoryList[i-1];
			memoryList[i].element = i;
			memoryList[i].cubeID = 0;
		}
		else
		{
			memoryList[i].after = &memoryList[i+1];
			memoryList[i].before = &memoryList[i-1];
			memoryList[i].element = i;
			memoryList[i].cubeID = 0;
		}
	}
}

LinkedList::~LinkedList()
{
	delete[] memoryList;
}


NodeLinkedList * LinkedList::getFromFirstPosition(index_node_t newIDcube, index_node_t * removedIDcube)
{
	NodeLinkedList * first = list;

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


lruCache::lruCache(char ** argv, int p_maxElementsGPU, int3 p_cubeDim, int p_cubeInc, int p_levelCube, int p_levelsOctree, int p_nLevels, int p_maxElementsCPU)
{
	maxElementsGPU 	= p_maxElementsGPU;
	maxElementsCPU 	= p_maxElementsCPU;
	cubeDim 	= p_cubeDim;
	cubeInc		= make_int3(p_cubeInc,p_cubeInc,p_cubeInc);
	realcubeDim	= p_cubeDim + 2 * p_cubeInc;
	levelCube	= p_levelCube;
	levelOctree	= p_levelsOctree;
	nLevels		= p_nLevels;
	offsetCube	= (cubeDim.x+2*cubeInc.x)*(cubeDim.y+2*cubeInc.y)*(cubeDim.z+2*cubeInc.z);
	queuePositionsGPU	= new LinkedList(maxElementsGPU);
	queuePositionsCPU	= new LinkedList(maxElementsCPU);

	std::cerr<<"Creating cache in GPU: "<< maxElementsGPU*offsetCube*sizeof(float)/1024/1024<<" MB: "<< cudaGetErrorString(cudaMalloc((void**)&cacheDataGPU, maxElementsGPU*offsetCube*sizeof(float)))<<std::endl;
	std::cerr<<"Creating cache in CPU: "<< maxElementsCPU*offsetCube*sizeof(float)/1024/1024<<" MB: "<< cudaGetErrorString(cudaHostAlloc((void**)&cacheDataCPU, maxElementsCPU*offsetCube*sizeof(float),cudaHostAllocDefault))<<std::endl;

	// Open File
	fileManager = OpenFile(argv, levelCube, nLevels, cubeDim, cubeInc);
}

lruCache::~lruCache()
{
	delete queuePositionsGPU;
	delete queuePositionsCPU;
	delete fileManager;
	cudaFree(cacheDataGPU);
	cudaFreeHost(cacheDataCPU);
}

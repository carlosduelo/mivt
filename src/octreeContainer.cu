#include "Octree.hpp"
#include <exception> 
#include "cuda_help.hpp"
#include <iostream>
#include <fstream>

__global__ void insertOctreePointers(index_node_t ** octreeGPU, int * sizes, index_node_t * memoryGPU)
{
	int offset = 0;
	for(int i=0;i<threadIdx.x; i++)
		offset+=sizes[i];

	octreeGPU[threadIdx.x] = &memoryGPU[offset];
}

/* Lee el Octree de un fichero */
OctreeContainer::OctreeContainer(const char * file_name, int p_maxLevel)
{
	maxLevel = p_maxLevel;

	/* Read octree from file */
	std::ifstream file;
	
	try
	{
		file.open(file_name, std::ifstream::binary);
	}
	catch(...)
	{
		std::cerr<<"Octree: error opening octree file"<<std::endl;
		throw;
	}

	int magicWord;

	file.read((char*)&magicWord, sizeof(magicWord));

	if (magicWord != 919278872)
	{
		std::cerr<<"Octree: error invalid file format"<<std::endl;
		throw;
	}

	file.read((char*)&isosurface, sizeof(isosurface));
	file.read((char*)&dimension, sizeof(dimension));
	file.read((char*)&realDim.x, sizeof(realDim.x));
	file.read((char*)&realDim.y, sizeof(realDim.y));
	file.read((char*)&realDim.z, sizeof(realDim.z));
	file.read((char*)&nLevels, sizeof(int));

	std::cout<<"Octree de dimension "<<dimension<<"x"<<dimension<<"x"<<dimension<<" niveles "<<nLevels<<std::endl;

	index_node_t ** octreeCPU       = new index_node_t*[nLevels+1];
	int     *       sizesCPU        = new int[nLevels+1];

	for(int i=nLevels; i>=0; i--)
	{
		int numElem = 0;
		file.read((char*)&numElem, sizeof(numElem));
		//std::cout<<"Dimension del node en el nivel "<<i<<" es de "<<powf(2.0,*nLevels-i)<<std::endl;
		//std::cout<<"Numero de elementos de nivel "<<i<<" "<<numElem<<std::endl;
		sizesCPU[i] = numElem;
		octreeCPU[i] = new index_node_t[numElem];
		for(int j=0; j<numElem; j++)
		{
			index_node_t node = 0;
			file.read((char*) &node, sizeof(index_node_t));
			octreeCPU[i][j]= node;
		}
	}

	file.close();
	/* end reading octree from file */

	std::cerr<<"Copying octree to GPU"<<std::endl;

	int total = 0;
	for(int i=0; i<=maxLevel; i++)
		total+=sizesCPU[i];

	std::cerr<<"Allocating memory octree CUDA octree "<<(maxLevel+1)*sizeof(index_node_t*)/1024.0f/1024.0f<<" MB: "<<std::endl;
	if (cudaSuccess != (cudaMalloc(&octree, (maxLevel+1)*sizeof(index_node_t*))))
	{
		std::cerr<<"Octree: error allocating octree in the gpu"<<std::endl;
		throw;
	}

	std::cerr<<"Allocating memory octree CUDA memory "<<total*sizeof(index_node_t)/1024.0f/1024.0f<<" MB: "<<std::endl;
	if (cudaSuccess != (cudaMalloc(&memoryGPU, total*sizeof(index_node_t))))
	{
		std::cerr<<"Octree: error allocating octree in the gpu"<<std::endl;
		throw;
	}
	std::cerr<<"Allocating memory octree CUDA sizes "<<(maxLevel+1)*sizeof(int)/1024.0f/1024.0f<<" MB: "<<std::endl;
	if (cudaSuccess != (cudaMalloc(&sizes,   (maxLevel+1)*sizeof(int))))
	{
		std::cerr<<"Octree: error allocating octree in the gpu"<<std::endl;
		throw;
	}

	/* Compiando sizes */
	std::cerr<<"Octree: coping to device the sizes ";
	if (cudaSuccess != (cudaMemcpy((void*)sizes, (void*)sizesCPU, (maxLevel+1)*sizeof(int), cudaMemcpyHostToDevice)))
	{
		std::cerr<<"Fail"<<std::endl;	
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;

	/* end sizes */

	/* Copying octree */

	int offset = 0;
	for(int i=0; i<=maxLevel; i++)
	{
		std::cerr<<"Coping to device level "<<i<<": ";
		if (cudaSuccess != (cudaMemcpy((void*)(memoryGPU+offset), (void*)octreeCPU[i], sizesCPU[i]*sizeof(index_node_t), cudaMemcpyHostToDevice)))
		{
			std::cerr<<"Fail"<<std::endl;
			throw;
		}
		else
			std::cerr<<"OK"<<std::endl;

		offset+=sizesCPU[i];
	}

	dim3 blocks(1);
	dim3 threads(maxLevel+1);

	insertOctreePointers<<<blocks,threads>>>(octree, sizes, memoryGPU);
//	std::cerr<<"Launching kernek blocks ("<<blocks.x<<","<<blocks.y<<","<<blocks.z<<") threads ("<<threads.x<<","<<threads.y<<","<<threads.z<<") error: "<< cudaGetErrorString(cudaGetLastError())<<std::endl;
	std::cerr<<"Octree: sorting pointers ";
	if (cudaSuccess != cudaDeviceSynchronize())
	{
			std::cerr<<"Fail"<<std::endl;
			throw;
	}
	else
		std::cerr<<"OK"<<std::endl;

	std::cerr<<"End copying octree to GPU"<<std::endl;

	delete[] sizesCPU;
	for(int i=0; i<=nLevels; i++)
	{
		delete[] octreeCPU[i];
	}
	delete[]        octreeCPU;

}

OctreeContainer::~OctreeContainer()
{
	cudaFree(octree);
	cudaFree(memoryGPU);
	cudaFree(sizes);
}

int 		OctreeContainer::getnLevels(){ return nLevels; }

int		OctreeContainer::getMaxLevel(){ return maxLevel; }

float 		OctreeContainer::getIsosurface(){ return isosurface; }

index_node_t ** OctreeContainer::getOctree(){ return octree; }

int *		OctreeContainer::getSizes(){ return sizes; }

/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include <octreeContainer_GPU.h>
#include <cuda_help.h>

namespace eqMivt
{

__global__ void insertOctreePointers(index_node_t ** octreeGPU, int * sizes, index_node_t * memoryGPU)
{
	int offset = 0;
	for(int i=0;i<threadIdx.x; i++)
		offset+=sizes[i];

	octreeGPU[threadIdx.x] = &memoryGPU[offset];
}



bool Create_OctreeContainer(index_node_t ** octreeCPU, int * sizesCPU, int maxLevel, index_node_t *** octree, index_node_t ** memoryGPU, int ** sizes, std::string * result)
{
	int total = 0;
	for(int i=0; i<=maxLevel; i++)
		total+=sizesCPU[i];

	(*result) += "Allocating memory octree CUDA octree ";
	(*result) +=(maxLevel+1)*sizeof(index_node_t*)/1024.0f/1024.0f;
	(*result) += " MB: \n";
	if (cudaSuccess != (cudaMalloc(octree, (maxLevel+1)*sizeof(index_node_t*))))
	{
		(*result) += "Octree: error allocating octree in the gpu\n";
		return false;
	}

	(*result) += "Allocating memory octree CUDA memory ";
	(*result) += total*sizeof(index_node_t)/1024.0f/1024.0f;
	(*result) += " MB:\n";
	if (cudaSuccess != (cudaMalloc(memoryGPU, total*sizeof(index_node_t))))
	{
		(*result) += "Octree: error allocating octree in the gpu\n";
		return false;
	}
	(*result) += "Allocating memory octree CUDA sizes ";
	(*result) += (maxLevel+1)*sizeof(int)/1024.0f/1024.0f;
	(*result) += " MB:\n";
	if (cudaSuccess != (cudaMalloc(sizes,   (maxLevel+1)*sizeof(int))))
	{
		(*result) += "Octree: error allocating octree in the gpu\n";
		return false;
	}

	/* Compiando sizes */
	(*result) += "Octree: coping to device the sizes ";
	if (cudaSuccess != (cudaMemcpy((void*)*sizes, (void*)sizesCPU, (maxLevel+1)*sizeof(int), cudaMemcpyHostToDevice)))
	{
		(*result) += "Fail\n";
		return false;
	}
	else
		(*result) += "OK\n";

	/* end sizes */

	/* Copying octree */

	int offset = 0;
	for(int i=0; i<=maxLevel; i++)
	{
		(*result) += "Coping to device level ";
		(*result) += i;
		(*result) +=": ";
		if (cudaSuccess != (cudaMemcpy((void*)((*memoryGPU)+offset), (void*)octreeCPU[i], sizesCPU[i]*sizeof(index_node_t), cudaMemcpyHostToDevice)))
		{
			(*result)+="Fail\n";
			return false;
		}
		else
			(*result) += "OK\n";

		offset+=sizesCPU[i];
	}

	dim3 blocks(1);
	dim3 threads(maxLevel+1);

	insertOctreePointers<<<blocks,threads>>>(*octree, *sizes,*memoryGPU);
	//      (*result)<<"Launching kernek blocks ("<<blocks.x<<","<<blocks.y<<","<<blocks.z<<") threads ("<<threads.x<<","<<threads.y<<","<<threads.z<<") error: "<< cudaGetErrorString(cudaGetLastError())<<std::endl;
	(*result) += "Octree: sorting pointers ";
	if (cudaSuccess != cudaDeviceSynchronize())
	{
		(*result) += "Fail\n";
		return false;
	}
	else
		(*result) += "OK\n";

	(*result) += "End copying octree to GPU\n";

	return true;
}


bool Destroy_OctreeContainer(index_node_t ** octree, index_node_t * memoryGPU, int * sizes)
{
	cudaFree(memoryGPU);
	cudaFree(octree);
	cudaFree(sizes);

	return true;
}

}

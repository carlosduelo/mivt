#include "threadWorker.hpp"
#include "cuda_help.hpp"
#include <exception>
#include <iostream>
#include <fstream>
#include <strings.h>


/*
 **************************************************************************************************************************************************************************
 ++++++++++++++++++++++++++++++++++++++++++++++++++++++GPU KERNEKS+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 **************************************************************************************************************************************************************************
 */

__global__ void cuda_refactorPixelBuffer(float * pixel_buffer, int numRays, int numRaysPixel)
{
	int i 	= blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;
	
	if (i < numRays)
	{
		int index 	= 3 * i * numRaysPixel;
		int pos		= index + 3;

		for(int j=1; j<numRaysPixel; j++)
		{
			pixel_buffer[index]	+= 0; 
			pixel_buffer[index+1]	+= 0;
			pixel_buffer[index+2]	+= 0;
		}
	}
	
}


__global__ void cuda_createRays_1(int2 tile, int2 tileDim, float * rays, int numRays, float3 up, float3 right, float3 look, int H, int W, float h, float w, float distance)
{
	int id = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

        if (id < numRays)
        {
		int i  = (tile.x * tileDim.x) + (id / tile.x);
                int j  = (tile.y * tileDim.y) + (id % tile.x);

                float ih  = h/H;
                float iw  = w/W;

                float3 A = (look * distance);
                A += up * ((h/2.0f) - (ih*(i + 0.5f)));
                A += right * (-(w/2.0f) + (iw*(j + 0.5f)));
                A = normalize(A);

                rays[id]                = A.x;
                rays[id+numRays]        = A.y;
                rays[id+2*numRays]      = A.z;
		
	}
}
__global__ void cuda_createRays_2(int2 tile, int2 tileDim, float * rays, int numRays, float3 up, float3 right, float3 look, int H, int W, float h, float w, float distance)
{
	int id = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

        if (id < numRays)
        {
		int i  = (id) / W;
                int j  = (id) % W;

                float ih  = h/H;
                float iw  = w/W;

		
	}
}
__global__ void cuda_createRays_3(int2 tile, int2 tileDim, float * rays, int numRays, float3 up, float3 right, float3 look, int H, int W, float h, float w, float distance)
{
	int id = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

        if (id < numRays)
        {
		int i  = (id) / W;
                int j  = (id) % W;

                float ih  = h/H;
                float iw  = w/W;

		
	}
}
__global__ void cuda_createRays_4(int2 tile, int2 tileDim, float * rays, int numRays, float3 up, float3 right, float3 look, int H, int W, float h, float w, float distance)
{
	int id = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

        if (id < numRays)
        {
		int i  = (id) / W;
                int j  = (id) % W;

                float ih  = h/H;
                float iw  = w/W;

		
	}
}

/*
 **************************************************************************************************************************************************************************
 ++++++++++++++++++++++++++++++++++++++++++++++++++++++ METHODS CPU++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 **************************************************************************************************************************************************************************
 */

threadWorker::threadWorker(char ** argv, int id_thread, int deviceID, Camera * p_camera, Cache * p_cache, OctreeContainer * p_octreeC, rayCaster_options_t * rCasterOptions)
{
	// Setting id thread
	id.id 		= id_thread;
	id.deviceID 	= deviceID;
	
	std::cerr<<"Thread: " << id.id<<" started device "<<id.deviceID<<": ";
	if (cudaSuccess != cudaSetDevice(id.deviceID))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;

	std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" Createing cudaStream: ";
	if (cudaSuccess != cudaStreamCreate(&id.stream))
	{
		std::cerr<<"Fail"<<std::endl;
	}
	else
		std::cerr<<"OK"<<std::endl;

	camera		= p_camera;
	cache 		= p_cache;
	pipe		= new Channel(MAX_WORKS);
	raycaster	= new rayCaster(p_octreeC->getIsosurface(), rCasterOptions);
	int2 tileDim	= camera->getTileDim();
	numRays		= tileDim.x * tileDim.y * camera->getNumRayPixel() * camera->getNumRayPixel();
	maxRays		= tileDim.x * tileDim.y * camera->getMaxRayPixel() * camera->getMaxRayPixel();

	createStructures();

	if (strcmp(argv[0], "complete_GPU") == 0)
	{
		octree 		= new Octree_completeGPU(p_octreeC, maxRays);
	}
	else
	{
		std::cerr<<"Error: octree option error"<<std::endl;
		throw;
	}

}

threadWorker::~threadWorker()
{
	cudaSetDevice(id.deviceID);

	delete pipe;
	delete octree;
	delete raycaster;
	destroyStructures();
}

void threadWorker::destroyStructures()
{
	cudaFree(visibleCubesGPU);
	cudaFreeHost(visibleCubesCPU);
	cudaFree(rays);
	cudaFree(pixel_buffer);
}

void threadWorker::createStructures()
{
	std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" Allocating memory visibleCubesCPU "<<maxRays*sizeof(visibleCube_t)/1024/1024 <<" MB : ";
	if (cudaSuccess != cudaHostAlloc((void**)&visibleCubesCPU, maxRays*sizeof(visibleCube_t), cudaHostAllocDefault))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;
        std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" Allocating memory visibleCubesGPU "<<maxRays*sizeof(visibleCube_t)/1024/1024 <<" MB : ";
	if (cudaSuccess != cudaMalloc((void**)&visibleCubesGPU, maxRays*sizeof(visibleCube_t)))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;

	resetVisibleCubes();

	// Create rays
        std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" Allocating memory rays "<<maxRays*3*sizeof(float)/1024/1024 <<" MB : ";
	if (cudaSuccess != cudaMalloc((void**)&rays, 3*maxRays*sizeof(float)))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;

	// Create pixle_buffer
        std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" Allocating memory pixel_buffer "<<maxRays*3*sizeof(float)/1024/1024 <<" MB : ";
	if (cudaSuccess != cudaMalloc((void**)&pixel_buffer, 3*maxRays*sizeof(float)))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;
}

Channel * threadWorker::getChannel()
{
	return pipe;
}

void threadWorker::resetVisibleCubes()
{
	cudaMemsetAsync((void*)visibleCubesGPU, 0, maxRays*sizeof(visibleCube_t), id.stream);
}

void threadWorker::createRays(int2 tile, int numPixels)
{
	dim3 threads = getThreads(numPixels);
        dim3 blocks = getBlocks(numPixels);
	
	switch(camera->getNumRayPixel())
	{
		case 1:
		{
			cuda_createRays_1<<<blocks, threads, 0, id.stream>>>(tile, camera->getTileDim() ,rays, numPixels, camera->get_up(), camera->get_right(), camera->get_look(), camera->getHeight(), camera->getWidth(), camera->getHeight_screen(), camera->getWidth_screen(), camera->getDistance());
			break;
		}
		case 2:
		{
			cuda_createRays_2<<<blocks, threads, 0, id.stream>>>(tile, camera->getTileDim() ,rays, numPixels, camera->get_up(), camera->get_right(), camera->get_look(), camera->getHeight(), camera->getWidth(), camera->getHeight_screen(), camera->getWidth_screen(), camera->getDistance());
			break;
		}
		case 3:
		{
			cuda_createRays_3<<<blocks, threads, 0, id.stream>>>(tile, camera->getTileDim() ,rays, numPixels, camera->get_up(), camera->get_right(), camera->get_look(), camera->getHeight(), camera->getWidth(), camera->getHeight_screen(), camera->getWidth_screen(), camera->getDistance());
			break;
		}
		case 4:
		{
			cuda_createRays_4<<<blocks, threads, 0, id.stream>>>(tile, camera->getTileDim() ,rays, numPixels, camera->get_up(), camera->get_right(), camera->get_look(), camera->getHeight(), camera->getWidth(), camera->getHeight_screen(), camera->getWidth_screen(), camera->getDistance());
			break;
		}
		default:
		{
			std::cerr<<"Error: numRayPixel not valid"<<std::endl;
			throw;
		}
	}
}

void threadWorker::refactorPixelBuffer(int numPixels)
{
	if (camera->getNumRayPixel() > 1)
	{
		dim3 threads = getThreads(numPixels);
        	dim3 blocks = getBlocks(numPixels);
 		cuda_refactorPixelBuffer<<<blocks, threads, 0 , id.stream>>>(pixel_buffer, numPixels, camera->getNumRayPixel()*camera->getNumRayPixel());
	}
}

void threadWorker::createFrame(int2 tile, float * buffer)
{
	// Reset visible cubes
	resetVisibleCubes();
	
	// Create rays
	int2 tileDim = camera->getTileDim();
	int numPixels = tileDim.x * tileDim.y;
	createRays(tile, numPixels);

	// Create frame
	bool notEnd = true;
        int iterations = 0;

	// Reset octree state
	octree->resetState(id.stream);

        while(notEnd)
        {
		octree->getBoxIntersected(camera->get_position(), rays, numRays, visibleCubesGPU, visibleCubesCPU, id.stream);
		if (cudaSuccess != cudaStreamSynchronize(id.stream))
		{
			std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<": Error creating frame on tile ("<<tile.x<<","<<tile.y<<")"<<std::endl;
			throw;
		}

		cache->push(visibleCubesCPU, numRays, octree->getOctreeLevel(), &id);
                int numP = 0;
                for(int i=0; i<numRays; i++)
                        if (visibleCubesCPU[i].state == PAINTED)
                                numP++;

                if (numP == numRays)
                {
                        notEnd = false;
                        break;
                }

                cudaMemcpyAsync((void*) visibleCubesGPU, (const void*) visibleCubesCPU, numRays*sizeof(visibleCube_t), cudaMemcpyHostToDevice, id.stream);
		if (cudaSuccess != cudaStreamSynchronize(id.stream))
		{
			std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<": Error creating frame on tile ("<<tile.x<<","<<tile.y<<")"<<std::endl;
			throw;
		}

                raycaster->render(rays, numRays, camera->get_position(), octree->getOctreeLevel(), cache->getCacheLevel(), octree->getnLevels(), visibleCubesGPU, cache->getCubeDim(), cache->getCubeInc(), pixel_buffer, id.stream);

		cache->pop(visibleCubesCPU, numRays, octree->getOctreeLevel(), &id);

                iterations++;
	}

	// Refactor pixel_buffer and copy
	refactorPixelBuffer(numPixels);

	// DANGER, I am not sure, this works
	cudaMemcpy2DAsync((void*)buffer, 3*camera->getWidth()*sizeof(float), (void*) pixel_buffer, 3*tileDim.y*sizeof(float), 3*tileDim.y*sizeof(float), 3*tileDim.x, cudaMemcpyDeviceToHost, id.stream);

	if (cudaSuccess != cudaStreamSynchronize(id.stream))
	{
		std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<": Error creating frame on tile ("<<tile.x<<","<<tile.y<<")"<<std::endl;
		throw;
	}

	std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" iterations per frame "<<iterations<<std::endl;
}

void threadWorker::run()
{
	std::cerr<<"Thread: " << id.id<<" started device "<<id.deviceID<<": ";
	if (cudaSuccess != cudaSetDevice(id.deviceID))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"OK"<<std::endl;

	bool end = false;

	while(!end)
	{
		work_packet_t work = pipe->pop();

		switch(work.work_id)
		{
			case UP_LEVEL_OCTREE:
			{
				std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" increase octree level"<<std::endl;
				octree->increaseLevel();
				break;
			}
			case DOWN_LEVEL_OCTREE:
			{
				std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" decrease octree level"<<std::endl;
				octree->decreaseLevel();
				break;
			}
			case INCREASE_STEP:
			{
				std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" increase step"<<std::endl;
				raycaster->increaseStep();
				break;
			}
			case DECREASE_STEP:
			{
				std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" decrease step"<<std::endl;
				raycaster->decreaseStep();
				break;
			}
			case CHANGE_ANTIALIASSING:
			{
				int2 tileDim	= camera->getTileDim();
				numRays		= tileDim.x * tileDim.y * camera->getNumRayPixel();
				std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" change supersampling"<<std::endl;
				break;
			}
			case END:
			{
				std::cout<<"Thread "<<id.id<<" on device "<<id.deviceID<<" End thread work"<<std::endl;
				end = true;
				break;
			}
			case NEW_TILE:
			{
				std::cout<<"Thread "<<id.id<<" on device "<<id.deviceID<<" New Tile"<<std::endl;
				createFrame(work.tile, work.pixel_buffer);
				break;
			}
			default:
			{
				std::cerr<<"Error: thread "<<id.id<<" on device "<<id.deviceID<<" recieve a not valid command"<<std::endl;
				throw;
			}
		}

		std::cout<<"Thread: " << id.id<<" on device "<<id.deviceID<<" new task done"<<std::endl;
	}
}

#include "threadWorker.hpp"
#include "cuda_help.hpp"
#include <exception>
#include <iostream>
#include <fstream>
#include <strings.h>

#define SECRET_WORD 345678987


/*
 **************************************************************************************************************************************************************************
 ++++++++++++++++++++++++++++++++++++++++++++++++++++++GPU KERNEKS+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 **************************************************************************************************************************************************************************
 */


// NOT WORK, OF COURSE
__global__ void cuda_refactorPixelBuffer(float * pixel_buffer, int numRays, int numRaysPixel)
{
	int i 	= blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;
	
	if (i < numRays)
	{
		int index 	= 3 * i * numRaysPixel;
		int pos		= index + 3;
	
		for(int j=1; j<numRaysPixel; j++)
		{
			pixel_buffer[index]   += pixel_buffer[pos]; 
			pixel_buffer[index+1] += pixel_buffer[pos+1];
			pixel_buffer[index+2] += pixel_buffer[pos+2];
		}
		pixel_buffer[index]   /= numRaysPixel; 
		pixel_buffer[index+1] /= numRaysPixel;
		pixel_buffer[index+2] /= numRaysPixel;
	}
}

__global__ void cuda_createRays_1(int2 tile, int2 tileDim, float * rays, int numRays, float3 up, float3 right, float3 look, int H, int W, float h, float w, float distance)
{
	int id = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

        if (id < numRays)
        {
		int i  = (tile.x * tileDim.x) + (id / tileDim.y);
                int j  = (tile.y * tileDim.y) + (id % tileDim.y);
	
		//printf("%d %d %d %d %d %d %d\n",id,tile.x, tile.y,tileDim.x,tileDim.y,i,j);

                float ih  = h/H;
                float iw  = w/W;

                float3 A = (look * distance);
                A += -up * 	((h/2.0f) - (ih*(i + 0.5f)));
                A += right * 	(-(w/2.0f) + (iw*(j + 0.5f)));
                A = normalize(A);

                rays[id]                = A.x;
                rays[id+numRays]        = A.y;
                rays[id+2*numRays]      = A.z;
		
	}
}
__global__ void cuda_createRays_2(int2 tile, int2 tileDim, float * rays, int numRays, int nRP, float3 up, float3 right, float3 look, int H, int W, float h, float w, float distance)
{
	int id = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

        if (id < numRays)
        {
		int i  = (tile.x * tileDim.x) + (id / tileDim.y);
                int j  = (tile.y * tileDim.y) + (id % tileDim.y);
	
                float ih  = h/H;
                float iw  = w/W;

                float3 B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (1.0f/3.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (1.0f/3.0f))));
                B = normalize(B);

                rays[id*nRP]                	= B.x;
                rays[id*nRP+numRays*nRP]	= B.y;
                rays[id*nRP+2*numRays*nRP]      = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (1.0f/3.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (2.0f/3.0f))));
                B = normalize(B);

                rays[id*nRP+1]			= B.x;
                rays[id*nRP+1+numRays*nRP]      = B.y;
                rays[id*nRP+1+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (2.0f/3.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (1.0f/3.0f))));
                B = normalize(B);

                rays[id*nRP+2]			= B.x;
                rays[id*nRP+2+numRays*nRP]    	= B.y;
                rays[id*nRP+2+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (2.0f/3.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (2.0f/3.0f))));
                B = normalize(B);

                rays[id*nRP+3]                	= B.x;
                rays[id*nRP+3+numRays*nRP]      = B.y;
                rays[id*nRP+3+2*numRays*nRP]    = B.z;
	}
}
__global__ void cuda_createRays_3(int2 tile, int2 tileDim, float * rays, int numRays, int nRP, float3 up, float3 right, float3 look, int H, int W, float h, float w, float distance)
{
	int id = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

        if (id < numRays)
        {
		int i  = (tile.x * tileDim.x) + (id / tileDim.y);
                int j  = (tile.y * tileDim.y) + (id % tileDim.y);
	
                float ih  = h/H;
                float iw  = w/W;

                float3 B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (1.0f/6.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (1.0f/6.0f))));
                B = normalize(B);

                rays[id*nRP]                	= B.x;
                rays[id*nRP+numRays*nRP]	= B.y;
                rays[id*nRP+2*numRays*nRP]      = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (1.0f/6.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (5.0f/6.0f))));
                B = normalize(B);

                rays[id*nRP+1]			= B.x;
                rays[id*nRP+1+numRays*nRP]      = B.y;
                rays[id*nRP+1+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (5.0f/6.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (5.0f/6.0f))));
                B = normalize(B);

                rays[id*nRP+2]			= B.x;
                rays[id*nRP+2+numRays*nRP]    	= B.y;
                rays[id*nRP+2+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (5.0f/6.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (1.0f/6.0f))));
                B = normalize(B);

                rays[id*nRP+3]                	= B.x;
                rays[id*nRP+3+numRays*nRP]      = B.y;
                rays[id*nRP+3+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (2.0f/6.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (1.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+4]			= B.x;
                rays[id*nRP+4+numRays*nRP]      = B.y;
                rays[id*nRP+4+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (1.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (4.0f/6.0f))));
                B = normalize(B);

                rays[id*nRP+5]			= B.x;
                rays[id*nRP+5+numRays*nRP]    	= B.y;
                rays[id*nRP+5+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (4.0f/6.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (1.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+6]                	= B.x;
                rays[id*nRP+6+numRays*nRP]      = B.y;
                rays[id*nRP+6+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (1.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (2.0f/6.0f))));
                B = normalize(B);

                rays[id*nRP+7]			= B.x;
                rays[id*nRP+7+numRays*nRP]    	= B.y;
                rays[id*nRP+7+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (1.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (1.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+8]                	= B.x;
                rays[id*nRP+8+numRays*nRP]      = B.y;
                rays[id*nRP+8+2*numRays*nRP]    = B.z;
	}
}
__global__ void cuda_createRays_4(int2 tile, int2 tileDim, float * rays, int numRays, int nRP, float3 up, float3 right, float3 look, int H, int W, float h, float w, float distance)
{
	int id = blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x + threadIdx.x;

        if (id < numRays)
        {
		int i  = (tile.x * tileDim.x) + (id / tileDim.y);
                int j  = (tile.y * tileDim.y) + (id % tileDim.y);
	
                float ih  = h/H;
                float iw  = w/W;

                float3 B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (1.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (1.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP]                	= B.x;
                rays[id*nRP+numRays*nRP]	= B.y;
                rays[id*nRP+2*numRays*nRP]      = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (1.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (2.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+1]			= B.x;
                rays[id*nRP+1+numRays*nRP]      = B.y;
                rays[id*nRP+1+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (1.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (3.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+2]			= B.x;
                rays[id*nRP+2+numRays*nRP]    	= B.y;
                rays[id*nRP+2+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (1.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (4.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+3]                	= B.x;
                rays[id*nRP+3+numRays*nRP]      = B.y;
                rays[id*nRP+3+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (2.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (1.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+4]			= B.x;
                rays[id*nRP+4+numRays*nRP]      = B.y;
                rays[id*nRP+4+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (2.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (2.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+5]			= B.x;
                rays[id*nRP+5+numRays*nRP]    	= B.y;
                rays[id*nRP+5+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (2.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (3.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+6]                	= B.x;
                rays[id*nRP+6+numRays*nRP]      = B.y;
                rays[id*nRP+6+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (2.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (4.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+7]			= B.x;
                rays[id*nRP+7+numRays*nRP]    	= B.y;
                rays[id*nRP+7+2*numRays*nRP]    = B.z;

		B += -up * 	((h/2.0f) - (ih*(i + (3.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (1.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+8]                	= B.x;
                rays[id*nRP+8+numRays*nRP]	= B.y;
                rays[id*nRP+8+2*numRays*nRP]      = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (3.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (2.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+9]			= B.x;
                rays[id*nRP+9+numRays*nRP]      = B.y;
                rays[id*nRP+9+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (2.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (1.0f/5.0f))));
                B = normalize(B);

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (3.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (3.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+10]			= B.x;
                rays[id*nRP+10+numRays*nRP]    	= B.y;
                rays[id*nRP+10+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (3.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (4.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+11]                	= B.x;
                rays[id*nRP+11+numRays*nRP]      = B.y;
                rays[id*nRP+11+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (4.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (1.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+12]			= B.x;
                rays[id*nRP+12+numRays*nRP]      = B.y;
                rays[id*nRP+12+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (4.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (2.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+13]			= B.x;
                rays[id*nRP+13+numRays*nRP]    	= B.y;
                rays[id*nRP+13+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (4.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (3.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+14]                	= B.x;
                rays[id*nRP+14+numRays*nRP]      = B.y;
                rays[id*nRP+14+2*numRays*nRP]    = B.z;

                B = (look * distance);
		B += -up * 	((h/2.0f) - (ih*(i + (4.0f/5.0f))));
		B += right * 	(-(w/2.0f) + (iw*(j + (4.0f/5.0f))));
                B = normalize(B);

                rays[id*nRP+15]			= B.x;
                rays[id*nRP+15+numRays*nRP]    	= B.y;
                rays[id*nRP+15+2*numRays*nRP]    = B.z;
	}
}

void sendSignalToMaster(cudaStream_t stream, cudaError_t status, void *data)
{
	threadWorker * worker = (threadWorker*) data;
	
	worker->signalFinishFrame(SECRET_WORD);	
}

// FILL PIXEL_BUFFER TESTING PROPOUSES
__global__ void cuda_fill_pixel_buffer(float * pixel_buffer, int numRays, int numRaysPixel, int2 tile, int2 tileDim, int width, int height)
{
	int i 	= blockIdx.y * blockDim.x * gridDim.y + blockIdx.x * blockDim.x +threadIdx.x;
	
	if (i < numRays)
	{
		int pos = i*3;
		int ray = i % numRaysPixel;
		int x = i / tileDim.x;
		int y = i % tileDim.x;

		pixel_buffer[pos]	= (float)(x + tile.x * tileDim.x) / (float)height;
		pixel_buffer[pos+1]	= (float)(y + tile.y * tileDim.y) / (float)width;
		pixel_buffer[pos+2]	= 0.0f;
	}
}

/*
 **************************************************************************************************************************************************************************
 ++++++++++++++++++++++++++++++++++++++++++++++++++++++ METHODS CPU++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 **************************************************************************************************************************************************************************
 */

threadWorker::threadWorker(char ** argv, int id_thread, int id_global, int deviceID, Camera * p_camera, Cache * p_cache, OctreeContainer * p_octreeC, rayCaster_options_t * rCasterOptions)
{
	// Setting id thread
	id.id 		= id_thread;
	id.id_global	= id_global;
	id.deviceID 	= deviceID;
	numWorks	= 0;
	
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
	numRays		= tileDim.x * tileDim.y * camera->getNumRayPixel();
	maxRays		= tileDim.x * tileDim.y * camera->getMaxRayPixel();

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
		case 4:
		{
			cuda_createRays_2<<<blocks, threads, 0, id.stream>>>(tile, camera->getTileDim() ,rays, numPixels, camera->getNumRayPixel(), camera->get_up(), camera->get_right(), camera->get_look(), camera->getHeight(), camera->getWidth(), camera->getHeight_screen(), camera->getWidth_screen(), camera->getDistance());
			break;
		}
		case 9:
		{
			cuda_createRays_3<<<blocks, threads, 0, id.stream>>>(tile, camera->getTileDim() ,rays, numPixels, camera->getNumRayPixel(), camera->get_up(), camera->get_right(), camera->get_look(), camera->getHeight(), camera->getWidth(), camera->getHeight_screen(), camera->getWidth_screen(), camera->getDistance());
			break;
		}
		case 16:
		{
			cuda_createRays_4<<<blocks, threads, 0, id.stream>>>(tile, camera->getTileDim() ,rays, numPixels, camera->getNumRayPixel(), camera->get_up(), camera->get_right(), camera->get_look(), camera->getHeight(), camera->getWidth(), camera->getHeight_screen(), camera->getWidth_screen(), camera->getDistance());
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
		float * caca = new float[3*maxRays];

		cudaMemcpyAsync((void*)caca, (void*) pixel_buffer, 3*maxRays*sizeof(float), cudaMemcpyDeviceToHost, id.stream);
		if (cudaSuccess != cudaStreamSynchronize(id.stream))
		{
			throw;
		}
		for(int i=0; i<numPixels; i++)
		{
			int pos = 3 * i * camera->getNumRayPixel();

			float r = 0.0f;
			float g = 0.0f;
			float b = 0.0f;
			
			for(int j=0; j<3*camera->getNumRayPixel(); j+=3)
			{
				r     	+= caca[pos+j];
				g	+= caca[pos+j+1];
				b	+= caca[pos+j+2];
			}
			caca[3*i]	= r / camera->getNumRayPixel(); 
			caca[3*i+1]	= g / camera->getNumRayPixel();
			caca[3*i+2]	= b / camera->getNumRayPixel();
		}

		cudaMemcpyAsync((void*)pixel_buffer, (void*) caca, 3*maxRays*sizeof(float), cudaMemcpyHostToDevice, id.stream);
		delete[] caca;
		#if 0
		dim3 threads = getThreads(numPixels);
        	dim3 blocks = getBlocks(numPixels);
 		cuda_refactorPixelBuffer<<<blocks, threads, 0 , id.stream>>>(pixel_buffer, numPixels, camera->getNumRayPixel());
		#endif
	}
}

void threadWorker::createFrame(int2 tile, float * buffer)
{
	int2 tileDim 	= camera->getTileDim();
	int numPixels 	= tileDim.x * tileDim.y;
	bool notEnd 	= true;
        int iterations 	= 0;

#if 1
	// Reset visible cubes
	resetVisibleCubes();
	
	// Create rays
	createRays(tile, numPixels);

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

                raycaster->render(rays, numRays, camera->get_position(), octree->getOctreeLevel(), cache->getCacheLevel(), octree->getnLevels(), visibleCubesGPU, cache->getCubeDim(), cache->getCubeInc(), pixel_buffer, id.stream);

		#if 0
                cudaMemcpyAsync((void*) visibleCubesCPU, (const void*) visibleCubesGPU, numRays*sizeof(visibleCube_t), cudaMemcpyDeviceToHost, id.stream);

		if (cudaSuccess != cudaStreamSynchronize(id.stream))
		{
			std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<": Error creating frame on tile ("<<tile.x<<","<<tile.y<<")"<<std::endl;
			throw;
		}
		#endif

		cache->pop(visibleCubesCPU, numRays, octree->getOctreeLevel(), &id);

                iterations++;
	}
#else
	dim3 threads = getThreads(numRays);
	dim3 blocks = getBlocks(numRays);
	cuda_fill_pixel_buffer<<<blocks, threads, 0, id.stream>>>(pixel_buffer, numRays, camera->getNumRayPixel(), tile, camera->getTileDim(), camera->getWidth(), camera->getHeight());
#endif

	// Refactor pixel_buffer and copy
	refactorPixelBuffer(numPixels);

	// DANGER, I am not sure, this works
	cudaMemcpy2DAsync((void*)buffer, 3*camera->getWidth()*sizeof(float), (void*) pixel_buffer, 3*tileDim.y*sizeof(float), 3*tileDim.y*sizeof(float), tileDim.x, cudaMemcpyDeviceToHost, id.stream);

	if ( cudaSuccess != cudaStreamAddCallback(id.stream, sendSignalToMaster, (void*)this, 0))
	{
		std::cerr<<"Error making cudaCallback"<<std::endl;
		throw;
	}

	std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<" iterations per frame "<<iterations<<" for tile "<<tile.x<<" "<<tile.y<<std::endl;
}


void threadWorker::waitFinishFrame()
{
	endFrame.set();
	endFrame.unset();
}

void threadWorker::signalFinishFrame(int secret_word)
{
	if (secret_word == SECRET_WORD)
	{
		numWorks--;
	}
	else
		std::cerr<<"Warning! you cannot call this funcion if you are not the same thread!!"<<std::endl;
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
			case NEW_FRAME:
			{
				numWorks = 0;
				recivedEndFrame = false;
				endFrame.set();
				std::cout<<"Thread "<<id.id<<" on device "<<id.deviceID<<" New frame recieved"<<std::endl;
				break;
			} 
			case END_FRAME:
			{
				recivedEndFrame = true;
				if (cudaSuccess != cudaStreamSynchronize(id.stream))
				{
					std::cerr<<"Thread "<<id.id<<" on device "<<id.deviceID<<": Error waiting to unlock master"<<std::endl;
					throw;
				}
				endFrame.unset();
				std::cout<<"Thread "<<id.id<<" on device "<<id.deviceID<<" End frame recieved"<<std::endl;
				break;
			}
			case NEW_TILE:
			{
				numWorks++;
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

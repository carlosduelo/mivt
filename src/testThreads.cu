#include "threadMaster.hpp"
#include "FreeImage.h"
#include <exception>
#include <iostream>
#include <fstream>

int main(int argc, char ** argv)
{
	if (argc < 3)
	{
		std::cerr<<"Error, testFileManger file_type file [dataset_name]"<<std::endl;
		return 0;
	}

	initParams_masterWorker_t params;

	// Workers
	params.numDevices	= 3;
	params.numWorkers[0]	= 4;
	params.numWorkers[1]	= 4;
	params.numWorkers[2]	= 4;
	params.deviceID[0]	= 2;
	params.deviceID[1]	= 0;
	params.deviceID[2]	= 1;

	// Cache
	params.maxElementsCache[0]	= 5500;
	params.maxElementsCache[1]	= 2500;
	params.maxElementsCache[2]	= 2500;
	params.cubeInc			= 2;
	params.cubeDim			= make_int3(32,32,32);
	params.levelCube		= 4;

	// Octree
	params.maxLevelOctree	= 9;

	// ray caster
	params.rayCasterOptions.ligth_position = make_float3(512.0f, 512.0f, 512.0f);

	// Camera
	params.displayOptions.height		= 1024;
	params.displayOptions.width		= 1024;
	params.displayOptions.distance		= 50.0f;
	params.displayOptions.fov_H		= 30.0f;
	params.displayOptions.fov_W		= 30.0f;
	params.displayOptions.numRayPixel	= 1;
	params.displayOptions.tileDim		= make_int2(32,32);
	params.displayOptions.position		= make_float3(2048.0f, 180.0f, 3500.0f);

	threadMaster * mivt = new threadMaster(&argv[1], &params);

	mivt->increaseLevelOctree();
	mivt->decreaseLevelOctree();

	float * buffer = 0;
	std::cerr<<"Allocating pixel buffer: ";
	if (cudaSuccess != cudaMallocHost((void**)&buffer, 3*params.displayOptions.height*params.displayOptions.width*sizeof(float)))
	{
		std::cerr<<"Fail"<<std::endl;
		throw;
	}
	else
		std::cerr<<"Ok"<<std::endl;

	FreeImage_Initialise();
	FIBITMAP * bitmap = FreeImage_Allocate(params.displayOptions.width, params.displayOptions.height, 24);
	RGBQUAD color;


	for(int frame=0; frame<10; frame++)
	{
		mivt->createFrame(buffer);
		mivt->StrafeRight(0.5f);

		for(int i=0; i<params.displayOptions.height; i++)
			for(int j=0; j<params.displayOptions.width; j++)
			{
				int id = i*params.displayOptions.width + j;
				color.rgbRed 	= buffer[id*3]*255;
				color.rgbGreen 	= buffer[id*3+1]*255;
				color.rgbBlue 	= buffer[id*3+2]*255;
				FreeImage_SetPixelColor(bitmap, j, i, &color);
			}

		std::stringstream name;
		name<<"prueba"<<frame<<".png";
		FreeImage_Save(FIF_PNG, bitmap, name.str().c_str(), 0);
	}

	FreeImage_DeInitialise();
	cudaFreeHost(buffer);
	delete mivt;
}

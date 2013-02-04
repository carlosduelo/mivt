#include "threadMaster.hpp"
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
	params.numWorkers	= 2;
	params.deviceID		= 0;

	// Cache
	params.maxElementsCache	= 100;
	params.cubeInc		= 2;
	params.cubeDim		= make_int3(32,32,32);
	params.levelCube	= 4;

	// Octree
	params.maxLevelOctree	= 9;

	// ray caster
	params.rayCasterOptions.ligth_position = make_float3(512.0f, 512.0f, 512.0f);

	// Camera
	params.displayOptions.height		= 512;
	params.displayOptions.width		= 512;
	params.displayOptions.distance		= 50.0f;
	params.displayOptions.fov_H		= 30.0f;
	params.displayOptions.fov_W		= 30.0f;
	params.displayOptions.numRayPixel	= 1;
	params.displayOptions.tileDim		= make_int2(32,32);
	params.displayOptions.position		= make_float3(128.0f, 128.0f, 512.0f);

	threadMaster * mivt = new threadMaster(&argv[1], &params);

	mivt->increaseLevelOctree();
	mivt->decreaseLevelOctree();

	delete mivt;
}

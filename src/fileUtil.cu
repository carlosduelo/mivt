#include "FileManager.hpp"

FileManager * OpenFile(char ** argv, int p_levelCube, int p_nLevels, int3 p_cubeDim, int3 p_cubeInc)
{
	return new hdf5File(argv[0], argv[1], p_levelCube, p_nLevels, p_cubeDim, p_cubeInc);
}

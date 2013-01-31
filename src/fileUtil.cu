#include "FileManager.hpp"
#include <exception> 
#include <iostream>
#include <fstream>
#include "strings.h"

FileManager * OpenFile(char ** argv, int p_levelCube, int p_nLevels, int3 p_cubeDim, int3 p_cubeInc)
{
	if (strcmp(argv[0], "hdf5_file") == 0)
	{
		return new hdf5File(argv[1], argv[2], p_levelCube, p_nLevels, p_cubeDim, p_cubeInc);
	}
	else
	{
		std::cerr<<"Error: wrong file option"<<std::endl;
		throw;
	}

}

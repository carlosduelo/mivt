#include "FileManager.hpp"
#include <iostream>
#include <fstream>


int main(int argc, char ** argv)
{
	if (argc < 3)
	{
		std::cerr<<"Error, testFileManger hdf5_file dataset_name"<<std::endl;
		return 0;
	}

	FileManager *  fileManager = new hdf5File(argv[1], argv[2], 4, 9, make_int3(32,32,32), make_int3(2,2,2));

	float * data = new float[36*36*36];

	fileManager->readCube(5500, data);

	for(int i=0;i<(36*36*36); i++)
		std::cout<<data[i]<<std::endl;

	delete fileManager;
	delete[] data;
}

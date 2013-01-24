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

	File *  fileManager = FileFactory.OpenFile(argv[1], argv[2], 4, 9, make_int3(32,32,32), make_int3(2,2,2));

	delete fileManager;
}

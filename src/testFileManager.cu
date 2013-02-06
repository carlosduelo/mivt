#include "fileUtil.hpp"
#include "mortonCodeUtil.hpp"
#include <iostream>
#include <fstream>

int main(int argc, char ** argv)
{
	if (argc < 4)
	{
		std::cerr<<"Error, testFileManger file_type file [dataset_name]"<<std::endl;
		return 0;
	}

	FileManager *  fileManager = OpenFile(&argv[1], 8, 9, make_int3(32,32,32), make_int3(2,2,2));

	float * data = new float[36*36*36];

	index_node_t inicio 	= coordinateToIndex(make_int3(0,0,0), 8, 9);
	index_node_t fin	= coordinateToIndex(make_int3(255,255,255),8,9); 

	while(inicio != fin)
	{
		fileManager->readCube(inicio, data);
		inicio++;
	}

	delete fileManager;
	delete[] data;
}

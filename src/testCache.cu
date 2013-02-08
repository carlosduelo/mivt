#include "lruCache.hpp"
#include "mortonCodeUtil.hpp"
#include <exception>
#include <iostream>
#include <fstream>
#include <stdlib.h>

#define CUBES 1000

int main(int argc, char ** argv)
{
	if (argc < 3)
	{
		std::cerr<<"Error, testCache file_type file [dataset_name] cache_type"<<std::endl;
		return 0;
	}

	Cache * cache = new Cache(&argv[1], 0,1, 100, make_int3(4,4,4), 2, 7, 9);	

	index_node_t inicio     = coordinateToIndex(make_int3(0,0,0), 8, 9);
        index_node_t fin        = coordinateToIndex(make_int3(255,255,255), 8, 9);

	int iterations = ((fin-inicio)/CUBES);
	
	visibleCube_t * cubes = new visibleCube_t[CUBES];
	threadID_t      thread;
	thread.id = 8;

	for(int i=0; i<iterations; i++)
	{
		std::cout<<"New iteration"<<std::endl;
		// init
		for(int j=0;j<CUBES; j++)
		{
			cubes[j].id 	= (rand()%(fin-inicio))+inicio;;
			cubes[j].state 	= CUBE;
		}
		int iter = 0;
		bool end = false;
		while(!end)
		{
			cache->push(cubes, CUBES, 8, &thread);
			cache->pop(cubes, CUBES, 8, &thread);

			int numPainted = 0;
			for(int j=0;j<CUBES; j++)
                	{
				if (cubes[j].state  == PAINTED)
				{
					numPainted++;
				}
				if (cubes[j].state  == CACHED)
				{
					cubes[j].state  = PAINTED;
				}
			}
			if (numPainted == CUBES)
				end = true;
			iter++;
		}
		std::cout<<" total iterations "<<iter<<std::endl;
	}
	

	delete cache;
}

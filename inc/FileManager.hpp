/*
 * FileManager
 * 
 * Author: Carlos Duelo Serrano
 */

#ifndef _FILE_MANAGER_H_
#define _FILE_MANAGER_H_

#include <config.hpp>
#include <cutil_math.h>
#include <hdf5.h>

class File
{
	protected:
		int	levelCube;
		int	nLevels;
		int3	cubeDim;
		int3	cubeInc;
		int3	realCubeDim;
		
	public:
		File(int p_levelCube, int p_nLevels, int3 p_cubeDim, int3 p_cubeInc)
		{
			levelCube 	= p_levelCube;
			nLevels		= p_nLevels;
			cubeDim		= p_cubeDim;
			cubeInc		= p_cubeInc;
			realCubeDim	= p_cubeDim + 2*p_cubeInc;
		}
		virtual void readCube(index_node_t index, float * cube);
		virtual ~File() { }
};


class hdf5File : public File
{
	private:
		// HDF5 stuff
		hid_t           file_id;
		hid_t           dataset_id;
		hid_t           spaceid;
		int             ndim;

	public:
		hsize_t         dims[3];

		hdf5File(const char * file_name, const char * dataset_name, int p_levelCube, int p_nLevels, int3 p_cubeDim, int3 p_cubeInc);

		~hdf5File();

		void readCube(index_node_t index, float * cube);
};

class FileFactory
{
	public:
		File * OpenFile(char ** argv, int p_levelCube, int p_nLevels, int3 p_cubeDim, int3 p_cubeInc)
		{
			return new hdf5File(argv[0], argv[1], p_levelCube, p_nLevels, p_cubeDim, p_cubeInc);
		}
		
		~FileFactory() { }
};

#endif

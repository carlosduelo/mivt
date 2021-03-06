#include <FileManager.hpp>
#include <mortonCodeUtil.hpp>
#include <exception>
#include <iostream>
#include <strings.h>

hdf5File::hdf5File(const char * file_name, const char * dataset_name, int p_levelCube, int p_nLevels, int3 p_cubeDim, int3 p_cubeInc) : FileManager(p_levelCube, p_nLevels, p_cubeDim, p_cubeInc)
{
	if ((file_id    = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT)) < 0)
	{
		std::cerr<<"hdf5: opening "<<file_name<<std::endl;
		throw;
	}

	if ((dataset_id = H5Dopen1(file_id, dataset_name)) < 0 )
	{
		std::cerr<<"hdf5: unable to open the requested data set "<<dataset_name<<std::endl;
		throw;	
	}

	if ((spaceid    = H5Dget_space(dataset_id)) < 0)
	{
		std::cerr<<"hdf5: unable to open the requested data space"<<std::endl;
		throw; 
	}

	if ((ndim       = H5Sget_simple_extent_dims (spaceid, dims, NULL)) < 0)
	{
		std::cerr<<"hdf5: handling file"<<std::endl;
		throw;
	}
}

hdf5File::~hdf5File()
{
	herr_t      status;

	if ((status = H5Dclose(dataset_id)) < 0)
	{
		std::cerr<<"hdf5: unable to close the data set"<<std::endl;
		throw;
	}


	if ((status = H5Fclose(file_id)) < 0);
	{
		std::cerr<<"hdf5: unable to close the file"<<std::endl;
		/*
		 * XXX cduelo: No entiendo porque falla al cerrar el fichero....
		 *
		 */
		 // throw execpGen;
	}
}


void hdf5File::readCube(index_node_t index, float * cube)
{
	int3 coord 	= getMinBoxIndex2(index, levelCube, nLevels);
	int3 s 		= coord - cubeInc;
	int3 e 		= s + realCubeDim;

	hsize_t dim[3] = {abs(e.x-s.x),abs(e.y-s.y),abs(e.z-s.z)};

	// Set zeros's
	bzero(cube, dim[0]*dim[1]*dim[2]*sizeof(float));

	// The data required is completly outside of the dataset
	if (s.x >= (int)this->dims[0] || s.y >= (int)this->dims[1] || s.z >= (int)this->dims[2] || e.x < 0 || e.y < 0 || e.z < 0)
	{
		#if DEBUG
		std::cerr<<"Warning: reading cube outsite the volume "<<std::endl;
		std::cerr<<"Dimension valume "<<this->dims[0]<<" "<<this->dims[1]<<" "<<this->dims[2]<<std::endl;
		std::cerr<<"start "<<s.x<<" "<<s.y<<" "<<s.z<<std::endl;
		std::cerr<<"end "<<e.x<<" "<<e.y<<" "<<e.z<<std::endl;
		std::cerr<<"Dimension cube "<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<std::endl;
		#endif
		return;
	}

	herr_t	status;
	hid_t	memspace; 
	hsize_t offset_out[3] 	= {s.x < 0 ? abs(s.x) : 0, s.y < 0 ? abs(s.y) : 0, s.z < 0 ? abs(s.z) : 0};
	hsize_t offset[3] 	= {s.x < 0 ? 0 : s.x, s.y < 0 ? 0 : s.y, s.z < 0 ? 0 : s.z};
	hsize_t dimR[3]		= {e.x > (int)this->dims[0] ? this->dims[0] - offset[0] : e.x - offset[0],
				   e.y > (int)this->dims[1] ? this->dims[1] - offset[1] : e.y - offset[1],
				   e.z > (int)this->dims[2] ? this->dims[2] - offset[2] : e.z - offset[2]};

	#if DEBUG 
	std::cout<<"Dimension cube "<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<std::endl;
	std::cout<<"Dimension hyperSlab "<<dimR[0]<<" "<<dimR[1]<<" "<<dimR[2]<<std::endl;
	std::cout<<"Offset in "<<offset[0]<<" "<<offset[1]<<" "<<offset[2]<<std::endl;
	std::cout<<"Offset out "<<offset_out[0]<<" "<<offset_out[1]<<" "<<offset_out[2]<<std::endl;
	#endif


	/* 
	* Define hyperslab in the dataset. 
	*/
	if ((status = H5Sselect_hyperslab(spaceid, H5S_SELECT_SET, offset, NULL, dimR, NULL)) < 0)
	{
		std::cerr<<"hdf5: defining hyperslab in the dataset"<<std::endl;
		throw;
	}

	/*
	* Define the memory dataspace.
	*/
	if ((memspace = H5Screate_simple(3, dim, NULL)) < 0)
	//if ((memspace = H5Screate_simple(3, dimR, NULL)) < 0)
	{
		std::cerr<<"hdf5: defining the memory space"<<std::endl;
		throw;
	}


	/* 
	* Define memory hyperslab. 
	*/
	if ((status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL, dimR, NULL)) < 0)
	{
		std::cerr<<"hdf5: defining the memory hyperslab"<<std::endl;
		throw;
	}

	/*
	* Read data from hyperslab in the file into the hyperslab in 
	* memory and display.
	*/
	if ((status = H5Dread(dataset_id, H5T_IEEE_F32LE, memspace, spaceid, H5P_DEFAULT, cube)) < 0)
	{
		std::cerr<<"hdf5: reading data from hyperslab un the file"<<std::endl;
		throw;
	}


	if ((status = H5Sclose(memspace)) < 0)
	{
		std::cerr<<"hdf5: closing dataspace"<<std::endl;
		throw;
	}
}

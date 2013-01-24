# mivt
# Author: Carlos Duelo Serrano


PROJECT(mivt)
cmake_minimum_required(VERSION 2.8)

#Support for CUDA Integration
FIND_PACKAGE(CUDA)
IF (CUDA_FOUND)
	SET (OPT "-O0")
	SET (CUDACP "sm_20")
	SET (CFLAGS "${OPT} -g -Wall")
	SET (NFLAGS "-G ${OPT} -arch=${CUDACP} -Xcompiler '${CFLAGS}'")
    SET(CUDA_NVCC_FLAGS ${NFLAGS})
ELSE(CUDA_FOUND)
        MESSAGE("CUDA is not installed on this system.")
ENDIF(CUDA_FOUND)

#DEFINING DIRECTORIES
SET(OBJFOLDER obj)

#INCLUDES
FIND_LIBRARY(HDF5 NAMES hdf5 PATHS /usr/lib)
IF (HDF5)
MESSAGE(STATUS "hdf5 library found")
ELSE (HDF5)
MESSAGE(STATUS "Error: hdf5 library not found")
ENDIF (HDF5)

#SET(OBJSCR hdf5File)

#SET(EXESRC caca)


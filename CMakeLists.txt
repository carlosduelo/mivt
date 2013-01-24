# mivt
# Author: Carlos Duelo Serrano
project(mivt)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA)
if (CUDA_FOUND)
	set (OPT "-O0")
	set (CUDACP "sm_20")
	set (CFLAGS "${OPT} -g -Wall")
	set (NFLAGS "-G ${OPT} -arch=${CUDACP} -Xcompiler '${CFLAGS}'")
	set(CUDA_NVCC_FLAGS ${NFLAGS})
else(CUDA_FOUND)
	message(SEND_ERROR "CUDA is not installed on this system.")
endif(CUDA_FOUND)

#FIND LIBRARIES
find_library(HDF5 NAME hdf5 PATHS /usr/lib)
if (HDF5)
	message(STATUS "hdf5 library found")
else (HDF5)
	message(FATAL_ERROR "Error: hdf5 library not found")
endif (HDF5)

message(STATUS "All libraries found")


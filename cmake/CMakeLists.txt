# The name of our project is "mivt". CMakeLists files in this project can
# refer to the root source directory of the project as ${mivt_SOURCE_DIR} and 
# to the root binary directory of the project as ${mivt_BINARY_DIR}. 

cmake_minimum_required (VERSION 2.8)

project (mivt)

SET( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/packages)

#Find Libraries
find_package(CUDA 5.0 REQUIRED)
find_package(HDF5 1.8.4 REQUIRED)
find_package(Lunchbox 1.7.0 REQUIRED)
find_package(FreeImage REQUIRED)
IF(FREEIMAGE_FOUND)
	SET(HAVE_FREEIMAGE ${FREEIMAGE_FOUND} CACHE BOOL "Set to TRUE if FreeImage is found, FALSE otherwise")
	MESSAGE(STATUS "Looking for FreeImage - found")
ELSE(FREEIMAGE_FOUND)
	MESSAGE(STATUS "Looking for FreeImage - not found")
ENDIF(FREEIMAGE_FOUND)
find_package(OpenGL)


#Set flags
list(APPEND CUDA_NVCC_FLAGS -arch=sm_20)
list(APPEND CUDA_NVCC_FLAGS -O3)
list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-O3)
list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wall)

set(SRC ${mivt_SOURCE_DIR}/../src/)
set(INC  ${mivt_SOURCE_DIR}/../inc/)

# ADD includes
cuda_include_directories(${INC})

# Tarject testFileManager
cuda_compile(testFileManagerOBJ ${SRC}/fileUtil.cu  ${SRC}/hdf5File.cu)
cuda_add_executable(testFileManager ${SRC}/testFileManager.cu ${testFileManagerOBJ})
target_link_libraries(testFileManager hdf5)

# Tarject testThreadMaster 
cuda_compile(testThreadMasterOBJ ${SRC}/cache_GPU_File.cu ${SRC}/cache_CPU_File.cu ${SRC}/cache_GPU_CPU_File.cu  ${SRC}/Camera.cu  ${SRC}/fileUtil.cu  ${SRC}/hdf5File.cu  ${SRC}/lruCache.cu  ${SRC}/octree_completeGPU.cu  ${SRC}/octreeContainer.cu  ${SRC}/rayCaster.cu  ${SRC}/threadMaster.cu ${SRC}/threadWorker.cu)
cuda_add_executable(testThreadMaster ${SRC}/testThreads.cu ${testThreadMasterOBJ})
target_link_libraries(testThreadMaster hdf5)
target_link_libraries(testThreadMaster Lunchbox)
target_link_libraries(testThreadMaster freeimage)

# Tarject testCache
cuda_compile(testCacheOBJ ${SRC}/cache_GPU_File.cu ${SRC}/cache_CPU_File.cu ${SRC}/cache_GPU_CPU_File.cu  ${SRC}/fileUtil.cu  ${SRC}/hdf5File.cu  ${SRC}/lruCache.cu)
cuda_add_executable(testCache ${SRC}/testCache.cu ${testCacheOBJ})
target_link_libraries(testCache hdf5)
target_link_libraries(testCache Lunchbox)

# Tarject testThreadMaster 
cuda_compile(mivtOBJ ${SRC}/cache_GPU_File.cu  ${SRC}/cache_CPU_File.cu ${SRC}/cache_GPU_CPU_File.cu ${SRC}/Camera.cu  ${SRC}/fileUtil.cu  ${SRC}/hdf5File.cu  ${SRC}/lruCache.cu  ${SRC}/octree_completeGPU.cu  ${SRC}/octreeContainer.cu  ${SRC}/rayCaster.cu  ${SRC}/threadMaster.cu ${SRC}/threadWorker.cu)
cuda_add_executable(mivt ${SRC}/mivt.cu ${mivtOBJ})
target_link_libraries(mivt hdf5)
target_link_libraries(mivt Lunchbox)
target_link_libraries(mivt GL)
target_link_libraries(mivt glut)

cuda_build_clean_target()

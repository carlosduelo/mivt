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


set(SRC ${mivt_SOURCE_DIR}/../src/)
set(INC  ${mivt_SOURCE_DIR}/../inc/)

# ADD includes
cuda_include_directories(${INC})

# Tarject testFileManager
cuda_compile(testFileManager ${SRC}/fileUtil.cu  ${SRC}/hdf5File.cu)
cuda_add_executable(tesFileManager ${SRC}/testFileManager.cu ${testFileManager})
target_link_libraries(tesFileManager hdf5)

# Tarject testThreadMaster 
cuda_compile(testThreadMaster ${SRC}/cache_GPU_File.cu  ${SRC}/Camera.cu  ${SRC}/fileUtil.cu  ${SRC}/hdf5File.cu  ${SRC}/lruCache.cu  ${SRC}/octree_completeGPU.cu  ${SRC}/octreeContainer.cu  ${SRC}/rayCaster.cu  ${SRC}/threadMaster.cu ${SRC}/threadWorker.cu)
cuda_add_executable(testThreadMaster ${SRC}/testThreads.cu ${testThreadMaster})
target_link_libraries(testThreadMaster hdf5)
target_link_libraries(testThreadMaster Lunchbox)
target_link_libraries(testThreadMaster freeimage)

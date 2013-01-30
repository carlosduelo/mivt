NVCC=nvcc
OPT=-O0
CUDACP=sm_20
CFLAGS=$(OPT) -g -Wall
NFLAGS=-G $(OPT) -arch=$(CUDACP) -Xcompiler '$(CFLAGS)'
INCLUDE=-Iinc/
LIBRARY=-lhdf5 

all: Objects testPrograms 

Objects: obj/fileUtil.o obj/hdf5File.o obj/lruCache.o obj/cache_GPU_File.o obj/octreeContainer.o obj/octree_completeGPU.o obj/rayCaster.o

obj/fileUtil.o: inc/FileManager.hpp src/fileUtil.cu
	$(NVCC) -c $(NFLAGS) $(INCLUDE) src/fileUtil.cu -o obj/fileUtil.o

obj/hdf5File.o: inc/FileManager.hpp src/hdf5File.cu
	$(NVCC) -c $(NFLAGS) $(INCLUDE) src/hdf5File.cu -o obj/hdf5File.o

obj/lruCache.o: inc/lruCache.hpp src/lruCache.cu
	$(NVCC) -c $(NFLAGS) $(INCLUDE) src/lruCache.cu -o obj/lruCache.o

obj/cache_GPU_File.o: inc/lruCache.hpp src/cache_GPU_File.cu
	$(NVCC) -c $(NFLAGS) $(INCLUDE) src/cache_GPU_File.cu -o obj/cache_GPU_File.o

obj/octreeContainer.o: inc/Octree.hpp src/octreeContainer.cu
	$(NVCC) -c $(NFLAGS) $(INCLUDE) src/octreeContainer.cu -o obj/octreeContainer.o

obj/octree_completeGPU.o: inc/Octree.hpp src/octree_completeGPU.cu
	$(NVCC) -c $(NFLAGS) $(INCLUDE) src/octree_completeGPU.cu -o obj/octree_completeGPU.o

obj/rayCaster.o: inc/rayCaster.hpp src/rayCaster.cu
	$(NVCC) -c $(NFLAGS) $(INCLUDE) src/rayCaster.cu -o obj/rayCaster.o

testPrograms: bin/testFileManager

bin/testFileManager: src/testFileManager.cu
	$(NVCC) $(NFLAGS) $(INCLUDE) obj/fileUtil.o obj/hdf5File.o src/testFileManager.cu -o bin/testFileManager $(LIBRARY)

clean:
	-rm bin/* obj/* 

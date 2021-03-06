NVCC=nvcc
OPT=-O0
CUDACP=sm_20
CFLAGS=$(OPT) -g -Wall
NFLAGS=-G $(OPT) -arch=$(CUDACP) -Xcompiler '$(CFLAGS)'
INCLUDE=-Iinc/
LIBRARY=-Llib/ -lLunchbox -lhdf5 -lm -lfreeimage

all: Objects testPrograms 

Objects: obj/fileUtil.o obj/hdf5File.o obj/lruCache.o obj/cache_GPU_File.o obj/octreeContainer.o obj/octree_completeGPU.o obj/rayCaster.o obj/camera.o obj/threadWorker.o obj/threadMaster.o

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

obj/camera.o: inc/Camera.hpp src/Camera.cu
	$(NVCC) -c $(NFLAGS) $(INCLUDE) src/Camera.cu -o obj/Camera.o

obj/threadWorker.o: inc/threadWorker.hpp src/threadWorker.cu
	$(NVCC) -c $(NFLAGS) $(INCLUDE) src/threadWorker.cu -o obj/threadWorker.o

obj/threadMaster.o: inc/threadMaster.hpp src/threadMaster.cu
	$(NVCC) -c $(NFLAGS) $(INCLUDE) src/threadMaster.cu -o obj/threadMaster.o

testPrograms: bin/testFileManager bin/testThreads

bin/testFileManager: src/testFileManager.cu
	$(NVCC) $(NFLAGS) $(INCLUDE) obj/fileUtil.o obj/hdf5File.o src/testFileManager.cu -o bin/testFileManager $(LIBRARY)

bin/testThreads: src/testThreads.cu
	$(NVCC) $(NFLAGS) $(INCLUDE) obj/fileUtil.o obj/hdf5File.o obj/lruCache.o obj/cache_GPU_File.o obj/octreeContainer.o obj/octree_completeGPU.o  obj/rayCaster.o obj/Camera.o  obj/threadWorker.o obj/threadMaster.o src/testThreads.cu -o bin/testThreads $(LIBRARY)

clean:
	-rm bin/* obj/* 

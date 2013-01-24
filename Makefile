NVCC=nvcc
OPT=-O0
CUDACP=sm_20
CFLAGS=$(OPT) -g -Wall
NFLAGS=-G $(OPT) -arch=$(CUDACP) -Xcompiler '$(CFLAGS)'
INCLUDE=-Iinc/
LIBRARY=-lhdf5 

all: Objects testPrograms 

Objects: obj/fileUtil.o obj/hdf5File.o obj/lruCache.o


obj/fileUtil.o:
	$(NVCC) -c $(NFLAGS) $(INCLUDE) src/fileUtil.cu -o obj/fileUtil.o

obj/hdf5File.o:
	$(NVCC) -c $(NFLAGS) $(INCLUDE) src/hdf5File.cu -o obj/hdf5File.o

obj/lruCache.o:
	$(NVCC) -c $(NFLAGS) $(INCLUDE) src/lruCache.cu -o obj/lruCache.o


testPrograms: bin/testFileManager

bin/testFileManager: src/testFileManager.cu
	$(NVCC) $(NFLAGS) $(INCLUDE) obj/fileUtil.o obj/hdf5File.o src/testFileManager.cu -o bin/testFileManager $(LIBRARY)

clean:
	-rm bin/* obj/* 

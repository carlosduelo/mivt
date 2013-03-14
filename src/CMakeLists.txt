# The name of our project is "mivt". CMakeLists files in this project can
# refer to the root source directory of the project as ${mivt_SOURCE_DIR} and 
# to the root binary directory of the project as ${mivt_BINARY_DIR}. 

set(MIVT_SOURCE	config.cpp
		error.cpp
		initParams.cpp
		mivt.cpp
		channel.cpp
		node.cpp
		pipe.cpp
		frameData.cpp
		hdf5File.cpp
)

set(LIBS_NEEDED ${EQUALIZER_LIBRARIES} GL ${Boost_LIBRARIES} ${HDF5_LIBRARIES}) 

add_executable(mivt ${MIVT_SOURCE} main.cpp) 
target_link_libraries(mivt ${LIBS_NEEDED}) 

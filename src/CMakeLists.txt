# The name of our project is "mivt". CMakeLists files in this project can
# refer to the root source directory of the project as ${mivt_SOURCE_DIR} and 
# to the root binary directory of the project as ${mivt_BINARY_DIR}. 

#add_executable(hello eqHello.cpp) 
#target_link_libraries(hello ${EQUALIZER_LIBRARIES}) 

set(MIVT_SOURCE	config.cpp
		error.cpp
		initParams.cpp
		mivt.cpp
		channel.cpp
)


add_executable(mivt ${MIVT_SOURCE} main.cpp) 
target_link_libraries(mivt ${EQUALIZER_LIBRARIES} GL) 

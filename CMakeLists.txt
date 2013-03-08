# The name of our project is "mivt". CMakeLists files in this project can
# refer to the root source directory of the project as ${mivt_SOURCE_DIR} and 
# to the root binary directory of the project as ${mivt_BINARY_DIR}. 

cmake_minimum_required (VERSION 2.8)

project (mivt)

# Equalizer
find_package(Equalizer 1.5.0 REQUIRED)
#if (EQUALIZER_FOUND)
include_directories(${EQUALIZER_INCLUDE_DIRS})


# MIVT
include_directories(${mivt_SOURCE_DIR}/inc)
add_subdirectory(${mivt_SOURCE_DIR}/src)

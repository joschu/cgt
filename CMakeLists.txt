cmake_minimum_required(VERSION 2.6)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.\nBut first clean the build files that just got created in the source directory:\nrm -rf CMakeFiles CMakeCache.txt")
endif()

project(cgt)

# http://cmake.3232098.n2.nabble.com/Default-value-for-CMAKE-BUILD-TYPE-td7550756.html
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    CACHE PATH "Output directory for static libraries.")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    CACHE PATH "Output directory for shared libraries.")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CACHE PATH "Output directory for executables and DLL's.")

# set(CMAKE_C_STANDARD 99)
list( APPEND CMAKE_C_FLAGS "-std=c99")
list( APPEND CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/4build/cmake)

include(${CMAKE_SOURCE_DIR}/4build/cmake/UseCython.cmake)

find_package(PythonLibs REQUIRED)
find_package(Numpy REQUIRED)

if (APPLE)
  set(CMAKE_MACOSX_RPATH 0)
endif()


option(CGT_ENABLE_CUDA OFF "Enable CUDA support")
option(CGT_ENABLE_CUDNN ON "Enable CUDNN support")
option(CGT_DEVEL_MODE OFF "Build with more compiler errors and warnings")

message(
    "\n"
    "*****************************\n"
    "OPTIONS: \n"
    "   CGT_ENABLE_CUDA:    ${CGT_ENABLE_CUDA}\n"
    # "   CGT_DEVEL_MODE:     ${CGT_DEVEL_MODE}\n"
    "   CMAKE_BUILD_TYPE:   ${CMAKE_BUILD_TYPE}\n"
    "*****************************\n"
    )


if (CGT_DEVEL_MODE)
  set(CMAKE_C_FLAGS "-Wall -Wextra -Wno-unused-function -Wno-sign-compare")
endif(CGT_DEVEL_MODE)

set(CGT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
message("include dir ${CGT_INCLUDE_DIR}")

add_library(cgt SHARED src/cgt_common.cpp include/cgt_common.h)
if (CGT_ENABLE_CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DCGT_ENABLE_CUDA)  
endif()

if (CGT_ENABLE_CUDNN)
  set(CUDNN_ROOT "/Users/joschu/Src/cudnn-6.5-osx-v2")
endif()

target_link_libraries(cgt ${CUDA_CUDART_LIBRARY})
target_include_directories(cgt PRIVATE ${CUDA_INCLUDE_DIRS} ${CGT_INCLUDE_DIR})


cython_add_module(cycgt2 ${CMAKE_SOURCE_DIR}/src/cycgt2.pyx 
  ${CMAKE_SOURCE_DIR}/include/execution.h
  ${CMAKE_SOURCE_DIR}/src/execution.cpp
)
set_source_files_properties(${CMAKE_SOURCE_DIR}/src/cycgt2.pyx PROPERTIES CYTHON_IS_CXX TRUE)
target_include_directories(cycgt2 PRIVATE ${NUMPY_INCLUDE_DIR} ${CGT_INCLUDE_DIR})
target_link_libraries(cycgt2 cgt)


# add_custom_command(
#   COMMAND ./do_autogen.py ${CMAKE_BINARY_DIR}
#   OUTPUT ${CMAKE_BINARY_DIR}/cgtcorefuns.c ${CMAKE_BINARY_DIR}/cgtcorefunscuda.cu
#   WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#   # DEPENDS cgt.py
# )
# add_custom_target(gencfiles ALL DEPENDS cgtcorefuns.c)

set(OPENBLAS_LIBRARY ${CMAKE_BINARY_DIR}/OpenBLAS/libopenblas.a)

add_custom_command(
  COMMAND 4build/download_and_build_openblas.py ${CMAKE_BINARY_DIR}/OpenBLAS
  OUTPUT ${OPENBLAS_LIBRARY}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_custom_target(openblas ALL DEPENDS ${OPENBLAS_LIBRARY})

# add_library(cgtcorefuns SHARED cgtcorefuns.c)
# add_dependencies(cgtcorefuns openblas) # WHY IS THIS NECESSARY?
# target_link_libraries(cgtcorefuns ${OPENBLAS_LIBRARY})
# target_include_directories(cgtcorefuns PRIVATE ${CMAKE_BINARY_DIR}/OpenBLAS ${CGT_INCLUDE_DIR})
# set_target_properties(cgtcorefuns PROPERTIES SUFFIX .so)

# if (CGT_ENABLE_CUDA)
#   # http://cuda-insight-toolkit.googlecode.com/svn-history/r46/trunk/Examples/Architecture1/src/CMakeLists.txt
#   find_package(CUDA REQUIRED)
#   cuda_include_directories(${CUDA_INCLUDE_DIRS} ${CGT_INCLUDE_DIR})
#   cuda_add_library(cgtcorefunscuda SHARED ${CMAKE_BINARY_DIR}/cgtcorefunscuda.cu)
#   target_link_libraries(cgtcorefunscuda ${CUDA_LIBRARIES})
#   set_target_properties(cgtcorefunscuda PROPERTIES SUFFIX .so)
# endif()

# cuda_add_library(cudadummy SHARED dummy.cu)

message("link flags ${CMAKE_SHARED_LINKER_FLAGS}")

get_property(CGT_LIBRARY_PATH TARGET cgt PROPERTY LOCATION)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/build_info.txt 
  "CUDA_ROOT := ${CUDA_TOOLKIT_ROOT_DIR}\n"
  "CGT_INCLUDE_DIR := ${CGT_INCLUDE_DIR}\n"
  "C_COMPILER := ${CMAKE_C_COMPILER}\n"
  "CGT_LIBRARY_PATH := ${CGT_LIBRARY_PATH}\n"
  "CUDA_LIBRARIES := ${CUDA_LIBRARIES}\n"
  "CGT_ENABLE_CUDA := ${CGT_ENABLE_CUDA}\n"
  "CGT_ENABLE_CUDNN := ${CGT_ENABLE_CUDNN}\n"
  "CUDNN_ROOT := ${CUDNN_ROOT}\n"
)

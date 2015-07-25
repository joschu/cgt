

if (NUMPY_INCLUDE_DIR)
  return()
endif()


file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/determineNumpyPath.py "try: import numpy; print numpy.get_include()\nexcept: pass\n")
exec_program("${PYTHON_EXECUTABLE}"
             ARGS "\"${CMAKE_CURRENT_BINARY_DIR}/determineNumpyPath.py\""
             OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
             CACHE INTERNAL
)

message("NUMPY_INCLUDE_DIR: ${NUMPY_INCLUDE_DIR}")

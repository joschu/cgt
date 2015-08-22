################################################################################################
# Short command for cuDNN detection. Believe it soon will be a part of CUDA toolkit distribution.
# That's why not FindcuDNN.cmake file, but just the macro
# Usage:
#   detect_cuDNN()
function(detect_cuDNN)
  set(CUDNN_ROOT "" CACHE PATH "CUDNN root folder")

  find_path(CUDNN_INCLUDE cudnn.h
            PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDA_TOOLKIT_INCLUDE}
            DOC "Path to cuDNN include directory." )

  get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)
  find_library(CUDNN_LIBRARY NAMES libcudnn.so # libcudnn_static.a
                             PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE} ${__libpath_hist}
                             DOC "Path to cuDNN library.")

  if(CUDNN_INCLUDE AND CUDNN_LIBRARY)
    set(HAVE_CUDNN  TRUE PARENT_SCOPE)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)

    mark_as_advanced(CUDNN_INCLUDE CUDNN_LIBRARY CUDNN_ROOT)
    message(STATUS "Found cuDNN (include: ${CUDNN_INCLUDE}, library: ${CUDNN_LIBRARY})")
  endif()
endfunction()



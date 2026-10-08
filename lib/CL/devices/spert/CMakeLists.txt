#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2026 SpacemiT
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================

add_pocl_device_library(pocl-devices-spert spert.h spert.c spert_scheduler.h spert_scheduler.cc)
target_include_directories(pocl-devices-spert PRIVATE "${SPERT_INCLUDE_DIR}")

if(ENABLE_LOADABLE_DRIVERS)
  target_link_libraries(pocl-devices-spert PRIVATE pocl-devices-pthread ${SPERT_LIBRARY})
endif()

# make sure libspert.so is found at runtime without LD_LIBRARY_PATH
get_filename_component(SPERT_LIBRARY_DIR "${SPERT_LIBRARY}" DIRECTORY)
set_target_properties(pocl-devices-spert PROPERTIES
  BUILD_RPATH "${SPERT_LIBRARY_DIR}"
  INSTALL_RPATH "${SPERT_LIBRARY_DIR}")

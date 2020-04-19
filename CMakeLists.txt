cmake_minimum_required(VERSION 3.5)
project(optimizers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_definitions(-O3 -mavx2 -mfma -DNDEBUG -Wfatal-errors)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=1")

add_executable(main main.cc)

# Add boost
find_package(Boost 1.72 REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
target_link_libraries(main LINK_PUBLIC ${Boost_LIBRARIES})

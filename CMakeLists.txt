cmake_minimum_required(VERSION 3.19)
project(cpp_vector_norm)
set(CMAKE_CXX_STANDARD 14)
add_executable(cpp_vector_norm main.cpp norms_cpu.cpp)
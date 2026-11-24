cmake_minimum_required(VERSION 3.16)
project(Ausmera_AR01 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ausmera src/main.cpp)
target_compile_options(ausmera PRIVATE -O2 -Wall -Wextra)

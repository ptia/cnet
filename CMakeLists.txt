cmake_minimum_required(VERSION 3.13)
project(cnet)

add_compile_options(-Wall -Wextra -pedantic)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

file(GLOB main_src *.c)
add_executable(run_test ${main_src} test/main.c)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

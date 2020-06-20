cmake_minimum_required(VERSION 3.13)
project(cnet)

file(GLOB main_src *.c)
file(GLOB test_src test/*.c)

add_executable(run_test ${main_src} ${test_src})

add_compile_options(-Wall -Wextra -pedantic -Werror)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

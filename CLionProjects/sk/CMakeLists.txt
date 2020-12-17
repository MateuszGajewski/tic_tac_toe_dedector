cmake_minimum_required(VERSION 3.17)
project(sk)

set(CMAKE_CXX_STANDARD 14)

add_executable(sk main.cpp communication.cpp communication.h user.cpp user.h server.cpp server.h)
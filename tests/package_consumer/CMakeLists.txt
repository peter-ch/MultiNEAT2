cmake_minimum_required(VERSION 3.15)

project(MultiNEATPackageConsumer LANGUAGES CXX)

find_package(MultiNEAT CONFIG REQUIRED)

add_executable(multineat_package_consumer main.cpp)
target_link_libraries(multineat_package_consumer PRIVATE MultiNEAT::multineat)

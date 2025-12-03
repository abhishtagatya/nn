cmake_minimum_required(VERSION 3.29)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 20)

add_executable(NeuralNetwork main.cpp
        Activation.hpp
        DataLoader.hpp
        Layer.hpp
        Loss.hpp
        Matrix.hpp
        Network.hpp
        Time.hpp
)

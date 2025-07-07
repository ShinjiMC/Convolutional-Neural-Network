cmake_minimum_required(VERSION 3.10)
project(perceptron)

set(CMAKE_CXX_STANDARD 17)

# Agregar flags de optimización y OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")

include_directories(include)

add_executable(cnn_train
     main_train_cnn.cpp
     src/layer.cpp
     src/mlp.cpp
     src/neuron.cpp
     src/dataset.cpp
     src/config.cpp
     src/regularizer.cpp
     src/conv2d.cpp
     src/flatten.cpp
     src/pooling.cpp
)
cmake_minimum_required(VERSION 3.10)
project(perceptron)

set(CMAKE_CXX_STANDARD 17)

# Agregar flags de optimización y OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")

include_directories(include)

# Ejecutable de entrenamiento
add_executable(mlp_train
     main_train.cpp
     src/layer.cpp
     src/mlp.cpp
     src/neuron.cpp
     src/dataset.cpp
     src/config.cpp
     src/regularizer.cpp
)

# Ejecutable de prueba
add_executable(mlp_test
     main_test.cpp
     src/layer.cpp
     src/mlp.cpp
     src/neuron.cpp
     src/dataset.cpp
     src/config.cpp
     src/regularizer.cpp
)

# Ejecutable de conv
add_executable(conv
     main_cnv.cpp
     src/conv2d.cpp
     src/flatten.cpp
     src/pooling.cpp
)
mpic++ \
    -o build/main src/main.cpp \
    -lopenblas \
    -std=c++23 \
    -fopenmp \
    -I extern/spdlog/include \
    -I extern/cereal/include \
    -I /usr/include \
    -L/usr/lib/x86_64-linux-gnu -llapacke -llapack -lblas
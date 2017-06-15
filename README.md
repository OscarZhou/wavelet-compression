# wavelet-compression
a seminar for wavelet compression technology


# How to run the program

### DWT16x16

command: g++ DWT_16X16.C -O dwt
        ./dwt


### Complex DWT algorithm

command: g++ -std=c++0x -o main -O3 main.cpp `pkg-config --libs --cflags opencv`
      ./main

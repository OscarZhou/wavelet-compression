# wavelet-compression
a seminar for wavelet compression technology


# How to run the program

### DWT16x16
```
 * Compile with:
 * g++ -o DWT DWT_16x16.c 
 * Execute static code: for example
 * ./DWT
```

### Complex DWT algorithm
```
 * Compile with:
 * g++ -std=c++0x -o waveletcompression -O3 waveletcompression.cpp `pkg-config --libs --cflags opencv`
 * Execute static code: for example
 * ./waveletcompression ~/Downloads/lena.jpg
```

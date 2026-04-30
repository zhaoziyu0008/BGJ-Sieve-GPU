OBJ = $(patsubst %.cpp,%.o, $(wildcard *.cpp))
CUOBJ = $(patsubst %.cu,%.o, $(wildcard *.cu))
HEADER = $(wildcard ../include/*.h*)
CUARCH = -gencode arch=compute_89,code=sm_89
SURP = -diag-suppress 549,550,815,20281

CXX = g++
CFLAGS = -O3 -g -fopenmp -march=native -pthread
INC_DIR =

NVCC = nvcc
NVCCFLAGS = -Xcompiler "$(CFLAGS)" $(CUARCH) -O3 $(SURP)

all: libllib.a

libllib.a: $(OBJ) $(CUOBJ)
	ar rcs libllib.a $(OBJ) $(CUOBJ)

%.o: %.cpp $(HEADER)
	$(CXX) $(CFLAGS) $< $(INC_DIR) -c

%.o: %.cu $(HEADER)
	$(NVCC) -ccbin $(CXX) $(NVCCFLAGS) $< $(INC_DIR) -c

.PHONY: clean

clean:
	-rm -f $(OBJ) $(CUOBJ) libllib.a


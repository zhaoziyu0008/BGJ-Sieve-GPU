CUBIN = $(patsubst bin_%.cu,%, $(wildcard *.cu))
CXX = g++
CFLAGS = -O3 -g -fopenmp -march=native -pthread -Wno-unused-result -Wno-format-truncation
LIBS = -L../src -lllib -lntl -lgmp -lm
INC_DIR =

SURP = -diag-suppress 549,550,815,1650 -Wno-deprecated-gpu-targets

NVCC = nvcc

all: $(CUBIN)

.PHONY: clean

clean:
  -rm -f $(CUBIN)

%: bin_%.cu ../src/libllib.a
  $(NVCC) -ccbin $(CXX) -Xcompiler "$(CFLAGS)" -O3 $< $(INC_DIR) $(LIBS) $(SURP) -lcudart -lnuma -o $*

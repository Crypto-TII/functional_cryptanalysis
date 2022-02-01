CXX      = g++
CXXFLAGS = -Xpreprocessor -fopenmp -std=c++11
OPTFLAGS = -Ofast -march=native -mtune=native
LDFLAGS  = -lomp

all:
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) differential.cpp -o differential $(LDFLAGS)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) functional.cpp -o functional $(LDFLAGS)

clean:
	rm -f differential functional

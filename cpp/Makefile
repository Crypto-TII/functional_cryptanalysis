CXX      = g++
CXXFLAGS = -Xpreprocessor -fopenmp -std=c++11
OPTFLAGS = -Ofast -march=native -mtune=native
LDFLAGS  = -lomp

TARGETS  = differential functional

$(TARGETS):
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $@.cpp $(LDFLAGS) -o $@

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

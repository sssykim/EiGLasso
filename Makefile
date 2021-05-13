MKLROOT=/home/junhoy/intel/mkl
IDIR=$(MKLROOT)/include
LDIR=$(MKLROOT)/lib/intel64

CXX=g++
CXXFLAGS= -std=c++11 -O3 -Wall -m64 -I$(IDIR)

LIBS=-Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

eiglasso: eiglasso.cpp
	$(CXX) $(CXXFLAGS) -o $@ eiglasso.cpp -L$(LDIR) $(LIBS)

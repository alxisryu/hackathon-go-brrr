CXX = g++
CXXFLAGS = -std=c++11 -Wall -I.
LDFLAGS = -lstdc++fs

SRC = src/main.cpp src/loader.cpp src/neural_network.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = speed_cpu

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(OBJ) $(TARGET)

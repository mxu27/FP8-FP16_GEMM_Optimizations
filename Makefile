NVCC        := nvcc
NVCCFLAGS   := -std=c++14 -O2 -arch=native

GPUTK_DIR   := ./libgputk
LIBS        := -L$(GPUTK_DIR) -lgputk -Xlinker -rpath -Xlinker $(shell pwd)/libgputk
INCLUDES    := -I$(GPUTK_DIR)
LIBS        := -L$(GPUTK_DIR)/lib -lgputk
LIBS_FP8    := $(LIBS) -lcublasLt -lcublas

SRC_DIR     := src
BUILD_DIR   := build

SRCS        := $(wildcard $(SRC_DIR)/*.cu)
TARGETS     := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%,$(SRCS))

.PHONY: all clean

all: $(BUILD_DIR) $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# These binaries use cuBLASLt for the FP8 pipeline
$(BUILD_DIR)/compareMatrixMultiplication: $(SRC_DIR)/compareMatrixMultiplication.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LIBS_FP8)

$(BUILD_DIR)/fp8Quantization: $(SRC_DIR)/fp8Quantization.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

$(BUILD_DIR)/%: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

clean:
	rm -rf $(BUILD_DIR)

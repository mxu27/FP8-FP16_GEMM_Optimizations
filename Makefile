NVCC        := nvcc
NVCCFLAGS   := -std=c++14 -O2 -arch=native

GPUTK_DIR   ?= /usr/local
INCLUDES    := -I$(GPUTK_DIR)/include
LIBS        := -L$(GPUTK_DIR)/lib -lgputk

SRC_DIR     := src
BUILD_DIR   := build

SRCS        := $(wildcard $(SRC_DIR)/*.cu)
TARGETS     := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%,$(SRCS))

.PHONY: all clean

all: $(BUILD_DIR) $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

clean:
	rm -rf $(BUILD_DIR)

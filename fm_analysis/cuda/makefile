##########################################################

### GPU architecture
SM_ARCH=86

## NVCC COMPILER OPTIONS ##
NVCC=nvcc
NVCC_CUBIN_FLAGS= -cubin -arch=sm_${SM_ARCH}
NVCC_PTX_FLAGS= -ptx

##########################################################

## Project file structure ##
# Source file directory:
SRC_DIR = kernel

# Output file directories:
OUTPUT_CUBIN_DIR = cubin
OUTPUT_PTX_DIR = ptx

# Source files:
SRC_FILES = $(wildcard $(SRC_DIR)/*.cu)

# Output files:
CUBIN_FILES = $(patsubst $(SRC_DIR)/%.cu, $(OUTPUT_CUBIN_DIR)/%.cubin, $(SRC_FILES))
PTX_FILES = $(patsubst $(SRC_DIR)/%.cu, $(OUTPUT_PTX_DIR)/%.ptx, $(SRC_FILES))

##########################################################

# Default target: Compile both CUBIN and PTX
all: $(CUBIN_FILES) $(PTX_FILES)

# Rule to generate .cubin files
$(OUTPUT_CUBIN_DIR)/%.cubin: $(SRC_DIR)/%.cu
	@mkdir -p $(OUTPUT_CUBIN_DIR)
	$(NVCC) $(NVCC_CUBIN_FLAGS) -o $@ $<

# Rule to generate .ptx files
$(OUTPUT_PTX_DIR)/%.ptx: $(SRC_DIR)/%.cu
	@mkdir -p $(OUTPUT_PTX_DIR)
	$(NVCC) $(NVCC_PTX_FLAGS) -o $@ $<

# Clean up all generated files
clean:
	rm -rf $(OUTPUT_CUBIN_DIR) $(OUTPUT_PTX_DIR)
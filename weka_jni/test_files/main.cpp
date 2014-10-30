/*
 * main.cpp
 *
 *  Created on: 20/10/2014
 *      Author: bsp
 */


#ifdef BUILD_TESTS

#include <stdlib.h>
#include <stdio.h>
#include "context.hpp"
#include "hsa.h"
#include "elf_utils.h"
#include <string.h>
/*
 * Define required BRIG data structures.
 */

typedef uint32_t BrigCodeOffset32_t;

typedef uint32_t BrigDataOffset32_t;

typedef uint16_t BrigKinds16_t;

typedef uint8_t BrigLinkage8_t;

typedef uint8_t BrigExecutableModifier8_t;

typedef BrigDataOffset32_t BrigDataOffsetString32_t;

enum BrigKinds {
    BRIG_KIND_NONE = 0x0000,
    BRIG_KIND_DIRECTIVE_BEGIN = 0x1000,
    BRIG_KIND_DIRECTIVE_KERNEL = 0x1008,
};

typedef struct BrigBase BrigBase;
struct BrigBase {
    uint16_t byteCount;
    BrigKinds16_t kind;
};

typedef struct BrigExecutableModifier BrigExecutableModifier;
struct BrigExecutableModifier {
    BrigExecutableModifier8_t allBits;
};

typedef struct BrigDirectiveExecutable BrigDirectiveExecutable;
struct BrigDirectiveExecutable {
    uint16_t byteCount;
    BrigKinds16_t kind;
    BrigDataOffsetString32_t name;
    uint16_t outArgCount;
    uint16_t inArgCount;
    BrigCodeOffset32_t firstInArg;
    BrigCodeOffset32_t firstCodeBlockEntry;
    BrigCodeOffset32_t nextModuleEntry;
    uint32_t codeBlockEntryCount;
    BrigExecutableModifier modifier;
    BrigLinkage8_t linkage;
    uint16_t reserved;
};

typedef struct BrigData BrigData;
struct BrigData {
    uint32_t byteCount;
    uint8_t bytes[1];
};

#define check(msg, status) \
if (status != HSA_STATUS_SUCCESS) { \
    printf("%s failed.\n", #msg); \
    exit(1); \
} else { \
   printf("%s succeeded.\n", #msg); \
}
/*
 * Finds the specified symbols offset in the specified brig_module.
 * If the symbol is found the function returns HSA_STATUS_SUCCESS,
 * otherwise it returns HSA_STATUS_ERROR.
 */
hsa_status_t find_symbol_offset(hsa_ext_brig_module_t* brig_module,
    const char* symbol_name,
    hsa_ext_brig_code_section_offset32_t* offset) {

    /*
     * Get the data section
     */
    hsa_ext_brig_section_header_t* data_section_header =
                brig_module->section[HSA_EXT_BRIG_SECTION_DATA];
    /*
     * Get the code section
     */
    hsa_ext_brig_section_header_t* code_section_header =
             brig_module->section[HSA_EXT_BRIG_SECTION_CODE];

    /*
     * First entry into the BRIG code section
     */
    BrigCodeOffset32_t code_offset = code_section_header->header_byte_count;
    BrigBase* code_entry = (BrigBase*) ((char*)code_section_header + code_offset);
    while (code_offset != code_section_header->byte_count) {
        if (code_entry->kind == BRIG_KIND_DIRECTIVE_KERNEL) {
            /*
             * Now find the data in the data section
             */
            BrigDirectiveExecutable* directive_kernel = (BrigDirectiveExecutable*) (code_entry);
            BrigDataOffsetString32_t data_name_offset = directive_kernel->name;
            BrigData* data_entry = (BrigData*)((char*) data_section_header + data_name_offset);
            if (!strncmp(symbol_name, (char*) data_entry->bytes, strlen(symbol_name))) {
                *offset = code_offset;
                return HSA_STATUS_SUCCESS;
            }
        }
        code_offset += code_entry->byteCount;
        code_entry = (BrigBase*) ((char*)code_section_header + code_offset);
    }
    return HSA_STATUS_ERROR;
}

/*
 * Determines if the given agent is of type HSA_DEVICE_TYPE_GPU
 * and sets the value of data to the agent handle if it is.
 */
static hsa_status_t find_gpu(hsa_agent_t agent, void *data) {
    if (data == NULL) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    hsa_device_type_t device_type;
    hsa_status_t stat =
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
    if (stat != HSA_STATUS_SUCCESS) {
        return stat;
    }
    if (device_type == HSA_DEVICE_TYPE_GPU) {
        *((hsa_agent_t *)data) = agent;
    }
    return HSA_STATUS_SUCCESS;
}

/*
 * Determines if a memory region can be used for kernarg
 * allocations.
 */
static hsa_status_t get_kernarg(hsa_region_t region, void* data) {
    hsa_region_flag_t flags;
    hsa_region_get_info(region, HSA_REGION_INFO_FLAGS, &flags);
    if (flags & HSA_REGION_FLAG_KERNARG) {
        hsa_region_t* ret = (hsa_region_t*) data;
        *ret = region;
    }
    return HSA_STATUS_SUCCESS;
}

int main()
{
	printf("Unit test profile\n");

	hsa_ext_brig_code_section_offset32_t section_offset;
	hsa_ext_brig_module_t* brig_module;
	status_t status = create_brig_module_from_brig_file("/home/bsp/hsa_jni/kernels/max_value.brig",&brig_module);
	hsa_status_t hsa_status = find_symbol_offset(brig_module,"&__OpenCL_run_kernel" ,&section_offset );

  hsa_status_t err;

	err = hsa_init();
	check(Initializing the hsa runtime, err);

	/*
	 * Iterate over the agents and pick the gpu agent using
	 * the find_gpu callback.
	 */
	hsa_agent_t device = 0;
	err = hsa_iterate_agents(find_gpu, &device);
	check(Calling hsa_iterate_agents, err);

	err = (device == 0) ? HSA_STATUS_ERROR : HSA_STATUS_SUCCESS;
	check(Checking if the GPU device is non-zero, err);

	/*
	 * Query the name of the device.
	 */
	char name[64] = { 0 };
	err = hsa_agent_get_info(device, HSA_AGENT_INFO_NAME, name);
	check(Querying the device name, err);
	printf("The device name is %s.\n", name);

	/*
	 * Query the maximum size of the queue.
	 */
	uint32_t queue_size = 0;
	err = hsa_agent_get_info(device, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
	check(Querying the device maximum queue size, err);
	printf("The maximum queue size is %u.\n", (unsigned int) queue_size);

	/*
	 * Create a queue using the maximum size.
	 */
	hsa_queue_t* commandQueue;
	err = hsa_queue_create(device, queue_size, HSA_QUEUE_TYPE_MULTI, NULL, NULL, &commandQueue);
	check(Creating the queue, err);

	/*
	 * Create hsa program.
	 */
	hsa_ext_program_handle_t hsaProgram;
	err = hsa_ext_program_create(&device, 1, HSA_EXT_BRIG_MACHINE_LARGE, HSA_EXT_BRIG_PROFILE_FULL, &hsaProgram);
	check(Creating the hsa program, err);

	/*
	 * Add the BRIG module to hsa program.
	 */
	hsa_ext_brig_module_handle_t module;
	err = hsa_ext_add_module(hsaProgram, brig_module, &module);
	check(Adding the brig module to the program, err);

  /*
	 * Construct finalization request list.
	 */
	hsa_ext_finalization_request_t finalization_request_list;
	finalization_request_list.module = module;
	finalization_request_list.program_call_convention = 0;
	char kernel_name[128] = "&__OpenCL_run_kernel";
	finalization_request_list.symbol = section_offset;
	check(Finding the symbol offset for the kernel, err);

	/*
	 * Finalize the hsa program.
	 */
	err = hsa_ext_finalize_program(hsaProgram, device, 1, &finalization_request_list, NULL, NULL, 0, NULL, 0);
	check(Finalizing the program, err);

	/*
	 * Destroy the brig module. The program was successfully created the kernel
	 * symbol was found and the program was finalized, so it is no longer needed.
	 */
	 destroy_brig_module(brig_module);


  /*
	 * Get the hsa code descriptor address.
	 */
	hsa_ext_code_descriptor_t *hsaCodeDescriptor;
	err = hsa_ext_query_kernel_descriptor_address(hsaProgram, module, finalization_request_list.symbol, &hsaCodeDescriptor);
	check(Querying the kernel descriptor address, err);

	/*
	 * Create a signal to wait for the dispatch to finish.
	 */
	hsa_signal_t signal;
	err=hsa_signal_create(1, 0, NULL, &signal);
	check(Creating a HSA signal, err);

	/*
	 * Initialize the dispatch packet.
	 */
	hsa_dispatch_packet_t aql;
	memset(&aql, 0, sizeof(aql));

	/*
	 * Setup the dispatch information.
	 */
	aql.completion_signal=signal;
	aql.dimensions=1;
	aql.workgroup_size_x=256;
	aql.workgroup_size_y=1;
	aql.workgroup_size_z=1;
	aql.grid_size_x=1024*1024;
	aql.grid_size_y=1;
	aql.grid_size_z=1;
	aql.header.type=HSA_PACKET_TYPE_DISPATCH;
	aql.header.acquire_fence_scope=2;
	aql.header.release_fence_scope=2;
	aql.header.barrier=1;
	aql.group_segment_size=0;
	aql.private_segment_size=0;


	/*

	HSAContext context;
	HSAKernel kernel = context.create_kernel_binary("&__OpenCL_run_kernel",
			"/home/bsp/hsa_jni/kernels/max_value.brig");

	MaxValueKernel max_value(kernel);

	double test[10] = {1,2,3,4,5,6,7,8,9,-1};
	for (int i = 0 ;i < 10 ; ++i)
	{
		double max = max_value.max_value(test, 10, 1, 0);
		printf("Max is %f \n", max);
	}
	*/
}

#endif

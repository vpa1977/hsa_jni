#ifndef HSA_CONTEXT_IMPL_HPP
#define HSA_CONTEXT_IMPL_HPP

#include <regex.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <future>
#include <thread>
#include <chrono>
#include <map>
#include <stdint.h>
#include "HSAContext.h"
#include <unistd.h>
#include "hsa.h"
#include "hsa_ext_finalize.h"
#include "elf_utils.h"

#ifndef STATUS_CHECK

#define STATUS_CHECK(s,line) if ((int)status != (int)HSA_STATUS_SUCCESS) {\
		printf("### Error: %d at line:%d\n", s, line);\
                assert(HSA_STATUS_SUCCESS == hsa_shut_down());\
		exit(-1);\
	}

#define STATUS_CHECK_Q(s,line) if (status != HSA_STATUS_SUCCESS) {\
		printf("### Error: %d at line:%d\n", s, line);\
                assert(HSA_STATUS_SUCCESS == hsa_queue_destroy(commandQueue));\
                assert(HSA_STATUS_SUCCESS == hsa_shut_down());\
		exit(-1);\
	}
#endif

hsa_status_t get_kernarg(hsa_region_t region, void* data);
hsa_status_t get_shared(hsa_region_t region, void* data);

class HSAContextKaveriImpl: public HSAContext {
	friend HSAContext * HSAContext::Create();

public:
	hsa_queue_t* commandQueue;
public:
	/** hardware-specific. hardcoded for simplicity */
	size_t GetComputeUnits() {
		return 6;
	}
	size_t GetMaxWorkgroup() {
		return 512;
	}

private:

	class DispatchImpl;

	class KernelImpl: public HSAContext::Kernel {
	private:
		hsa_ext_program_handle_t hsaProgram;
		friend class DispatchImpl;
	public:
		HSAContextKaveriImpl* context;
		hsa_ext_code_descriptor_t *hsaCodeDescriptor;
		void *m_run_arg_buffer;

		KernelImpl(hsa_ext_program_handle_t program,
				hsa_ext_code_descriptor_t* _hsaCodeDescriptor,
				HSAContextKaveriImpl* _context);
		virtual ~KernelImpl();
	}; // end of KernelImpl

	class DispatchImpl: public HSAContext::Dispatch {
	private:
		HSAContextKaveriImpl* context;
		const KernelImpl* kernel;

		std::vector<uint8_t> arg_vec;
		uint32_t arg_count;
		size_t prevArgVecCapacity;
		int launchDimensions;
		uint32_t workgroup_size[3];
		uint32_t global_size[3];
		static const int ARGS_VEC_INITIAL_CAPACITY = 256 * 8;
		// bind kernel arguments
		//printf("arg_vec size: %d in bytes: %d\n", arg_vec.size(), arg_vec.size());
		hsa_region_t region;

		uint64_t kernargs_address;
		uint32_t kernargs_length;

		hsa_dispatch_packet_t aql;
		bool isDispatched;

	public:
		DispatchImpl(const KernelImpl* _kernel);
		virtual ~DispatchImpl();

		hsa_status_t pushFloatArg(float f);

		hsa_status_t pushIntArg(int i);

		hsa_status_t pushBooleanArg(unsigned char z);

		hsa_status_t pushByteArg(char b);

		hsa_status_t pushLongArg(long j);

		hsa_status_t pushDoubleArg(double d);

		hsa_status_t pushPointerArg(void *addr);

		// allow a previously pushed arg to be changed
		hsa_status_t setPointerArg(int idx, void *addr);

		hsa_status_t clearArgs();

		hsa_status_t setLaunchAttributes(int dims, size_t *globalDims,
				size_t *localDims);
		hsa_status_t dispatchKernelWaitComplete();
		std::future<void> dispatchKernelAndGetFuture();
		// dispatch a kernel asynchronously
		hsa_status_t dispatchKernel(hsa_signal_t& signal);
		// wait for the kernel to finish execution
		hsa_status_t waitComplete(hsa_signal_t& signal);
		void dispose();
	private:
		template<typename T>
		hsa_status_t pushArgPrivate(T val) {
			/* add padding if necessary */
			int padding_size = arg_vec.size() % sizeof(T);
			//printf("push %lu bytes into kernarg: ", sizeof(T) + padding_size);
			for (size_t i = 0; i < padding_size; ++i) {
				arg_vec.push_back((uint8_t) 0x00);
				//printf("%02X ", (uint8_t)0x00);
			}
			uint8_t*ptr = static_cast<uint8_t*>(static_cast<void*>(&val));
			//int offset = arg_vec.size();
			//arg_vec.resize(arg_vec.size() + sizeof(T));
			//memcpy(&arg_vec[offset], ptr, sizeof(T));

			for (size_t i = 0; i < sizeof(T); ++i) {
				arg_vec.push_back(ptr[i]);
				//printf("%02X ", ptr[i]);
			}
			//printf("\n");
			arg_count++;
			return HSA_STATUS_SUCCESS;
		}

		template<typename T>
		hsa_status_t setArgPrivate(int idx, T val) {
			// XXX disable this member function for now
			/*
			 // each arg takes up a 64-bit slot, no matter what its size
			 uint64_t  argAsU64 = 0;
			 T* pt = (T *) &argAsU64;
			 *pt = val;
			 arg_vec.at(idx) = argAsU64;
			 */
			return HSA_STATUS_SUCCESS;
		}

		void registerArgVecMemory();
		void computeLaunchAttr(int level, int globalSize, int localSize,
				int recommendedSize);
		// find largest factor less than or equal to start
		int findLargestFactor(int n, int start);

	}; // end of DispatchImpl

private:

	// constructor
	HSAContextKaveriImpl();
public:

	hsa_agent_t device;

	hsa_agent_t* getDevice();

	hsa_queue_t* getQueue();
	Dispatch* createDispatch(const Kernel* kernel);
	Kernel* createKernel(const char *brig_file, const char *entryName);
	hsa_status_t dispose();
	hsa_status_t registerArrayMemory(void *addr, int lengthInBytes);
	// destructor
	~HSAContextKaveriImpl();
};
// end of HSAContextKaveriImpl

template<typename T>
class TemplateDispatch {
public:
	TemplateDispatch(HSAContext::Kernel* p_kernel);
	virtual ~TemplateDispatch();
	void dispatchKernel(T kern_args, hsa_signal_t& signal, Launch_params_t lp);

	void waitComplete(hsa_signal_t& signal);
private:
	HSAContext::Kernel* m_p_kernel;
	void* run_kernel_arg_buffer = NULL;
	size_t run_kernel_arg_buffer_size;
};

template<typename T>
class MultiKernelDispatch {
public:
	MultiKernelDispatch(HSAContext * p_context) :
			m_pcontext(p_context) {
	}
	virtual ~MultiKernelDispatch();
	void addKernel(HSAContext::Kernel* p_kernel);
	void begin();
	void queueKernel(HSAContext::Kernel* p_kernel, T kern_args,
			Launch_params_t lp, bool wait =true);
	void runAndWait();
private:
	hsa_dispatch_packet_t m_aql;
	hsa_signal_t m_signal;
	hsa_signal_value_t m_expected_value;
	HSAContext * m_pcontext;
};

template<typename T>
MultiKernelDispatch<T>::~MultiKernelDispatch() {

}

template<typename T>
void MultiKernelDispatch<T>::addKernel(HSAContext::Kernel* p_kernel) {
	void *run_kernel_arg_buffer;
	hsa_region_t region;
	HSAContextKaveriImpl::KernelImpl *p_impl =
			(HSAContextKaveriImpl::KernelImpl *) p_kernel;
	size_t run_kernel_arg_buffer_size =
			p_impl->hsaCodeDescriptor->kernarg_segment_byte_size;
	hsa_agent_iterate_regions(p_impl->context->device, get_kernarg, &region);
	hsa_memory_allocate(region, run_kernel_arg_buffer_size,
			&run_kernel_arg_buffer);

	p_impl->m_run_arg_buffer= run_kernel_arg_buffer;
}

template<typename T>
void MultiKernelDispatch<T>::begin() {

}

template<typename T>
void MultiKernelDispatch<T>::queueKernel(HSAContext::Kernel* p_kernel,
		T kern_args, Launch_params_t lparm, bool wait) {

	if (wait)
	{
		hsa_status_t err;
		m_expected_value  =1;
		err = hsa_signal_create(m_expected_value, 0, NULL, &m_signal);
	}

	HSAContextKaveriImpl* p_ctx = (HSAContextKaveriImpl*) m_pcontext;
	/*  Obtain the current queue write index. increases with each call to kernel  */
	uint64_t index = hsa_queue_load_write_index_relaxed(p_ctx->commandQueue);
	/* printf("DEBUG:Call #%d to kernel \"%s\" \n",(int) index+1,"run"); */

	HSAContextKaveriImpl::KernelImpl *p_impl =
			(HSAContextKaveriImpl::KernelImpl *) p_kernel;

	/*  Setup this call to this kernel dispatch packet from scratch.  */
	memset(&m_aql, 0, sizeof(m_aql));

	/*  Set the dimensions passed from the application */
	m_aql.dimensions = (uint16_t) lparm.ndim;
	m_aql.grid_size_x = lparm.gdims[0];
	m_aql.workgroup_size_x = lparm.ldims[0];
	m_aql.completion_signal = m_signal;
	if (lparm.ndim > 1) {
		m_aql.grid_size_y = lparm.gdims[1];
		m_aql.workgroup_size_y = lparm.ldims[1];
	} else {
		m_aql.grid_size_y = 1;
		m_aql.workgroup_size_y = 1;
	}
	if (lparm.ndim > 2) {
		m_aql.grid_size_z = lparm.gdims[2];
		m_aql.workgroup_size_z = lparm.ldims[2];
	} else {
		m_aql.grid_size_z = 1;
		m_aql.workgroup_size_z = 1;
	}

	/*  In the future, we may use environment variables for some of these */
	m_aql.header.type = HSA_PACKET_TYPE_DISPATCH;
	m_aql.header.acquire_fence_scope = 2;
	m_aql.header.release_fence_scope = 2;
	m_aql.header.barrier = 1;
	m_aql.group_segment_size =
			p_impl->hsaCodeDescriptor->workgroup_group_segment_byte_size;
	m_aql.private_segment_size =
			p_impl->hsaCodeDescriptor->workitem_private_segment_byte_size;

	void * run_kernel_arg_buffer = p_impl->m_run_arg_buffer;
	/*  copy args from the custom run_args structure */
	/*  FIXME We should align kernel_arg_buffer because run_args is aligned */
	memcpy(run_kernel_arg_buffer, &kern_args, sizeof(kern_args));

	/*  Bind kernelcode to the packet.  */
	m_aql.kernel_object_address = p_impl->hsaCodeDescriptor->code.handle;

	/*  Bind kernel argument buffer to the aql packet.  */
	m_aql.kernarg_address = (uint64_t) run_kernel_arg_buffer;


	const uint32_t queueMask = p_ctx->commandQueue->size - 1;
	const uint32_t pos = (index) & queueMask;
	((hsa_dispatch_packet_t*) (p_ctx->commandQueue->base_address))[pos] = m_aql;

	if (wait)
		hsa_signal_store_relaxed(p_ctx->commandQueue->doorbell_signal, index);
	/* Increment the write index and ring the doorbell to dispatch the kernel.  */
	hsa_queue_store_write_index_relaxed(p_ctx->commandQueue, index + 1);

	if (wait)
	{
		hsa_signal_wait_acquire(m_signal, HSA_LT, m_expected_value, (uint64_t)-1, HSA_WAIT_EXPECTANCY_SHORT);
		hsa_signal_destroy(m_signal);
		m_signal = 0;
	}
	m_expected_value--;

}

template<typename T>
void MultiKernelDispatch<T>::runAndWait() {

	/*  Wait on the dispatch signal until the kernel is finished.  */
	/*while (hsa_signal_load_relaxed(m_signal) != m_expected_value)
		usleep(1000);
	*/

}


template<typename T>
TemplateDispatch<T>::TemplateDispatch(HSAContext::Kernel* p_kernel) :
	m_p_kernel(p_kernel) {
hsa_region_t region;
HSAContextKaveriImpl::KernelImpl *p_impl =
		(HSAContextKaveriImpl::KernelImpl *) m_p_kernel;
run_kernel_arg_buffer_size =
		p_impl->hsaCodeDescriptor->kernarg_segment_byte_size;
hsa_agent_iterate_regions(p_impl->context->device, get_kernarg, &region);
hsa_memory_allocate(region, run_kernel_arg_buffer_size, &run_kernel_arg_buffer);

}

template<typename T>
TemplateDispatch<T>::~TemplateDispatch() {
hsa_memory_free(run_kernel_arg_buffer);
}

template<typename T>
void TemplateDispatch<T>::waitComplete(hsa_signal_t& signal) {
/*  Wait on the dispatch signal until the kernel is finished.  */
hsa_signal_wait_acquire(signal, HSA_LT, 1, (uint64_t) -1,
		HSA_WAIT_EXPECTANCY_UNKNOWN);

hsa_signal_destroy(signal);

}

template<typename T>
void TemplateDispatch<T>::dispatchKernel(T run_args, hsa_signal_t& signal,
	const Launch_params_t lparm) {
hsa_dispatch_packet_t run_Aql;
HSAContextKaveriImpl::KernelImpl *p_impl =
		(HSAContextKaveriImpl::KernelImpl *) m_p_kernel;

hsa_status_t err;
status_t status = STATUS_SUCCESS;
/*  Create a signal to wait for the dispatch to finish.  */
err = hsa_signal_create(1, 0, NULL, &signal);
STATUS_CHECK(err, __LINE__);

/*  Setup this call to this kernel dispatch packet from scratch.  */
memset(&run_Aql, 0, sizeof(run_Aql));
run_Aql.completion_signal = signal;

/*  Set the dimensions passed from the application */
run_Aql.dimensions = (uint16_t) lparm.ndim;
run_Aql.grid_size_x = lparm.gdims[0];
run_Aql.workgroup_size_x = lparm.ldims[0];
if (lparm.ndim > 1) {
	run_Aql.grid_size_y = lparm.gdims[1];
	run_Aql.workgroup_size_y = lparm.ldims[1];
} else {
	run_Aql.grid_size_y = 1;
	run_Aql.workgroup_size_y = 1;
}
if (lparm.ndim > 2) {
	run_Aql.grid_size_z = lparm.gdims[2];
	run_Aql.workgroup_size_z = lparm.ldims[2];
} else {
	run_Aql.grid_size_z = 1;
	run_Aql.workgroup_size_z = 1;
}

/*  In the future, we may use environment variables for some of these */
run_Aql.header.type = HSA_PACKET_TYPE_DISPATCH;
run_Aql.header.acquire_fence_scope = 2;
run_Aql.header.release_fence_scope = 2;
run_Aql.header.barrier = 1;
run_Aql.group_segment_size =
		p_impl->hsaCodeDescriptor->workgroup_group_segment_byte_size;
run_Aql.private_segment_size =
		p_impl->hsaCodeDescriptor->workitem_private_segment_byte_size;

/*  copy args from the custom run_args structure */
/*  FIXME We should align kernel_arg_buffer because run_args is aligned */
memcpy(run_kernel_arg_buffer, &run_args, sizeof(run_args));

/*  Bind kernelcode to the packet.  */
run_Aql.kernel_object_address = p_impl->hsaCodeDescriptor->code.handle;

/*  Bind kernel argument buffer to the aql packet.  */
run_Aql.kernarg_address = (uint64_t) run_kernel_arg_buffer;

/*  Obtain the current queue write index. increases with each call to kernel  */
uint64_t index = hsa_queue_load_write_index_relaxed(
		p_impl->context->commandQueue);
/* printf("DEBUG:Call #%d to kernel \"%s\" \n",(int) index+1,"run"); */

/*  Write the run_Aql packet at the calculated queue index address.  */
const uint32_t queueMask = p_impl->context->commandQueue->size - 1;
const uint32_t pos = index & queueMask;
((hsa_dispatch_packet_t*) (p_impl->context->commandQueue->base_address))[pos] =
		run_Aql;

/* Increment the write index and ring the doorbell to dispatch the kernel.  */
hsa_queue_store_write_index_relaxed(p_impl->context->commandQueue, index + 1);

hsa_signal_store_relaxed(p_impl->context->commandQueue->doorbell_signal, index);
return;
}

#endif

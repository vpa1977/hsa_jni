#ifndef TRANSFORM_HPP_
#define TRANSFORM_HPP_

#include "HSAContext.h"
#include "HSAContextImpl.hpp"
#include <vector>
#include <math.h>
#include <algorithm>
#include <string.h>
#include <unistd.h>
/**
 * Reduces a double vector in single valur
 */
class VectorSum
{
public:
	VectorSum(std::shared_ptr<HSAContext> context, const std::string& dump)
	{
		m_local_kernel = context->createKernel(dump.c_str(), "&__OpenCL_run_kernel");
		m_dispatch = new MultiKernelDispatch<GlobalArgs>(context.get());
		m_dispatch->addKernel(m_local_kernel);
		memset(&m_args, 0, sizeof(m_args));
	}
	virtual ~VectorSum()
	{
		delete m_dispatch; m_dispatch = NULL;
	}

	void begin()
	{
		m_dispatch->begin();
	}

	void commit()
	{
		m_dispatch->runAndWait();
	}

	void compute(int global_size, double * input1, double* input2, double* result, int size, bool wait)
	{
		m_args.size = size;
		m_args.max = size / global_size;
		m_args.stride = global_size;
		m_args.input1 = input1;
		m_args.input2 = input2;
		m_args.result = result;
		hsa_signal_t signal;
		Launch_params_t lp1 {.ndim=1, .gdims={global_size}, .ldims={256}};
		m_dispatch->queueKernel(m_local_kernel,m_args, lp1,wait);
	}

protected:
	struct GlobalArgs
		{
			long global_offset_0;
			long global_offset_1;
			long global_offset_2;
			long* printf_buffer;
			long* vqueue_pointer;
			long* aqlwrap_pointer;
			double* input1;
			double* input2;
			double* result;
			long max;
			long stride;
			long size;

		} __attribute__ ((aligned (16)));

		GlobalArgs m_args;
private:
	std::vector<double> m_result;
	std::shared_ptr<HSAContext> m_context;
	HSAContext::Kernel* m_local_kernel;
	MultiKernelDispatch<GlobalArgs>* m_dispatch;
};

class VectorSumTemplate
{
public:
	VectorSumTemplate(std::shared_ptr<HSAContext> context, const std::string& dump)
	{
		m_local_kernel = context->createKernel(dump.c_str(), "&__OpenCL_run_kernel");
		m_dispatch = new TemplateDispatch<GlobalArgs>(m_local_kernel);

		memset(&m_args, 0, sizeof(m_args));
	}
	virtual ~VectorSumTemplate()
	{
		delete m_dispatch; m_dispatch = NULL;
	}



	void compute(int global_size, double * input1, double* input2, double* result, int size)
	{
		m_args.size = size;
		m_args.max = size / global_size;
		m_args.stride = global_size;
		m_args.input1 = input1;
		m_args.input2 = input2;
		m_args.result = result;
		hsa_signal_t signal;
		Launch_params_t lp1 {.ndim=1, .gdims={global_size}, .ldims={256}};

		m_dispatch->dispatchKernel(m_args,signal, lp1);
		m_dispatch->waitComplete(signal);
	}

protected:
	struct GlobalArgs
		{
			long global_offset_0;
			long global_offset_1;
			long global_offset_2;
			long* printf_buffer;
			long* vqueue_pointer;
			long* aqlwrap_pointer;
			double* input1;
			double* input2;
			double* result;
			long max;
			long stride;
			long size;

		} __attribute__ ((aligned (16)));

		GlobalArgs m_args;
private:
	std::vector<double> m_result;
	std::shared_ptr<HSAContext> m_context;
	HSAContext::Kernel* m_local_kernel;
	TemplateDispatch<GlobalArgs>* m_dispatch;
};



#endif /* MERGE_SORT_HPP_ */

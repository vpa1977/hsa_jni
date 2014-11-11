#ifndef DUMP_HPP_
#define DUMP_HPP_

#include "HSAContext.h"
#include <vector>
#include <math.h>
#include <algorithm>
#include <string.h>

class Dump
{
public:
	Dump(std::shared_ptr<HSAContext> context, const std::string& dump)
	{
		m_local_kernel = context->createKernel(dump.c_str(), "&__OpenCL_run_kernel");
		m_local_dispatch = context->createDispatch(m_local_kernel);
	}
	virtual ~Dump()
	{
		delete m_local_dispatch; m_local_dispatch = NULL;
	}

public:

	void run(int global_size, double* arg)
	{
		m_local_dispatch->clearArgs();
		FIX_ARGS_STABLE(m_local_dispatch);
		m_local_dispatch->pushPointerArg(arg);
		size_t global_dims[3] = {global_size,1,1};
		size_t local_dims[3] = {256,1,1};
		m_local_dispatch->setLaunchAttributes(1, global_dims,  local_dims);
		m_local_dispatch->dispatchKernelWaitComplete();
	}

private:
	std::shared_ptr<HSAContext> m_context;
	HSAContext::Kernel* m_local_kernel;
	HSAContext::Dispatch* m_local_dispatch;
};


#endif /* MERGE_SORT_HPP_ */

#ifndef REDUCE_HPP_
#define REDUCE_HPP_

#include "HSAContext.h"
#include <vector>
#include <math.h>
#include <algorithm>
#include <string.h>

/**
 * Reduces a double vector in single valur
 */
class Transform
{
public:
	Transform(std::shared_ptr<HSAContext> context, const std::string& dump)
	{
		m_local_kernel = context->createKernel(dump.c_str(), "&__OpenCL_run_kernel");
		m_local_dispatch = context->createDispatch(m_local_kernel);
	}
	virtual ~Reduce()
	{
		delete m_local_dispatch; m_local_dispatch = NULL;
	}

public:


	double transform(double* ptr, size_t size)
	{

		size_t num_wg = 64;
		size_t num_compute_units = 6;
		size_t global_size = num_wg * num_compute_units;
		size_t workgroup_size = 256;

		m_result.resize( global_size);


		m_local_dispatch->clearArgs();
		FIX_ARGS_STABLE(m_local_dispatch);

		m_local_dispatch->pushPointerArg((void*)ptr);
		m_local_dispatch->pushIntArg((int)size );
		m_local_dispatch->pushPointerArg((void*)&m_result[0]);
		size_t global_dims[3] = { std::min(roundUp(size, workgroup_size), global_size*workgroup_size),1,1};
		size_t local_dims[3] = {workgroup_size,1,1};
		m_local_dispatch->setLaunchAttributes(1, global_dims,  local_dims);
		m_local_dispatch->dispatchKernelWaitComplete();
		if (size < workgroup_size)
		{
			return m_result[0];
		}
		return reduceTail(size);
	}
protected:

	size_t roundUp(size_t size, size_t wg_size) {
	  size_t times = size / wg_size;
	  size_t rem = size % wg_size;
	  if (rem != 0) ++times;
	  return times * wg_size;
	}


	double reduceTail(size_t size)
	{
		double result = 0;
		size_t num_wg = 64;
		size_t workgroup_size = 256;
		size_t ceilNumWg = (size_t)ceil(size/workgroup_size);
		int numTailReduce = std::min( ceilNumWg, num_wg );
		for (int i = 0; i < numTailReduce ; ++i)
			result += m_result[i];
		return result;
	}

private:
	std::vector<double> m_result;
	std::shared_ptr<HSAContext> m_context;
	HSAContext::Kernel* m_local_kernel;
	HSAContext::Dispatch* m_local_dispatch;
};


#endif /* MERGE_SORT_HPP_ */

/*
 * max_value.hpp
 *
 *  Created on: 23/10/2014
 *      Author: bsp
 */

#ifndef MAX_VALUE_HPP_
#define MAX_VALUE_HPP_

#include "HSAContext.h"
#include <vector>
#include <math.h>
#include <algorithm>
/**
 * Implements a hsa-backed max-value kernel
 * the brig-module is a temporary parameter, until proper resource management is established
 */
class MaxValue
{
public:
	MaxValue(std::shared_ptr<HSAContext> context, const std::string& brig_module)
	{
		m_kernel = context->createKernel(brig_module.c_str(), "&__OpenCL_run_kernel");
		m_dispatch = context->createDispatch(m_kernel);
		m_context = context;
	}
	virtual ~MaxValue()
	{
		delete m_dispatch; m_dispatch = NULL;
		delete m_kernel; m_kernel = NULL;
	}

public:
	double find(const std::vector<double>& in)
	{
		return find((double*)&in[0], in.size(),1,0);
	}


	double find(double* ptr, size_t size, int stride, int offset)
	{

		size_t num_wg = 64;
		size_t num_compute_units = 6;
		size_t global_size = num_wg * num_compute_units;
		size_t workgroup_size = 256;

		m_result.resize( global_size);

		m_dispatch->clearArgs();
		m_dispatch->pushPointerArg(0);
		m_dispatch->pushPointerArg(0);
		m_dispatch->pushPointerArg(0);
		m_dispatch->pushPointerArg(0);
		m_dispatch->pushPointerArg(0);
		m_dispatch->pushPointerArg(0);

		m_dispatch->pushPointerArg((void*)ptr);
		m_dispatch->pushIntArg((int)size );
		m_dispatch->pushIntArg(stride);
		m_dispatch->pushIntArg(offset);
		m_dispatch->pushPointerArg((void*)&m_result[0]);
		size_t global_dims[3] = {workgroup_size*workgroup_size,1,1};
		size_t local_dims[3] = {workgroup_size,1,1};
		m_dispatch->setLaunchAttributes(1, global_dims,  local_dims);
		m_dispatch->dispatchKernelWaitComplete();

		return post_reduce(ptr, size);
	}
protected:
	virtual double post_reduce(double* in, size_t size)
	{
		int stride = 1;
		int offset = 0;

		size_t num_wg = 64;
		size_t workgroup_size = 256;

		size_t ceilNumWg = (size_t)ceil(size/workgroup_size);
		int numTailReduce = std::min( ceilNumWg, num_wg );
		int minele_indx =  m_result[0] *stride + offset;
		float minele =  in[ minele_indx ];
		for (int i = 0 ;i < numTailReduce ; ++i)
		{
			int index = m_result[i]*stride + offset;
			bool stat = in[ index ] < minele;
			minele = stat ? minele : in[ index ];
			minele_indx =  stat ? minele_indx : m_result[i];
		}
		return minele;


	}
protected:
	std::vector<int> m_result;
private:
	std::shared_ptr<HSAContext> m_context;
	HSAContext::Kernel* m_kernel;
	HSAContext::Dispatch* m_dispatch;


};

class MinValue : public MaxValue
{
public:
	MinValue(std::shared_ptr<HSAContext> context, const std::string& brig_module) : MaxValue(context, brig_module){}
private:
	virtual double post_reduce(double* in, size_t size)
	{
		int stride = 1;
		int offset = 0;

		size_t num_wg = 64;

		size_t workgroup_size = 256;

		size_t ceilNumWg = (size_t)ceil(size/workgroup_size);
		int numTailReduce = std::min( ceilNumWg, num_wg );
		int minele_indx =  m_result[0] *stride + offset;
		float minele =  in[ minele_indx ];
		for (int i = 0 ;i < numTailReduce ; ++i)
		{
			int index = m_result[i]*stride + offset;
			bool stat = in[ index ] > minele;
			minele = stat ? minele : in[ index ];
			minele_indx =  stat ? minele_indx : m_result[i];
		}
		return minele;
	}

};




#endif /* MAX_VALUE_HPP_ */

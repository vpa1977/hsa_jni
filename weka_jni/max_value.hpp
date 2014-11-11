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


class MinMaxValue
{
public:
	MinMaxValue(std::shared_ptr<HSAContext> context, const std::string& brig_module)
	{
		m_kernel = context->createKernel(brig_module.c_str(), "&__OpenCL_run_kernel");
		m_dispatch = context->createDispatch(m_kernel);
		m_context = context;
	}
	virtual ~MinMaxValue()
	{
		delete m_dispatch; m_dispatch = NULL;
		delete m_kernel; m_kernel = NULL;
	}


	void find(double* ptr, size_t size,  double& min, double &max)
	{

		size_t num_wg = 64;
		size_t num_compute_units = 6;
		size_t global_size = num_wg * num_compute_units;
		size_t workgroup_size = 256;

		m_result_min.resize( global_size);
		m_result_max.resize( global_size);

		m_dispatch->clearArgs();
		FIX_ARGS_STABLE(m_dispatch);

		m_dispatch->pushPointerArg((void*)ptr);
		m_dispatch->pushIntArg((int)size );
		m_dispatch->pushPointerArg((void*)&m_result_min[0]);
		m_dispatch->pushPointerArg((void*)&m_result_max[0]);
		size_t global_dims[3] = { std::min(roundUp(size, workgroup_size), global_size*workgroup_size),1,1};
		size_t local_dims[3] = {workgroup_size,1,1};
		m_dispatch->setLaunchAttributes(1, global_dims,  local_dims);
		m_dispatch->dispatchKernelWaitComplete();
		if (size < workgroup_size)
		{
			min = ptr[ m_result_min[0]];
			max = ptr[ m_result_max[0]];
		}
		return reduce(ptr, size, min,max);
	}
protected:

	size_t roundUp(size_t size, size_t wg_size) {
	  size_t times = size / wg_size;
	  size_t rem = size % wg_size;
	  if (rem != 0) ++times;
	  return times * wg_size;
	}


	virtual void reduce(double* in, size_t size, double& min, double& max)
	{
		size_t num_wg = 64;
		size_t workgroup_size = 256;

		size_t ceilNumWg = (size_t)ceil(size/workgroup_size);
		int numTailReduce = std::min( ceilNumWg, num_wg );
		int min_idx =  m_result_min[0];
		min =  in[ min_idx ];

		int max_idx = m_result_max[0];
		max = in[ max_idx ];

		for (int i = 0 ;i < numTailReduce ; ++i)
		{
			int index = m_result_min[i];
			if (in[index] < min)
			{
				min = in[index];
				min_idx = m_result_min[i];
			}
			index = m_result_max[i];
			if (in[index] > max)
			{
				max = in[index];
				max_idx = m_result_max[i];
			}
		}
	}
protected:
	std::vector<int> m_result_max;
	std::vector<int> m_result_min;
private:
	std::shared_ptr<HSAContext> m_context;
	HSAContext::Kernel* m_kernel;
	HSAContext::Dispatch* m_dispatch;


};


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
		FIX_ARGS_STABLE(m_dispatch);

		m_dispatch->pushPointerArg((void*)ptr);
		m_dispatch->pushIntArg((int)size );
		m_dispatch->pushIntArg(stride);
		m_dispatch->pushIntArg(offset);
		m_dispatch->pushPointerArg((void*)&m_result[0]);
		size_t global_dims[3] = { std::min(roundUp(size, workgroup_size), global_size*workgroup_size),1,1};
		size_t local_dims[3] = {workgroup_size,1,1};
		m_dispatch->setLaunchAttributes(1, global_dims,  local_dims);
		m_dispatch->dispatchKernelWaitComplete();
		if (size < workgroup_size)
			return ptr[m_result[0]];
		return post_reduce(ptr, size);
	}
protected:

	size_t roundUp(size_t size, size_t wg_size) {
	  size_t times = size / wg_size;
	  size_t rem = size % wg_size;
	  if (rem != 0) ++times;
	  return times * wg_size;
	}


	virtual double post_reduce(double* in, size_t size)
	{
		size_t num_wg = 64;
		size_t workgroup_size = 256;

		size_t ceilNumWg = (size_t)ceil(size/workgroup_size);
		int numTailReduce = std::min( ceilNumWg, num_wg );
		int minele_indx =  m_result[0];
		float minele =  in[ minele_indx ];
		for (int i = 0 ;i < numTailReduce ; ++i)
		{
			int index = m_result[i];
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
		size_t num_wg = 64;

		size_t workgroup_size = 256;

		size_t ceilNumWg = (size_t)ceil(size/workgroup_size);
		int numTailReduce = std::min( ceilNumWg, num_wg );
		int minele_indx =  m_result[0];
		float minele =  in[ minele_indx ];
		for (int i = 0 ;i < numTailReduce ; ++i)
		{
			int index = m_result[i];
			bool stat = in[ index ] > minele;
			minele = stat ? minele : in[ index ];
			minele_indx =  stat ? minele_indx : m_result[i];
		}
		return minele;
	}

};




#endif /* MAX_VALUE_HPP_ */

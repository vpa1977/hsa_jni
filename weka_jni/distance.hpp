/*
 * distance.hpp
 *
 *  Created on: 28/10/2014
 *      Author: bsp
 */

#ifndef DISTANCE_HPP_
#define DISTANCE_HPP_

#include <string>
#include "HSAContext.h"


class PerAttributeDistance
{
public:
	PerAttributeDistance( std::shared_ptr<HSAContext> context, const std::string numerics_module,
			const std::string nominals_module)
	{
		m_context = context;
		m_numeric_kernel = context->createKernel(numerics_module.c_str(), "&__OpenCL_run_kernel");
		m_nominal_kernel = context->createKernel(nominals_module.c_str(), "&__OpenCL_run_kernel");
		m_numeric_dispatch = context->createDispatch(m_numeric_kernel);
		m_nominal_dispatch = context->createDispatch(m_nominal_kernel);
	}


public:
	void distance(double* test, double* samples,double * ranges, size_t numerics, int nominals, size_t window_size,size_t current_size, double* result)
	{
		size_t global_dims[3] = {current_size,1,1};
		size_t local_dims[3] = {256,1,1};
		std::vector<double> tmp;
		tmp.resize(window_size * numerics)
		int i =0;
		for (;i < numerics ; ++i)
		{
			int samples_offset = i*window_size;
			m_numeric_dispatch->clearArgs();
			FIX_ARGS_STABLE(m_numeric_dispatch);
			m_numeric_dispatch->pushPointerArg(samples);
			m_numeric_dispatch->pushDoubleArg(test[i]);
			m_numeric_dispatch->pushDoubleArg(ranges[i*2]);
			m_numeric_dispatch->pushDoubleArg(ranges[i*2]+1);
			m_numeric_dispatch->pushIntArg(window_size);
			m_numeric_dispatch->pushIntArg(samples_offset);
			m_numeric_dispatch->pushPointerArg( result );
			m_numeric_dispatch->setLaunchAttributes(1, global_dims, local_dims);
			m_numeric_dispatch->dispatchKernelWaitComplete();
		}

		for ( ; i < numerics+nominals; ++i )
		{
			int samples_offset = i*window_size;
			m_nominal_dispatch->clearArgs();
			FIX_ARGS_STABLE(m_nominal_dispatch);
			m_nominal_dispatch->pushPointerArg(samples);
			m_nominal_dispatch->pushDoubleArg(test[i]);
			m_nominal_dispatch->pushDoubleArg(ranges[i*2]);
			m_nominal_dispatch->pushDoubleArg(ranges[i*2]+1);
			m_nominal_dispatch->pushIntArg(window_size);
			m_nominal_dispatch->pushIntArg(samples_offset);
			m_nominal_dispatch->pushPointerArg( result );
			m_nominal_dispatch->setLaunchAttributes(1, global_dims, local_dims);
			m_nominal_dispatch->dispatchKernelWaitComplete();
		}

	}

private:
	std::shared_ptr<HSAContext> m_context;
	HSAContext::Dispatch* m_numeric_dispatch;
	HSAContext::Kernel* m_numeric_kernel;

	HSAContext::Dispatch* m_nominal_dispatch;
	HSAContext::Kernel* m_nominal_kernel;

};

class SquareDistance
{
public:
		SquareDistance(std::shared_ptr<HSAContext> context, const std::string& brig_module)
		{
			m_kernel = context->createKernel(brig_module.c_str(), "&__OpenCL_square_distance_kernel");
			m_dispatch = context->createDispatch(m_kernel);
			m_context = context;
		}
		virtual ~SquareDistance()
		{
			delete m_dispatch; m_dispatch = NULL;
			delete m_kernel; m_kernel = NULL;
		}

		void calculate(double* input, double * samples, size_t samples_size, double* ranges, int element_count,
						int numerics_size, double* result)
		{
			m_dispatch->clearArgs();
			FIX_ARGS_STABLE(m_dispatch);
			size_t workgroup_size = m_context->GetMaxWorkgroup();

			m_dispatch->pushPointerArg((void*)input);
			m_dispatch->pushPointerArg((void*)samples);
			m_dispatch->pushPointerArg((void*)ranges);
			m_dispatch->pushPointerArg((void*)result);
			m_dispatch->pushIntArg(element_count);
			m_dispatch->pushIntArg(numerics_size);
			size_t global_dims[3] = {samples_size,1,1};
			size_t local_dims[3] = {workgroup_size,1,1};
			m_dispatch->setLaunchAttributes(1, global_dims,  local_dims);
			m_dispatch->dispatchKernelWaitComplete();
		}
private:
		std::shared_ptr<HSAContext> m_context;
		HSAContext::Dispatch* m_dispatch;
		HSAContext::Kernel* m_kernel;
};




#endif /* DISTANCE_HPP_ */


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


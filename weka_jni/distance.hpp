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
#include "HSAContextImpl.hpp"


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
	void distance(double* test, double* samples,double * ranges, size_t numerics, size_t instance_size, size_t window_size,size_t current_size, double* result)
	{
		size_t global_dims[3] = {current_size,instance_size,1};
		size_t local_dims[3] = {256,256,1};
		size_t global_dims1[3] = { current_size, 1, 1};
		size_t local_dims1[3] = {256,1,1};
		tmp.resize(current_size * (instance_size));
		m_numeric_dispatch->clearArgs();
		FIX_ARGS_STABLE(m_numeric_dispatch);
		m_numeric_dispatch->pushPointerArg(samples);
		m_numeric_dispatch->pushPointerArg(test);
		m_numeric_dispatch->pushPointerArg(ranges);
		m_numeric_dispatch->pushIntArg( window_size );
		m_numeric_dispatch->pushIntArg( numerics );
		m_numeric_dispatch->pushIntArg( instance_size );
		m_numeric_dispatch->pushPointerArg(&tmp[0]);
		m_numeric_dispatch->setLaunchAttributes(2, global_dims, local_dims);
		m_numeric_dispatch->dispatchKernelWaitComplete();
		m_nominal_dispatch->clearArgs();
		FIX_ARGS_STABLE(m_nominal_dispatch);
		m_nominal_dispatch->pushPointerArg( &tmp[0]);
		m_nominal_dispatch->pushIntArg(instance_size);
		m_nominal_dispatch->pushPointerArg( result );
		m_nominal_dispatch->setLaunchAttributes(1, global_dims1, local_dims1);
		m_nominal_dispatch->dispatchKernelWaitComplete();

/*		int i =0;
		for (;i < numerics ; ++i)
		{
			int samples_offset = i*window_size;

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
		}*/

	}

private:
	std::vector<double> tmp;
	std::shared_ptr<HSAContext> m_context;
	HSAContext::Dispatch* m_numeric_dispatch;
	HSAContext::Kernel* m_numeric_kernel;

	HSAContext::Dispatch* m_nominal_dispatch;
	HSAContext::Kernel* m_nominal_kernel;

};

/**
 * 1 thread per dot product
 */
class SquareDistance
{
public:
	struct GlobalArgs
	{
		long global_offset_0;
		long global_offset_1;
		long global_offset_2;
		long* printf_buffer;
		long* vqueue_pointer;
		long* aqlwrap_pointer;
		double* input;
		double* samples;
		double* ranges;
		double* result;
		int element_count;
		int numerics_size;

	} __attribute__ ((aligned (16)));

	GlobalArgs m_global_args;

		SquareDistance(std::shared_ptr<HSAContext> context, const std::string& brig_module)
		{
			m_kernel = context->createKernel(brig_module.c_str(), "&__OpenCL_square_distance_kernel");
			m_dispatch = new TemplateDispatch<GlobalArgs>(m_kernel);//context->createDispatch(m_kernel);
			m_context = context;
			memset(&m_global_args, 0, sizeof(m_global_args));
		}
		virtual ~SquareDistance()
		{
			delete m_dispatch; m_dispatch = NULL;
			delete m_kernel; m_kernel = NULL;
		}

		void calculate(double* input, double * samples, size_t samples_size, double* ranges, int element_count,
						int numerics_size, double* result)
		{
			//m_dispatch->clearArgs();
			//FIX_ARGS_STABLE(m_dispatch);
			//size_t workgroup_size = m_context->GetMaxWorkgroup();
			m_global_args.input = input;
			m_global_args.samples = samples;
			m_global_args.ranges = ranges;
			m_global_args.result = result;
			m_global_args.element_count = element_count;
			m_global_args.numerics_size = numerics_size;

			/*m_dispatch->pushPointerArg((void*)input);
			m_dispatch->pushPointerArg((void*)samples);
			m_dispatch->pushPointerArg((void*)ranges);
			m_dispatch->pushPointerArg((void*)result);
			m_dispatch->pushIntArg(element_count);
			m_dispatch->pushIntArg(numerics_size);
			*/
	//		size_t global_dims[3] = {samples_size,1,1};
	//		size_t local_dims[3] = {workgroup_size,1,1};
			hsa_signal_t signal;
			Launch_params_t lp1 {.ndim=1, .gdims={samples_size}, .ldims={256}};
			m_dispatch->dispatchKernel(m_global_args, signal, lp1);
			m_dispatch->waitComplete(signal);
			//m_dispatch->setLaunchAttributes(1, global_dims,  local_dims);
			//m_dispatch->dispatchKernelWaitComplete();
		}
private:
		std::shared_ptr<HSAContext> m_context;
		TemplateDispatch<GlobalArgs>* m_dispatch;
		HSAContext::Kernel* m_kernel;
};

/**
 * use p-threads with pre-loading
 */
class SquareDistance2
{
public:
	struct GlobalArgs
	{
		long global_offset_0;
		long global_offset_1;
		long global_offset_2;
		long* printf_buffer;
		long* vqueue_pointer;
		long* aqlwrap_pointer;
		double* input;
		double* samples;
		double* ranges;
		double* result;
		int element_count;
		int numerics_size;

	} __attribute__ ((aligned (16)));

	GlobalArgs m_global_args;

		SquareDistance2(std::shared_ptr<HSAContext> context, const std::string& brig_module)
		{
			m_kernel = context->createKernel(brig_module.c_str(), "&__OpenCL_square_distance_kernel");
			m_dispatch = new TemplateDispatch<GlobalArgs>(m_kernel);//context->createDispatch(m_kernel);
			m_context = context;
			memset(&m_global_args, 0, sizeof(m_global_args));
		}
		virtual ~SquareDistance2()
		{
			delete m_dispatch; m_dispatch = NULL;
			delete m_kernel; m_kernel = NULL;
		}

		void calculate(double* input, double * samples, size_t samples_size, double* ranges, int element_count,
						int numerics_size, double* result)
		{
			//m_dispatch->clearArgs();
			//FIX_ARGS_STABLE(m_dispatch);
			//size_t workgroup_size = m_context->GetMaxWorkgroup();
			m_global_args.input = input;
			m_global_args.samples = samples;
			m_global_args.ranges = ranges;
			m_global_args.result = result;
			m_global_args.element_count = element_count;
			m_global_args.numerics_size = numerics_size;

			/*m_dispatch->pushPointerArg((void*)input);
			m_dispatch->pushPointerArg((void*)samples);
			m_dispatch->pushPointerArg((void*)ranges);
			m_dispatch->pushPointerArg((void*)result);
			m_dispatch->pushIntArg(element_count);
			m_dispatch->pushIntArg(numerics_size);
			*/
	//		size_t global_dims[3] = {samples_size,1,1};
	//		size_t local_dims[3] = {workgroup_size,1,1};

			hsa_signal_t signal;
			Launch_params_t lp1 {.ndim=2, .gdims={samples_size,std::max(1,element_count/64) }, .ldims={256,64}};
			//Launch_params_t lp1 {.ndim=2, .gdims={element_count/16,samples_size }, .ldims={16,256}};
			m_dispatch->dispatchKernel(m_global_args, signal, lp1);
			m_dispatch->waitComplete(signal);
			//m_dispatch->setLaunchAttributes(1, global_dims,  local_dims);
			//m_dispatch->dispatchKernelWaitComplete();
		}
private:
		std::shared_ptr<HSAContext> m_context;
		TemplateDispatch<GlobalArgs>* m_dispatch;
		HSAContext::Kernel* m_kernel;
};



#endif /* DISTANCE_HPP_ */


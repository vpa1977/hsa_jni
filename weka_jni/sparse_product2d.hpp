#ifndef SPARSE_PRODUCT_2D_HPP_
#define SPARSE_PRODUCT_2D_HPP_


#include "HSAContextImpl.hpp"
#include <vector>
#include <math.h>
#include <algorithm>
#include <string.h>



/**
 *
 */
class SparseProduct2D
{
public:

	struct SparseProductArgs2D
	{

		long global_offset_0;
		long global_offset_1;
		long global_offset_2;
		long* printf_buffer;
		long* vqueue_pointer;
		long* aqlwrap_pointer;
		double* result;
	    int classIndex;
		double* weights;
		void**  weights_indices2d;
		void** input_iter2d;
		int* length2d;

	} __attribute__ ((aligned (16))) ;

	SparseProductArgs2D m_args;

	SparseProduct2D(std::shared_ptr<HSAContext> context, const std::string& dump)
	{
		m_local_kernel = context->createKernel(dump.c_str(), "&__OpenCL_run_kernel");
		m_p_template = new TemplateDispatch<SparseProductArgs2D>(m_local_kernel);
		memset(&m_args, 0, sizeof(m_args));
	//	m_local_dispatch = context->createDispatch(m_local_kernel);
	}
	virtual ~SparseProduct2D()
	{
		//delete m_local_dispatch;
		//m_local_dispatch = NULL;
		delete m_p_template; m_p_template = 0;
		delete m_local_kernel; m_local_kernel = 0;
	}

public:


	void product(size_t tuple_count, int classIndex, double **values, int ** indices, double* weights, int* size2d, double* tmp)
	{
		hsa_signal_t signal;
		size_t num_wg = 1;
		size_t num_compute_units = 6;

		size_t workgroup_size = 256;
		size_t global_size = num_wg * num_compute_units;

		Launch_params_t lp {.ndim=2, .gdims={global_size*workgroup_size, tuple_count}, /*.ldims={workgroup_size, 1}*/};
		m_args.classIndex = classIndex;
		m_args.input_iter2d = (void**)values;
		m_args.weights_indices2d = (void**)indices;
		m_args.weights = weights;
		m_args.length2d = size2d;
		m_args.result = tmp;
		m_p_template->dispatchKernel(m_args, signal, lp);
		m_p_template->waitComplete(signal);

		for (int i = 0 ;i < tuple_count ; ++i)
		{
			if ((size_t)size2d[i] < workgroup_size)
			{
				tmp[i] =  tmp[global_size * i];
			}
			else
				tmp[i] =  reduceTail(&tmp[global_size*i],size2d[i]);
		}
	}

	/*

	double product(int classIndex, double *values, int * indices, double* weights, int size)
	{

		size_t num_wg = 64;
		size_t num_compute_units = 6;
		size_t global_size = num_wg * num_compute_units;
		size_t workgroup_size = 256;
		m_result.resize( global_size);

		m_local_dispatch->clearArgs();
		FIX_ARGS_STABLE(m_local_dispatch);
		m_local_dispatch->pushIntArg(classIndex);
		m_local_dispatch->pushPointerArg((void*)weights);
		m_local_dispatch->pushPointerArg((void*)indices);
		m_local_dispatch->pushPointerArg((void*)values);
		m_local_dispatch->pushIntArg((int)size );
		m_local_dispatch->pushPointerArg((void*)&m_result[0]);
		size_t global_dims[3] = { std::min(roundUp(size, workgroup_size), global_size*workgroup_size),1,1};
		size_t local_dims[3] = {workgroup_size,1,1};
		m_local_dispatch->setLaunchAttributes(1, global_dims,  local_dims);
		m_local_dispatch->dispatchKernelWaitComplete();
		if (size < workgroup_size)
		{
			return  m_result[0];
		}
		return reduceTail(size);
	}
	*/
protected:

	size_t roundUp(size_t size, size_t wg_size) {
	  size_t times = size / wg_size;
	  size_t rem = size % wg_size;
	  if (rem != 0) ++times;
	  return times * wg_size;
	}


	double reduceTail(double*tmp , size_t size)
	{
		double result = 0;
		size_t num_wg = 64;
		size_t workgroup_size = 256;
		size_t ceilNumWg = (size_t)ceil(((double)size)/workgroup_size);
		int numTailReduce = std::min( ceilNumWg, num_wg );
		for (int i = 0; i < numTailReduce ; ++i)
			result += tmp[i];
		return result;
	}

private:
	TemplateDispatch<SparseProductArgs2D>* m_p_template;
	std::vector<double> m_result;
	std::shared_ptr<HSAContext> m_context;
	HSAContext::Kernel* m_local_kernel;
};


#endif /* MERGE_SORT_HPP_ */

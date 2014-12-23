/*
 * merge_sort.hpp
 *
 *  Created on: 27/10/2014
 *      Author: bsp
 */

#ifndef MERGE_SORT_HPP_
#define MERGE_SORT_HPP_

#include "HSAContext.h"
#include "HSAContextImpl.hpp"
#include <vector>
#include <math.h>
#include <algorithm>
#include <string.h>

class MergeSort
{
public:

	struct LocalArgs
		{
			long global_offset_0;
			long global_offset_1;
			long global_offset_2;
			long* printf_buffer;
			long* vqueue_pointer;
			long* aqlwrap_pointer;
			int vecSize;
			double * data;
			int * offsets;
		} __attribute__ ((aligned (16))) ;

		LocalArgs m_local_args;

		struct GlobalArgs
		{
			long global_offset_0;
			long global_offset_1;
			long global_offset_2;
			long* printf_buffer;
			long* vqueue_pointer;
			long* aqlwrap_pointer;
			int vecSize;
			int srcLogicalBlockSize;
			double * data;
			double * data_output;
			int * offsets;
			int * offsets_output;
		} __attribute__ ((aligned (16)));

		GlobalArgs m_global_args;
	MergeSort(std::shared_ptr<HSAContext> context, const std::string& local_merge_brig, const std::string& global_merge_brig)
	{
		m_local_kernel = context->createKernel(local_merge_brig.c_str(), "&__OpenCL_run_kernel");
		m_local_dispatch = new TemplateDispatch<LocalArgs>(m_local_kernel);

		m_global_kernel = context->createKernel(global_merge_brig.c_str(), "&__OpenCL_run_kernel");
		m_global_dispatch = new TemplateDispatch<GlobalArgs>(m_global_kernel);
		m_context = context;

		memset(&m_local_args ,0, sizeof(m_local_args));
		memset(&m_global_args,0, sizeof(m_global_args));
	}
	virtual ~MergeSort()
	{
		delete m_local_dispatch; m_local_dispatch = NULL;
		delete m_global_dispatch; m_global_dispatch = NULL;
		delete m_local_kernel; m_local_kernel = NULL;
		delete m_global_kernel; m_global_kernel = NULL;
	}

public:

	void sort(double* data, int* offsets, int size)
	{
		size_t vecSize = size;
		size_t globalRange = vecSize;
		size_t localRange = 256;
		size_t modlocalRange = (globalRange & (localRange - 1));
		if (modlocalRange > 0) {
			globalRange &= ~modlocalRange;
			globalRange += localRange;
		}

		do_local_sort(offsets,data, vecSize, globalRange, localRange);

		if ( vecSize <= localRange)
			return;

		do_global_merge(offsets, data, vecSize, globalRange, localRange);

	}


	void sort(std::vector<double>& data, std::vector<int>& offsets)
	{
		offsets.resize( data.size() );
		size_t vecSize = data.size();
		size_t globalRange = vecSize;
		size_t localRange = 256;
		size_t modlocalRange = (globalRange & (localRange - 1));
		if (modlocalRange > 0) {
			globalRange &= ~modlocalRange;
			globalRange += localRange;
		}

		do_local_sort(offsets,data, vecSize, globalRange, localRange);

		if ( vecSize <= localRange)
			return;

		do_global_merge(offsets, data, vecSize, globalRange, localRange);
	}
private:





	void do_local_sort(int* offsets, double* data, int vecSize, size_t global_size, size_t local_size)
	{
		m_local_args.vecSize = vecSize;
		m_local_args.data = data;
		m_local_args.offsets = offsets;
		Launch_params_t lp1 {.ndim=1, .gdims={global_size}, .ldims={local_size}};
		hsa_signal_t signal;
		m_local_dispatch->dispatchKernel(m_local_args, signal, lp1);
		m_local_dispatch->waitComplete(signal);
	}

	void do_global_merge(int* offsets, double* data, int vecSize, size_t global_size, size_t localRange)
	{
		std::vector<double> tmp;
		tmp.resize(vecSize);
		std::vector<int> offsets2;
		offsets2.resize(vecSize);

		// An odd number of elements requires an extra merge pass to sort
		int numMerges = 0;
		// Calculate the log2 of vecSize, taking into account our block size
		// from kernel 1 is 256
		// this is how many merge passes we want
		int log2BlockSize = vecSize >> 8;

		for (; log2BlockSize > 1; log2BlockSize >>= 1) {
			++numMerges;
		}
		// Check to see if the input vector size is a power of 2, if not we will
		// need last merge pass
		int vecPow2 = (vecSize & (vecSize - 1));
		numMerges += vecPow2 > 0 ? 1 : 0;

		for (int pass = 1; pass <= numMerges; ++pass) {
			int srcLogicalBlockSize = localRange << (pass - 1);

			m_global_args.vecSize = vecSize;
			m_global_args.srcLogicalBlockSize = srcLogicalBlockSize;

			if ((pass & 0x1) > 0) {
				m_global_args.data= data;
				m_global_args.data_output = &tmp[0];
				m_global_args.offsets = offsets;
				m_global_args.offsets_output = &offsets2[0];

			} else {

				m_global_args.data= &tmp[0];
				m_global_args.data_output = data;
				m_global_args.offsets = &offsets2[0];
				m_global_args.offsets_output = offsets;
			}

			Launch_params_t lp1 {.ndim=1, .gdims={global_size}, .ldims={localRange}};
			hsa_signal_t signal;
			m_local_dispatch->dispatchKernel(m_local_args, signal, lp1);
			m_local_dispatch->waitComplete(signal);
		}

		if ((numMerges & 1) > 0)
		{
			memcpy((void*)data,   (void*) &tmp[0], tmp.size() * sizeof(double));
			memcpy(offsets, &offsets2[0], tmp.size() * sizeof(int));
		}
	}



	void do_local_sort(std::vector<int>& offsets, std::vector<double>& data, int vecSize, size_t global_size, size_t local_size)
	{
		m_local_args.vecSize = vecSize;
		m_local_args.data = &data[0];
		m_local_args.offsets = &offsets[0];
		Launch_params_t lp1 {.ndim=1, .gdims={global_size}, .ldims={local_size}};
		hsa_signal_t signal;
		m_local_dispatch->dispatchKernel(m_local_args, signal, lp1);
		m_local_dispatch->waitComplete(signal);
	}

	void do_global_merge(std::vector<int>& offsets, std::vector<double>& data, int vecSize, size_t global_size, size_t localRange)
	{
		std::vector<double> tmp;
		tmp.resize(vecSize);
		std::vector<int> offsets2;
		offsets2.resize(vecSize);

		// An odd number of elements requires an extra merge pass to sort
		int numMerges = 0;
		// Calculate the log2 of vecSize, taking into account our block size
		// from kernel 1 is 256
		// this is how many merge passes we want
		int log2BlockSize = vecSize >> 8;

		for (; log2BlockSize > 1; log2BlockSize >>= 1) {
			++numMerges;
		}
		// Check to see if the input vector size is a power of 2, if not we will
		// need last merge pass
		int vecPow2 = (vecSize & (vecSize - 1));
		numMerges += vecPow2 > 0 ? 1 : 0;

		for (int pass = 1; pass <= numMerges; ++pass) {
			int srcLogicalBlockSize = localRange << (pass - 1);
			m_global_args.vecSize = vecSize;
			m_global_args.srcLogicalBlockSize = srcLogicalBlockSize;

			if ((pass & 0x1) > 0) {
				m_global_args.data= &data[0];
				m_global_args.data_output = &tmp[0];
				m_global_args.offsets = &offsets[0];
				m_global_args.offsets_output = &offsets2[0];

			} else {

				m_global_args.data= &tmp[0];
				m_global_args.data_output = &data[0];
				m_global_args.offsets = &offsets2[0];
				m_global_args.offsets_output = &offsets[0];
			}

			Launch_params_t lp1 {.ndim=1, .gdims={global_size}, .ldims={localRange}};
			hsa_signal_t signal;
			m_local_dispatch->dispatchKernel(m_local_args, signal, lp1);
			m_local_dispatch->waitComplete(signal);
		}

		if ((numMerges & 1) > 0)
		{
			data.swap(tmp);
			offsets.swap(offsets2);
		}


	}
private:
	std::shared_ptr<HSAContext> m_context;
	HSAContext::Kernel* m_local_kernel;
	HSAContext::Kernel* m_global_kernel;
	TemplateDispatch<LocalArgs>* m_local_dispatch;
	TemplateDispatch<GlobalArgs>* m_global_dispatch;
};


#endif /* MERGE_SORT_HPP_ */

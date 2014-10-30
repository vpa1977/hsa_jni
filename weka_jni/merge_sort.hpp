/*
 * merge_sort.hpp
 *
 *  Created on: 27/10/2014
 *      Author: bsp
 */

#ifndef MERGE_SORT_HPP_
#define MERGE_SORT_HPP_

#include "HSAContext.h"
#include <vector>
#include <math.h>
#include <algorithm>
#include <string.h>

class MergeSort
{
public:
	MergeSort(HSAContext* context, const std::string& local_merge_brig, const std::string& global_merge_brig)
	{
		m_local_kernel = context->createKernel(local_merge_brig.c_str(), "&__OpenCL_run_kernel");
		m_local_dispatch = context->createDispatch(m_local_kernel);

		m_global_kernel = context->createKernel(global_merge_brig.c_str(), "&__OpenCL_run_kernel");
		m_global_dispatch = context->createDispatch(m_global_kernel);

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
		size_t localRange = 512;
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
		size_t localRange = 512;
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
		m_local_dispatch->clearArgs();
		FIX_ARGS_STABLE(m_local_dispatch);

		m_local_dispatch->pushIntArg(vecSize);
		m_local_dispatch->pushPointerArg(data);
		m_local_dispatch->pushPointerArg(offsets);

		size_t global_dims[3] = {global_size,1,1};
		size_t local_dims[3] = {local_size,1,1};
		m_local_dispatch->setLaunchAttributes(1, global_dims,  local_dims);
		m_local_dispatch->dispatchKernelWaitComplete();
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
			m_global_dispatch->clearArgs();
			FIX_ARGS_STABLE(m_global_dispatch);

			m_global_dispatch->pushIntArg(vecSize);
			m_global_dispatch->pushIntArg(srcLogicalBlockSize);

			if ((pass & 0x1) > 0) {
				m_global_dispatch->pushPointerArg(data);
				m_global_dispatch->pushPointerArg(&tmp[0]);
				m_global_dispatch->pushPointerArg(offsets);
				m_global_dispatch->pushPointerArg(&offsets2[0]);

			} else {
				m_global_dispatch->pushPointerArg(&tmp[0]);
				m_global_dispatch->pushPointerArg(data);
				m_global_dispatch->pushPointerArg(&offsets2[0]);
				m_global_dispatch->pushPointerArg(offsets);
			}
			size_t global_dims[3] = {global_size,1,1};
			size_t local_dims[3] = {localRange,1,1};
			m_global_dispatch->setLaunchAttributes(1, global_dims,  local_dims);
			m_global_dispatch->dispatchKernelWaitComplete();
		}

		if ((numMerges & 1) > 0)
		{
			memcpy((void*)data,   (void*) &tmp[0], tmp.size() * sizeof(double));
			memcpy(offsets, &offsets2[0], tmp.size() * sizeof(int));
		}
	}



	void do_local_sort(std::vector<int>& offsets, std::vector<double>& data, int vecSize, size_t global_size, size_t local_size)
	{
		m_local_dispatch->clearArgs();
		FIX_ARGS_STABLE(m_local_dispatch);

		m_local_dispatch->pushIntArg(vecSize);
		m_local_dispatch->pushPointerArg(&data[0]);
		m_local_dispatch->pushPointerArg(&offsets[0]);

		size_t global_dims[3] = {global_size,1,1};
		size_t local_dims[3] = {local_size,1,1};
		m_local_dispatch->setLaunchAttributes(1, global_dims,  local_dims);
		m_local_dispatch->dispatchKernelWaitComplete();
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
			m_global_dispatch->clearArgs();
			FIX_ARGS_STABLE(m_global_dispatch);

			m_global_dispatch->pushIntArg(vecSize);
			m_global_dispatch->pushIntArg(srcLogicalBlockSize);

			if ((pass & 0x1) > 0) {
				m_global_dispatch->pushPointerArg(&data[0]);
				m_global_dispatch->pushPointerArg(&tmp[0]);
				m_global_dispatch->pushPointerArg(&offsets[0]);
				m_global_dispatch->pushPointerArg(&offsets2[0]);

			} else {
				m_global_dispatch->pushPointerArg(&tmp[0]);
				m_global_dispatch->pushPointerArg(&data[0]);
				m_global_dispatch->pushPointerArg(&offsets2[0]);
				m_global_dispatch->pushPointerArg(&offsets[0]);
			}
			size_t global_dims[3] = {global_size,1,1};
			size_t local_dims[3] = {localRange,1,1};
			m_global_dispatch->setLaunchAttributes(1, global_dims,  local_dims);
			m_global_dispatch->dispatchKernelWaitComplete();
		}

		if ((numMerges & 1) > 0)
		{
			data.swap(tmp);
			offsets.swap(offsets2);
		}


	}
private:
	HSAContext::Kernel* m_local_kernel;
	HSAContext::Kernel* m_global_kernel;
	HSAContext::Dispatch* m_local_dispatch;
	HSAContext::Dispatch* m_global_dispatch;
};


#endif /* MERGE_SORT_HPP_ */

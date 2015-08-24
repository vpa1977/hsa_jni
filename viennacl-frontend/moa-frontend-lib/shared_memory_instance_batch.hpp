#ifndef SHARED_MEMORY_INSTANCE_BATCH_HPP
#define SHARED_MEMORY_INSTANCE_BATCH_HPP


#include <viennacl/context.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/compressed_matrix.hpp>
#include "instance_interface.hpp"

#ifdef VIENNACL_WITH_OPENCL20
#include <CL/cl.h>
#else

#endif

struct shared_memory_instance_batch
{
	shared_memory_instance_batch(size_t num_rows, size_t num_columns, viennacl::context& ctx) :
		m_num_rows(num_rows), m_num_columns(num_columns),
		m_index(0), m_instance_values(num_rows*num_columns), m_gpu_instance_values(num_rows*num_columns), m_class_values(num_rows), m_ctx(ctx)
	{
		cl_int err;
		const cl_context& cl_ctx = ctx.opencl_context().handle().get();
		m_ptr = clSVMAlloc(cl_ctx, CL_MEM_READ_WRITE, num_rows*num_columns * sizeof(double), sizeof(double));


		clear();

	}
	virtual ~shared_memory_instance_batch()
	{
		const cl_context& cl_ctx = m_ctx.opencl_context().handle().get();
		clSVMFree(cl_ctx, m_ptr);
	}

	void clear()
	{
		m_class_values.clear();
		m_class_values.resize(m_num_rows);
		m_instance_values.reserve(m_num_rows);
		m_index = 0;
	}

	bool add(dense_storage* ptr)
	{
		if (m_index < m_class_values.size())
		{
			m_class_values.at(m_index) = ptr->m_class_value;
			std::copy(ptr->m_values.begin(), ptr->m_values.end(), m_instance_values.begin() + m_num_columns*m_index);
			++m_index;
		}
		return m_index == m_class_values.size();
	}

	void commit()
	{

			viennacl::copy(m_instance_values, m_gpu_instance_values);
	}

	void* m_ptr;
	cl_mem memory_handle;

	size_t m_num_rows;
	size_t m_num_columns;
	size_t m_index;
	std::vector<double> m_instance_values;
        viennacl::vector<double> m_gpu_instance_values;
	std::vector<double> m_class_values;

	viennacl::context& m_ctx;
};

#endif
/*
 * native_instance_batch.hpp
 *
 *  Created on: 25/05/2015
 *      Author: bsp
 */

#ifndef SPARSE_INSTANCE_BATCH_HPP_
#define SPARSE_INSTANCE_BATCH_HPP_

#include <viennacl/context.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/compressed_matrix.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

struct sparse_instance_batch
{
	sparse_instance_batch(size_t num_rows, size_t num_columns, viennacl::context& ctx ) :
		m_num_rows(num_rows), m_num_columns(num_columns),
		m_index(0), m_instance_values(num_rows, num_columns, ctx), m_class_values(num_rows, ctx)
	{
		clear();
	}

	void clear()
	{
		m_individual_instances.clear();
		m_index = 0;
		m_individual_instances.resize(m_num_rows);
	}

	bool add(sparse_storage* ptr)
	{
		if (m_index < m_individual_instances.size())
		{
			sparse_storage storage = *ptr;
			m_individual_instances[m_index] = storage;
			++m_index;
		}
		return m_index  == m_individual_instances.size();
	}

	void commit();


	size_t m_num_rows;
	size_t m_num_columns;
	size_t m_index;
	std::vector<sparse_storage> m_individual_instances;
	viennacl::compressed_matrix<double> m_instance_values;
	viennacl::vector<double> m_class_values;

};

struct dense_instance_batch
{
	dense_instance_batch(size_t num_rows, size_t num_columns, viennacl::context& ctx ) :
		m_num_rows(num_rows), m_num_columns(num_columns),
		m_index(0), m_instance_values(num_rows*num_columns), m_gpu_instance_values(num_rows*num_columns), m_class_values(num_rows), m_ctx(ctx)
	{
		clear();
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
		return m_index  == m_class_values.size();
	}

	void commit()
	{
		if (m_gpu_instance_values.handle().get_active_handle_id() == viennacl::OPENCL_MEMORY)
			write_with_data_queue(m_gpu_instance_values.handle().opencl_handle().get(), &m_instance_values[0], m_instance_values.size() * sizeof(m_instance_values[0]));
		else
			viennacl::copy(m_instance_values, m_gpu_instance_values);
	}


	size_t m_num_rows;
	size_t m_num_columns;
	size_t m_index;
	std::vector<double> m_instance_values;
	std::vector<double> m_class_values;
	viennacl::vector<double> m_gpu_instance_values;
	viennacl::context& m_ctx;

};


#endif /* SPARSE_INSTANCE_BATCH_HPP_ */

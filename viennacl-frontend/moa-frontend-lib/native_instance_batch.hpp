/*
 * native_instance_batch.hpp
 *
 *  Created on: 25/05/2015
 *      Author: bsp
 */

#ifndef NATIVE_INSTANCE_BATCH_HPP_
#define NATIVE_INSTANCE_BATCH_HPP_

#include <viennacl/context.hpp>
#include <viennacl/compressed_matrix.hpp>


struct native_instance_batch
{
	native_instance_batch(size_t num_rows, size_t num_columns, viennacl::context& ctx ) :
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
			m_individual_instances.at(m_index) = *ptr;
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


#endif /* NATIVE_INSTANCE_BATCH_HPP_ */

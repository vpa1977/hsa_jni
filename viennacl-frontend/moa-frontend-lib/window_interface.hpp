/*
 * window_interface.hpp
 * Interface to the Window class
 *  Created on: 27/04/2015
 *      Author: bsp
 */

#ifndef WINDOW_INTERFACE_HPP_
#define WINDOW_INTERFACE_HPP_

#include <jni.h>
#include <vector>
#include "instance_interface.hpp"

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

template <class ublas_matrix, class ublas_vector>
class native_window
{
public:
	native_window(JNIEnv* env, jobject window) : env_(env), window_(window)
	{
		static bool init = false;
		static jclass window_class;
		static jmethodID get_values_method_id;
		static jmethodID get_rows_method_id;
		static jmethodID get_columns_method_id;
		static jmethodID get_classes_method_id;
		static jmethodID get_row_count_method_id;
		static jmethodID get_column_count_method_id;

		if (!init)
		{
			window_class = env->FindClass("org/moa/gpu/Window");
			get_values_method_id = env->GetMethodID(window_class, "getValues", "()[D");
			get_rows_method_id = env->GetMethodID(window_class, "getRowIndices", "()[I");
			get_columns_method_id = env->GetMethodID(window_class, "getColumnIndices", "()[I");
			get_classes_method_id = env->GetMethodID(window_class, "getClassValues", "()[D");
			get_row_count_method_id = env->GetMethodID(window_class, "getRowCount", "()I");
			get_column_count_method_id = env->GetMethodID(window_class, "getColumnCount", "()I");
			init = true;
		}
		jboolean no_copy = false;
		jdoubleArray value_list = (jdoubleArray)env_->CallObjectMethod(window_, get_values_method_id);
		jintArray rows_list = (jintArray)env_->CallObjectMethod(window, get_rows_method_id);
		jintArray column_list = (jintArray)env_->CallObjectMethod(window, get_columns_method_id);
		jdoubleArray classes_list = (jdoubleArray)env_->CallObjectMethod(window, get_classes_method_id);

		jsize element_count = env_->GetArrayLength(rows_list);
		jsize classes_count = env_->GetArrayLength(classes_list);

		int rows_count = env_->CallIntMethod(window, get_row_count_method_id);
		int columns_count = env_->CallIntMethod(window, get_column_count_method_id);

		m_value_matrix = ublas_matrix(rows_count,columns_count);
		m_class_vector = ublas_vector(classes_count);

		double * classes = reinterpret_cast<double*>(env_->GetPrimitiveArrayCritical(classes_list, &no_copy));
		double * values = reinterpret_cast<double*>(env_->GetPrimitiveArrayCritical(value_list, &no_copy));
		int* rows = reinterpret_cast<int*>(env_->GetPrimitiveArrayCritical(rows_list, &no_copy));
		int* columns = reinterpret_cast<int*>(env_->GetPrimitiveArrayCritical(column_list, &no_copy));

		std::vector<int> unique_columns(columns_count);
		for (int i = 0;i < element_count ; ++i)
		{
			m_value_matrix( rows[i], columns[i]) = values[i];
			unique_columns[columns[i]] = 1;
		}

		int accumulator = 0;
		std::for_each(unique_columns.begin(), unique_columns.end(), [&accumulator](int a){
			++accumulator;
		});

		m_unique_columns.resize(accumulator);
		accumulator = 0;
		for (int i = 0 ; i < columns_count ; ++i)
		{
			if (unique_columns[i]> 0)
			{
				m_unique_columns[accumulator++] = i;
			}
		}


		for (int i = 0;i < classes_count ; ++i)
			m_class_vector(i) = classes[i];


		env_->ReleasePrimitiveArrayCritical(value_list, values, JNI_ABORT);
		env_->ReleasePrimitiveArrayCritical(rows_list, rows, JNI_ABORT);
		env_->ReleasePrimitiveArrayCritical(column_list, columns, JNI_ABORT);
		env_->ReleasePrimitiveArrayCritical(classes_list, classes, JNI_ABORT);


	}
private:
	ublas_matrix m_value_matrix;
	ublas_vector m_class_vector;
	std::vector<int> m_unique_columns;
	JNIEnv* env_;
	jobject window_;
};


template <class ublas_matrix, class ublas_vector>
class direct_memory_window
{
public:
	direct_memory_window(JNIEnv* env, jobject window) : env_(env), window_(window)
	{
		static bool init = false;
		static jclass window_class;
		static jmethodID get_rows_method_id;
		static jmethodID get_classes_method_id;
		static jmethodID get_row_count_method_id;
		static jmethodID get_column_count_method_id;
		static jmethodID get_element_count_method_id;

		if (!init)
		{
			window_class = env->FindClass("org/moa/gpu/SimpleDirectMemoryWindow");
			get_rows_method_id = env->GetMethodID(window_class, "rowHandle", "()J");
			get_row_count_method_id = env->GetMethodID(window_class, "getRowCount", "()I");
			get_column_count_method_id = env->GetMethodID(window_class, "getColumnCount", "()I");
			get_element_count_method_id = env->GetMethodID(window_class, "getElementCount", "()J");
			get_classes_method_id = env->GetMethodID(window_class, "classesHandle", "()J");
			init = true;
		}


		int rows_count = env_->CallIntMethod(window, get_row_count_method_id);
		int columns_count = env_->CallIntMethod(window, get_column_count_method_id);
		m_value_matrix = ublas_matrix(rows_count,columns_count);
		m_class_vector = ublas_vector(rows_count);

		long ** elements = (long**)env_->CallLongMethod(window, get_element_count_method_id);
		double ** values = (double**)env_->CallLongMethod(window, get_rows_method_id);
		double * class_values = (double*)env_->CallLongMethod(window, get_classes_method_id);

		for (int row = 0; row < rows_count ; ++ row)
		{
			long* indices = elements[row];
			double * att_values = values[row];
			long num_atts = indices[0];
			for (int att = 0; att < num_atts; ++att)
			{
				m_value_matrix(row, indices[att+1] )=  att_values[att];
			}
			m_class_vector(row) = class_values[row];
		}
	}

	ublas_matrix& values()
	{
		return m_value_matrix;
	}
	ublas_vector& classes()
	{
		return m_class_vector;
	}
private:
	ublas_matrix m_value_matrix;
	ublas_vector m_class_vector;

	JNIEnv* env_;
	jobject window_;
};



/*
 * obtain vector of instances in the sliding window
 */
std::vector<instance_interface> get_window(JNIEnv* env, jobject window);




/**
 * convert instance vector into matrix of attribute values and a vector of class values.
 * the vector expects both matrix and vector to be of correct size
 */
template <typename ublas_matrix_type>
void instances_to_matrix(std::vector<instance_interface>& instances,
		ublas_matrix_type& values_matrix,
		boost::numeric::ublas::vector<double>& classes_vector)
{
	for (int row = 0; row < instances.size(); ++row) {
		instance_interface& instance = instances.at(row);
		int num_attributes = instance.get_num_attributes();
		int class_index = instance.get_class_index();
		if (classes_vector.size())
			classes_vector[row] = instance.get_attribute(class_index);
		for (int i = 0; i < num_attributes; ++i) {
			int column = i < class_index ? i : i - 1;
			if (!instance.is_missing(i) && i != class_index)
				values_matrix(row, column) = instance.get_attribute(i);
		}
	};
}


#endif /* WINDOW_INTERFACE_HPP_ */

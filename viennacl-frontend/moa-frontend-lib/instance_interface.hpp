/*
 * instance_interface.hpp
 *
 *  Created on: 27/04/2015
 *      Author: bsp
 */

#ifndef INSTANCE_INTERFACE_HPP_
#define INSTANCE_INTERFACE_HPP_

#include <jni.h>
#include <org_moa_gpu_bridge_NativeSparseInstance.h>
#include <vector>
#include <viennacl/vector.hpp>


class dataset_interface
{
	public:
		dataset_interface(JNIEnv* env, jobject dataset);
		int get_class_index();
		int get_num_attributes();
	private:
		JNIEnv* env_;
		jobject instance_;
	private:
		jmethodID class_index_method_;
		jmethodID num_attributes_method_;

};

class instance_interface
{
public:
	instance_interface(JNIEnv* env,jobject inst);
	double get_attribute(int index);
	int get_class_index();
	bool is_missing( int index);
	int get_num_attributes();
	std::vector<int> get_indices();
	std::vector<double> get_values();
private:
	JNIEnv* env_;
	jobject instance_;
private:
	jmethodID get_indices_method_;
	jmethodID get_values_method_;
	jmethodID get_method_;
	jmethodID class_index_method_;
	jmethodID is_missing_method_;
	jmethodID num_attributes_method_;

};

struct sparse_storage
{

	viennacl::vector<double> vector(size_t num_attributes);

	double m_class_value;
	std::vector<int> m_indices;
	std::vector<double> m_values;
};






#endif /* INSTANCE_INTERFACE_HPP_ */

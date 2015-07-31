/*
 * instance_interface.cpp
 *
 *  Created on: 27/04/2015
 *      Author: bsp
 */
#include "library.hpp"
#include "instance_interface.hpp"
#include <assert.h>
#include <string.h>
#include <boost/numeric/ublas/vector.hpp>


viennacl::vector<double> sparse_storage::vector(size_t size)
{
	viennacl::vector<double> data(size, get_global_context());
	std::vector<double> cpu_vec(size);
	for (size_t i = 0 ;i < m_indices.size() ; ++i)
		cpu_vec.at( m_indices.at(i)) = m_values.at(i);
	viennacl::copy( cpu_vec, data);
	return data;
}



instance_interface::instance_interface(JNIEnv* env, jobject inst) : env_(env), instance_(inst)
{
	static jclass instance_class = env->FindClass("org/moa/gpu/AccessibleSparseInstance");
	assert(instance_class != 0);
	static jmethodID get_method = env->GetMethodID(instance_class, "value", "(I)D");
	assert(get_method != 0);
	get_method_ = get_method;
	static jmethodID class_index_method = env->GetMethodID(instance_class, "classIndex", "()I");
	class_index_method_= class_index_method;

	static jmethodID is_missing_method = env->GetMethodID(instance_class, "isMissing","(I)Z");
	is_missing_method_ = is_missing_method;

	static jmethodID num_attributes_method = env->GetMethodID(instance_class, "numAttributes", "()I");
	num_attributes_method_ = num_attributes_method;

	static jclass sparse_instance_class = env->FindClass("org/moa/gpu/AccessibleSparseInstance");

	static jclass dense_instance_class = env->FindClass("weka/core/DenseInstance");

	static jmethodID get_indices_method = env->GetMethodID( sparse_instance_class,"getIndices", "()[I");
	get_indices_method_ = get_indices_method;

	static jmethodID get_values_method = env->GetMethodID( sparse_instance_class, "getValues", "()[D");
	get_values_method_ = get_values_method;

	static jmethodID get_double_array_method = env->GetMethodID(dense_instance_class,"toDoubleArray", "()[D");
	get_double_array_method_ = get_double_array_method;

}

std::vector<double> instance_interface::to_double_array()
{
	std::vector<double> ret;
	jdoubleArray array = (jdoubleArray)env_->CallObjectMethod(instance_, get_double_array_method_);
	size_t len = env_->GetArrayLength(array);
	ret.resize(len);
	double * ptr = (double*)env_->GetPrimitiveArrayCritical(array, 0);
	memcpy(&ret[0], ptr, len* sizeof(double));
	env_->ReleasePrimitiveArrayCritical(array, ptr, JNI_ABORT );
	return ret;

}

std::vector<double> instance_interface::get_values()
{
	std::vector<double> ret;
	jdoubleArray array = (jdoubleArray)env_->CallObjectMethod(instance_, get_values_method_);
	size_t len = env_->GetArrayLength(array);
	ret.resize(len);
	double * ptr = (double*)env_->GetPrimitiveArrayCritical(array, 0);
	memcpy(&ret[0], ptr, len* sizeof(double));
	env_->ReleasePrimitiveArrayCritical(array, ptr, JNI_ABORT );
	return ret;
}

std::vector<int> instance_interface::get_indices()
{
	std::vector<int> ret;
	jintArray array = (jintArray)env_->CallObjectMethod(instance_, get_indices_method_);
	size_t len = env_->GetArrayLength(array);
	ret.resize(len);
	int * ptr = (int*)env_->GetPrimitiveArrayCritical(array, 0);
	memcpy(&ret[0], ptr, len* sizeof(int));
	env_->ReleasePrimitiveArrayCritical(array, ptr, JNI_ABORT );
	return ret;

}

double instance_interface::get_attribute(int index)
{
	return env_->CallDoubleMethod(instance_, get_method_, index);
}

int instance_interface::get_class_index()
{
	return env_->CallIntMethod(instance_, class_index_method_);
}

int instance_interface::get_num_attributes()
{
	return env_->CallIntMethod(instance_, num_attributes_method_);
}

bool instance_interface::is_missing( int index)
{
	return env_->CallBooleanMethod(instance_, is_missing_method_, index);
}


dataset_interface::dataset_interface(JNIEnv* env, jobject inst) : env_(env), instance_(inst)
{
	static jclass instance_class = env->FindClass("weka/core/Instances");
	assert(instance_class != 0);

	static jmethodID class_index_method = env->GetMethodID(instance_class, "classIndex", "()I");
	class_index_method_= class_index_method;


	static jmethodID num_attributes_method = env->GetMethodID(instance_class, "numAttributes", "()I");
	num_attributes_method_ = num_attributes_method;
}

int dataset_interface::get_class_index()
{
	return env_->CallIntMethod(instance_, class_index_method_);
}

int dataset_interface::get_num_attributes()
{
	return env_->CallIntMethod(instance_, num_attributes_method_);
}




///// Native Sparse Instance
/*
 * Class:     org_moa_gpu_bridge_NativeSparseInstance
 * Method:    writeToNative
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeSparseInstance_writeToNative(JNIEnv * env, jobject native_sparse_instance)
{
	static jclass _class = env->FindClass("org/moa/gpu/bridge/NativeSparseInstance");
	static jfieldID _context_field = env->GetFieldID(_class, "m_native_context", "J");
	static jfieldID _instance_field = env->GetFieldID(_class, "m_instance", "Lorg/moa/gpu/AccessibleSparseInstance;");
	sparse_storage* storage = (sparse_storage*)env->GetLongField(native_sparse_instance, _context_field);
	jobject instance = env->GetObjectField(native_sparse_instance,_instance_field);
	instance_interface iface(env, instance);

	int class_index = iface.get_class_index();
	std::vector<double> values =iface.get_values();
	std::vector<int> indices = iface.get_indices();
	// reserve memory
	storage->m_values.reserve(values.size());
	storage->m_indices.reserve(values.size());

	for (size_t i = 0 ; i < values.size();++i )
	{
		if (indices.at(i) == class_index)
			storage->m_class_value = values.at(i);
		else
		{
			storage->m_values.push_back(values.at(i));
			int new_index = indices.at(i) > class_index ? indices.at(i) - 1 : indices.at(i);
			storage->m_indices.push_back(new_index);
		}
	}

}

/*
 * Class:     org_moa_gpu_bridge_NativeSparseInstance
 * Method:    release
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeSparseInstance_release(JNIEnv * env, jobject native_sparse_instance)
{
	static jclass _class = env->FindClass("org/moa/gpu/bridge/NativeSparseInstance");
	static jfieldID _context_field = env->GetFieldID(_class, "m_native_context", "J");
	sparse_storage* storage = (sparse_storage*)env->GetLongField(native_sparse_instance, _context_field);
	delete storage;
}

/*
 * Class:     org_moa_gpu_bridge_NativeSparseInstance
 * Method:    init
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeSparseInstance_init(JNIEnv * env, jobject native_sparse_instance)
{
	static jclass _class = env->FindClass("org/moa/gpu/bridge/NativeSparseInstance");
	static jfieldID _context_field = env->GetFieldID(_class, "m_native_context", "J");
	sparse_storage* storage = new sparse_storage();
	env->SetLongField(native_sparse_instance,_context_field, (jlong)storage);
	Java_org_moa_gpu_bridge_NativeSparseInstance_writeToNative(env, native_sparse_instance);
}


/*
 * Class:     org_moa_gpu_bridge_NativeDenseInstance
 * Method:    writeToNative
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeDenseInstance_writeToNative
  (JNIEnv * env, jobject obj)
{
	static jclass _class = env->FindClass("org/moa/gpu/bridge/NativeDenseInstance");
	static jfieldID _context_field = env->GetFieldID(_class, "m_native_context", "J");
	static jfieldID _instance_field = env->GetFieldID(_class, "m_instance", "Lweka/core/DenseInstance;");
	jobject instance = env->GetObjectField(obj,_instance_field);
	dense_storage* storage = (dense_storage*)env->GetLongField(obj, _context_field);
	instance_interface iface(env, instance);

	int class_index = iface.get_class_index();
	std::vector<double> values =iface.to_double_array();
	storage->m_class_value = values.at(class_index);
	values.erase(values.begin()+class_index);
	storage->m_values.swap(values);
}

/*
 * Class:     org_moa_gpu_bridge_NativeDenseInstance
 * Method:    release
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeDenseInstance_release
  (JNIEnv *env, jobject obj)
{
	static jclass _class = env->FindClass("org/moa/gpu/bridge/NativeDenseInstance");
	static jfieldID _context_field = env->GetFieldID(_class, "m_native_context", "J");
	dense_storage* storage = (dense_storage*)env->GetLongField(obj, _context_field);
	if (storage) delete storage;
	env->SetLongField(obj, _context_field, 0);
}

/*
 * Class:     org_moa_gpu_bridge_NativeDenseInstance
 * Method:    init
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeDenseInstance_init
  (JNIEnv* env, jobject obj)
{
	static jclass _class = env->FindClass("org/moa/gpu/bridge/NativeDenseInstance");
	static jfieldID _context_field = env->GetFieldID(_class, "m_native_context", "J");
	dense_storage* storage = new dense_storage();
	env->SetLongField(obj,_context_field, (jlong)storage);
	Java_org_moa_gpu_bridge_NativeDenseInstance_writeToNative(env, obj);

}




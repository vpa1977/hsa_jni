/*
 * SGD interface implementation
 *
 *  Created on: 26/04/2015
 *      Author: bsp
 */
#include <jni.h>
#include "library.hpp"
#include "org_moa_gpu_SGD.h"
#include <viennacl/ml/sgd.hpp>
#include "instance_interface.hpp"
#include "window_interface.hpp"
#include <algorithm>


viennacl::ml::sgd* GetNativeImpl(JNIEnv* env, jobject instance)
{
	static jclass sgd_class = env->FindClass("org/moa/gpu/SGD");
	static jfieldID context_field = env->GetFieldID(sgd_class, "m_native_context", "[B");
	assert(context_field != 0);
	jbyteArray array = (jbyteArray)env->GetObjectField(instance, context_field);
	viennacl::ml::sgd* sgd_impl = NULL;
	read_pointer(env, array,(void**) &sgd_impl);
	return sgd_impl;

}



jdoubleArray votesForInstance(JNIEnv* env, viennacl::vector<double>& instance, jobject sgd)
{
	viennacl::ml::sgd * sgd_impl  = GetNativeImpl(env,sgd);

	bool is_nominal_value = sgd_impl->is_nominal();
	size_t result_size = is_nominal_value ? 2 : 1;
	jdoubleArray d = env->NewDoubleArray(result_size);

	std::vector<double> result = sgd_impl->get_votes_for_instance(instance);
	assert(result.size() == result_size);
	env->SetDoubleArrayRegion(d,0, result_size, &result[0]);
	return d;

}



/*
 * Class:     org_moa_gpu_SGD
 * Method:    getVotesForSparseInstance
 * Signature: (JJJJLorg/moa/gpu/Window;)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_org_moa_gpu_SGD_getVotesForSparseInstance
  (JNIEnv * env, jobject sgd, jlong values, jlong indices, jlong ind_len, jlong total_len, jobject window)
{
	viennacl::vector<double> vcl_vector(total_len, get_global_context());
	fill_sparse(vcl_vector, values, indices, ind_len, total_len);
	return votesForInstance(env, vcl_vector, sgd);
}

/*
 * Class:     org_moa_gpu_SGD
 * Method:    getVotesForDenseInstance
 * Signature: (JJLorg/moa/gpu/Window;)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_org_moa_gpu_SGD_getVotesForDenseInstance
  (JNIEnv * env, jobject sgd, jlong values, jlong total_len, jobject window)
{
	viennacl::vector<double> vcl_vector(total_len, get_global_context());
	fill_dense(vcl_vector, values,  total_len);
	return votesForInstance(env, vcl_vector, sgd);
}



/*
 * Class:     org_moa_gpu_SGD
 * Method:    initNative
 * Signature: (IIZ)V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_SGD_initNative
  (JNIEnv * env, jobject sgd, jint size, jint loss, jboolean nominal)
{
	static jclass sgd_class = env->FindClass("org/moa/gpu/SGD");
	static jfieldID context_field = env->GetFieldID(sgd_class, "m_native_context", "[B");
	assert(context_field != 0);
	jbyteArray array = (jbyteArray)env->GetObjectField(sgd, context_field);
	viennacl::ml::sgd* sgd_impl =  new viennacl::ml::sgd(size, (viennacl::ml::sgd::LossFunction)loss, nominal, get_global_context());
	write_pointer(env, sgd_impl, array);
}

/*
 * Class:     org_moa_gpu_SGD
 * Method:    dispose
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_SGD_dispose
  (JNIEnv * env, jobject sgd)
{
	static jclass sgd_class = env->FindClass("org/moa/gpu/SGD");
	static jfieldID context_field = env->GetFieldID(sgd_class, "m_native_context", "[B");
	assert(context_field != 0);
	jbyteArray array = (jbyteArray)env->GetObjectField(sgd, context_field);
	viennacl::ml::sgd* sgd_impl = NULL;
	read_pointer(env, array,(void**) &sgd_impl);
	delete sgd_impl;
	sgd_impl = NULL;
	write_pointer(env, sgd_impl, array);
}



JNIEXPORT void JNICALL Java_org_moa_gpu_SGD_trainNative
  (JNIEnv *env, jobject sgd, jobject window)
{
	viennacl::ml::sgd * sgd_impl  = GetNativeImpl(env,sgd);
	typedef boost::numeric::ublas::coordinate_matrix<double> double_matrix;
	typedef boost::numeric::ublas::vector<double> double_vector;
	direct_memory_window<double_matrix, double_vector> m_native_window(env, window);
	viennacl::compressed_matrix<double>  values(m_native_window.values().size1(), m_native_window.values().size2(), get_global_context());
	viennacl::vector<double> classes(m_native_window.classes().size(), get_global_context());
	viennacl::copy(m_native_window.values(), values);
	viennacl::copy(m_native_window.classes(), classes);
	sgd_impl->train(classes, values);
	//std::vector<instance_interface> instances = get_window(env, window);
	//boost::numeric::ublas::coordinate_matrix<double> values_matrix(instances.size(), sgd_impl->instance_size());
	//boost::numeric::ublas::vector<double> classes_vector(instances.size());
	//instances_to_matrix(instances, values_matrix, classes_vector);
}









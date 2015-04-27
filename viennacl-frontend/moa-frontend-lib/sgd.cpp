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



JNIEXPORT jdoubleArray JNICALL Java_org_moa_gpu_SGD_getVotesForInstance
  (JNIEnv * env, jobject sgd, jobject instance, jobject window)
{
	static jclass sgd_class = env->FindClass("org/moa/gpu/SGD");
	static jfieldID is_nominal_field = env->GetFieldID(sgd_class, "m_nominal", "Z");
	bool is_nominal_value = env->GetBooleanField(sgd, is_nominal_field);
	size_t result_size = is_nominal_value ? 2 : 1;
	jdoubleArray d = env->NewDoubleArray(result_size);
	viennacl::ml::sgd * sgd_impl  = GetNativeImpl(env,sgd);

	sgd_impl->print_warning();


	return d;
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
	std::vector<instance_interface> instances = get_window(env, window);
	boost::numeric::ublas::matrix<double> m_values_matrix(instances.size(), sgd_impl->instance_size());
}









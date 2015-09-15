/*
 * SGD interface implementation
 *
 *  Created on: 26/04/2015
 *      Author: bsp
 */
#include <jni.h>
#include "library.hpp"
#include "org_moa_gpu_SparseSGD.h"
#include "viennacl/ml/sgd.hpp"
#include "instance_interface.hpp"
//#include "window_interface.hpp"
#include <algorithm>
#include "offheap_helpers.hpp"


const char* const thisClazz = "org/moa/gpu/SparseSGD";

static viennacl::ml::sgd* GetNativeImpl(JNIEnv* env, jobject instance)
{
	static jclass sgd_class = env->FindClass(thisClazz);
	static jfieldID context_field = env->GetFieldID(sgd_class, "m_native_context", "J");
	assert(context_field != 0);

	viennacl::ml::sgd* sgd_impl = (viennacl::ml::sgd*)env->GetLongField(instance, context_field);
	return sgd_impl;

}



static jdoubleArray votesForInstance(JNIEnv* env, const viennacl::vector<double>& instance, jobject sgd)
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
 * Signature: (Lorg/moa/gpu/bridge/NativeSparseInstance;)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_org_moa_gpu_SparseSGD_getVotesForSparseInstance
  (JNIEnv *env, jobject sgd, jobject native_instance)
{
	dense_offheap_buffer buffer(env, native_instance);
	return votesForInstance(env, buffer.values(), sgd);
}


/*
 * Class:     org_moa_gpu_SGD
 * Method:    initNative
 * Signature: (IIZ)V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_SparseSGD_initNative
  (JNIEnv * env, jobject sgd, jint attribute_size,jint window_size, jint loss, jboolean nominal, jdouble learning_rate, jdouble lambda)
{
	static jclass sgd_class = env->FindClass(thisClazz);
	static jfieldID context_field = env->GetFieldID(sgd_class, "m_native_context", "J");
	assert(context_field != 0);
	viennacl::ml::sgd* sgd_impl =  new viennacl::ml::sgd(attribute_size, window_size, (viennacl::ml::sgd::LossFunction)loss, get_global_context(),
			learning_rate, lambda, nominal);
	env->SetLongField(sgd, context_field, (jlong)sgd_impl);
}

/*
 * Class:     org_moa_gpu_SGD
 * Method:    dispose
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_SparseSGD_dispose
  (JNIEnv * env, jobject sgd)
{
	static jclass sgd_class = env->FindClass(thisClazz);
	static jfieldID context_field = env->GetFieldID(sgd_class, "m_native_context", "J");
	assert(context_field != 0);
	viennacl::ml::sgd* sgd_impl = (viennacl::ml::sgd*)env->GetLongField(sgd, context_field);
	delete sgd_impl;
	sgd_impl = NULL;
	env->SetLongField(sgd, context_field, 0);
}


JNIEXPORT void JNICALL Java_org_moa_gpu_SparseSGD_trainNative
  (JNIEnv *env, jobject sgd, jobject instance_batch)
{
	viennacl::ml::sgd * sgd_impl  = GetNativeImpl(env,sgd);
	sparse_offheap_buffer buffer(env, instance_batch);
	sgd_impl->train(buffer.class_values(), buffer.values());
//	sgd_impl->print_weights(); 
//	get_global_context().opencl_context().get_queue().finish();
}









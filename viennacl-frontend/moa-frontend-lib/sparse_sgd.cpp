/*
 * SGD interface implementation
 *
 *  Created on: 26/04/2015
 *      Author: bsp
 */
#include <jni.h>
#include "library.hpp"
#include "org_moa_gpu_SparseSGD.h"
#include <viennacl/ml/sgd.hpp>
#include "instance_interface.hpp"
//#include "window_interface.hpp"
#include <algorithm>
#include "native_instance_batch.hpp"


const char* const thisClazz = "org/moa/gpu/SparseSGD";

static viennacl::ml::sgd* GetNativeImpl(JNIEnv* env, jobject instance)
{
	static jclass sgd_class = env->FindClass(thisClazz);
	static jfieldID context_field = env->GetFieldID(sgd_class, "m_native_context", "J");
	assert(context_field != 0);

	viennacl::ml::sgd* sgd_impl = (viennacl::ml::sgd*)env->GetLongField(instance, context_field);
	return sgd_impl;

}



static jdoubleArray votesForInstance(JNIEnv* env, viennacl::vector<double>& instance, jobject sgd)
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
	static jclass _instance_class = env->FindClass("org/moa/gpu/bridge/NativeSparseInstance");
	static jfieldID _instance_context_field = env->GetFieldID(_instance_class, "m_native_context", "J");
	sparse_storage* storage = (sparse_storage*)env->GetLongField(native_instance, _instance_context_field);
	viennacl::ml::sgd * sgd_impl  = GetNativeImpl(env,sgd);
	instance_interface iface(env,native_instance);

	viennacl::vector<double> instance = storage->vector(sgd_impl->instance_size());
	return votesForInstance(env, instance, sgd);
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

	static jclass _class = env->FindClass("org/moa/gpu/bridge/NativeSparseInstanceBatch");
	static jfieldID _context_field = env->GetFieldID(_class, "m_native_context", "J");
	sparse_instance_batch* batch = (sparse_instance_batch*)env->GetLongField(instance_batch, _context_field);
	sgd_impl->train(batch->m_class_values, batch->m_instance_values);
	get_global_context().opencl_context().get_queue().finish();

}








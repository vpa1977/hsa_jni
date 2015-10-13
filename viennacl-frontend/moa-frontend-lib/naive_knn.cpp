#include "library.hpp"
#include "viennacl/ml/naive_knn.hpp"
#include "org_moa_gpu_NaiveKnn.h"
#include "offheap_helpers.hpp"

typedef double KnnNumericType;

const char* const thisClazz = "org/moa/gpu/NaiveKnn";

static viennacl::ml::knn::naive_knn<KnnNumericType>* GetNativeImpl(JNIEnv* env, jobject instance)
{
	static jclass sgd_class = env->FindClass(thisClazz);
	static jfieldID context_field = env->GetFieldID(sgd_class, "m_native_context", "J");
	assert(context_field != 0);

	viennacl::ml::knn::naive_knn<KnnNumericType>* knn_impl = (viennacl::ml::knn::naive_knn<KnnNumericType>*)env->GetLongField(instance, context_field);
	return knn_impl;

}

/*
* Class:     org_moa_gpu_NaiveKnn
* Method:    getVotesForDenseInstance
* Signature: (Lorg/moa/gpu/bridge/DenseOffHeapBuffer;)[D
*/
JNIEXPORT jdoubleArray JNICALL Java_org_moa_gpu_NaiveKnn_getVotesForDenseInstance
(JNIEnv * env, jobject knn, jobject test_object, jobject model_object)
{
	viennacl::ml::knn::naive_knn<KnnNumericType>* knn_impl = GetNativeImpl(env, knn);
	dense_offheap_buffer test(env, test_object);
	dense_offheap_buffer model(env, model_object);
	
	std::vector<double> result = knn_impl->distribution_for_instance(test.values(), model.class_values(), model.values());
	jdoubleArray d = env->NewDoubleArray((jsize)result.size());
	env->SetDoubleArrayRegion(d, 0, (jsize)result.size(), &result[0]);
	return d;
}


/*
* Class:     org_moa_gpu_NaiveKnn
* Method:    initNative
* Signature: (II)V
*/
JNIEXPORT void JNICALL Java_org_moa_gpu_NaiveKnn_initNative
(JNIEnv * env, jobject knn , jint, jint, jint class_index, jintArray attributeTypes)
{
	static jclass knn_class = env->FindClass(thisClazz);
	static jfieldID context_field = env->GetFieldID(knn_class, "m_native_context", "J");
	assert(context_field != 0);

	viennacl::ml::knn::naive_knn<KnnNumericType>* knn_impl
		= new viennacl::ml::knn::naive_knn<KnnNumericType>(class_index, get_global_context());

	
	env->SetLongField(knn, context_field, (jlong)knn_impl);

}

/*
* Class:     org_moa_gpu_NaiveKnn
* Method:    dispose
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_org_moa_gpu_NaiveKnn_dispose
(JNIEnv * env, jobject inst)
{
	static jclass knn_class = env->FindClass(thisClazz);
	static jfieldID context_field = env->GetFieldID(knn_class, "m_native_context", "J");

	viennacl::ml::knn::naive_knn<KnnNumericType> * knn_impl = GetNativeImpl(env, inst);
	delete knn_impl;
	env->SetLongField(inst, context_field, 0);
}


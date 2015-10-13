#include "library.hpp"
#include "viennacl/ml/naive_knn.hpp"
#include "org_moa_gpu_NaiveKnn.h"
#include "offheap_helpers.hpp"

typedef double KnnNumericType;

const char* const thisClazz = "org/moa/gpu/NaiveKnn";

class default_knn_impl : public viennacl::ml::knn::naive_knn<KnnNumericType>
{
public:
	default_knn_impl(int num_attrs, const viennacl::ml::knn::knn_options& options, const viennacl::vector<int>& attr_map) :
		viennacl::ml::knn::naive_knn<KnnNumericType>(options, attr_map,
			viennacl::ml::knn::euclidean_distance_wg<KnnNumericType>(get_global_context(), options.class_index_, num_attrs),
			viennacl::ml::knn::default_sort_strategy<KnnNumericType>(options.distance_weighting_ != 0),
			get_global_context())
	{}
};


static default_knn_impl* GetNativeImpl(JNIEnv* env, jobject instance)
{
	static jclass sgd_class = env->FindClass(thisClazz);
	static jfieldID context_field = env->GetFieldID(sgd_class, "m_native_context", "J");
	assert(context_field != 0);

	default_knn_impl* knn_impl = (default_knn_impl*)env->GetLongField(instance, context_field);
	return knn_impl;
}

JNIEXPORT void JNICALL Java_org_moa_gpu_NaiveKnn_train(JNIEnv * env, jobject knn, jobject model)
{
	default_knn_impl* knn_impl = GetNativeImpl(env, knn);
	dense_offheap_buffer test(env, model);
	knn_impl->train(test.values(), test.class_values_std(), test.weight_values_std());
}

/*
* Class:     org_moa_gpu_NaiveKnn
* Method:    getVotesForDenseInstance
* Signature: (Lorg/moa/gpu/bridge/DenseOffHeapBuffer;)[D
*/
JNIEXPORT jdoubleArray JNICALL Java_org_moa_gpu_NaiveKnn_getVotesForDenseInstance
(JNIEnv * env, jobject knn, jobject test_object)
{
	default_knn_impl* knn_impl = GetNativeImpl(env, knn);
	dense_offheap_buffer test(env, test_object);
	std::vector<double> result = knn_impl->distribution_for_instance(test.values());
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
(JNIEnv * env, jobject knn , jint num_attrs, jint batch_size, jint class_index, jintArray attributeTypes, jint num_classes, jint k, jint distance_weighting)
{
	static jclass knn_class = env->FindClass(thisClazz);
	static jfieldID context_field = env->GetFieldID(knn_class, "m_native_context", "J");
	assert(context_field != 0);
	jboolean is_copy = 0;
	
	jint* jni_attr_map = env->GetIntArrayElements(attributeTypes, &is_copy);
	std::vector<int> cpu_attr_map(num_attrs);
	std::copy(jni_attr_map, jni_attr_map + num_attrs, cpu_attr_map.begin());
	viennacl::vector<int> attribute_map(num_attrs, get_global_context());
	viennacl::copy(cpu_attr_map, attribute_map);

	viennacl::ml::knn::knn_options options;
	options.class_index_ = class_index;
	options.class_type_ = (viennacl::ml::knn::AttributeType)cpu_attr_map[class_index];
	options.distance_weighting_ = (viennacl::ml::knn::DistanceWeighting)distance_weighting;
	options.k_ = k;
	options.num_classes_ = num_classes;
	default_knn_impl* knn_impl = new default_knn_impl(num_attrs,options, attribute_map);

	
	env->SetLongField(knn, context_field, (jlong)knn_impl);
	env->ReleaseIntArrayElements(attributeTypes, jni_attr_map, JNI_ABORT);
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

	default_knn_impl * knn_impl = GetNativeImpl(env, inst);
	delete knn_impl;
	env->SetLongField(inst, context_field, 0);
}


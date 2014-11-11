/*
 * context.cpp
 *
 *  Created on: 19/10/2014
 *      Author: bsp
 */



#include "context.hpp"

/**
 * context wrapper for the hsa algorithms
 */


std::shared_ptr<HSAContext> p_context(HSAContext::Create());

Algorithms g_algorithms(p_context);




JNIEXPORT void JNICALL Java_hsa_1jni_hsa_1jni_WekaHSAContext_init
  (JNIEnv *, jobject)
{

}


JNIEXPORT void JNICALL Java_hsa_1jni_hsa_1jni_WekaHSAContext_00024KnnNativeContext_rescanRanges
  (JNIEnv *env, jobject obj, jdoubleArray window, jdoubleArray ranges, jint numeric_size, jint instance_size, jint window_size, jint current_size)
{
	jboolean is_copy;
	double* window_ptr =(double*) env->GetPrimitiveArrayCritical(window, &is_copy);
	double* ranges_ptr =(double*) env->GetPrimitiveArrayCritical(ranges, &is_copy);

	// test code please ignore
	int num_instances = window_size / instance_size;
	for (int i = 0 ; i < numeric_size; ++i)
	{
		int offset = i * num_instances;
		int off = i*2;
		g_algorithms.m_min_max_value.find(window_ptr+offset,current_size,ranges_ptr[off], ranges_ptr[off+1] );
	}

	env->ReleasePrimitiveArrayCritical(window, window_ptr, 0);
	env->ReleasePrimitiveArrayCritical(ranges, ranges_ptr, 0);
}


//		private native void knn(double[] instance, double[] m_window, double[] m_ranges, int numerics_size,int instance_size, int window_size, double[] result_distance, int[] result_index);
JNIEXPORT void JNICALL Java_hsa_1jni_hsa_1jni_WekaHSAContext_00024KnnNativeContext_knn
  (JNIEnv * env, jobject obj, jdoubleArray instance, jdoubleArray window, jdoubleArray ranges, jint numerics_size, jint instance_size, jint window_size, jint current_size, jdoubleArray distances, jintArray indexes)
{
	jboolean is_copy;
	double* window_ptr =(double*) env->GetPrimitiveArrayCritical(window, &is_copy);
	double* instance_ptr =(double*) env->GetPrimitiveArrayCritical(instance, &is_copy);
	double* ranges_ptr =(double*) env->GetPrimitiveArrayCritical(ranges, &is_copy);
	double* distances_ptr = (double*) env->GetPrimitiveArrayCritical(distances, &is_copy);
	int* indexes_ptr = (int*) env->GetPrimitiveArrayCritical(indexes, &is_copy);
	{
		g_algorithms.m_per_attribute_distances.distance(instance_ptr,window_ptr, ranges_ptr,numerics_size, instance_size, window_size,current_size, distances_ptr );
		g_algorithms.m_merge_sort.sort(distances_ptr, indexes_ptr, window_size);
	}

	env->ReleasePrimitiveArrayCritical(window, window_ptr, 0);
	env->ReleasePrimitiveArrayCritical(instance, instance_ptr, 0);
	env->ReleasePrimitiveArrayCritical(ranges, ranges_ptr, 0);
	env->ReleasePrimitiveArrayCritical(distances, distances_ptr, 0);
	env->ReleasePrimitiveArrayCritical(indexes, indexes_ptr, 0);

}



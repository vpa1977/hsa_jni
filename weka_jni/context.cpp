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
Algorithms g_algorithms(HSAContext::Create());



JNIEXPORT void JNICALL Java_hsa_1jni_hsa_1jni_WekaHSAContext_init
  (JNIEnv *, jobject)
{
	// nothing for now.
	// bind object to context in the future
}


JNIEXPORT void JNICALL Java_hsa_1jni_hsa_1jni_WekaHSAContext_00024KnnNativeContext_rescanRanges
  (JNIEnv *env, jobject, jdoubleArray window, jdoubleArray ranges, jint numerics, jint instance_size, jint window_size, jint attr)
{
	jboolean is_copy;
	double* window_ptr =(double*) env->GetPrimitiveArrayCritical(window, &is_copy);
	double* ranges_ptr =(double*) env->GetPrimitiveArrayCritical(ranges, &is_copy);
	int off = attr*2;
	double min = g_algorithms.m_min_value.find(window_ptr,window_size, instance_size, attr);
	double max = g_algorithms.m_max_value.find(window_ptr,window_size, instance_size, attr);
	ranges_ptr[off] = min;
	ranges_ptr[off+1] = max;

	env->ReleasePrimitiveArrayCritical(window, window_ptr, 0);
	env->ReleasePrimitiveArrayCritical(ranges, ranges_ptr, 0);
}


//		private native void knn(double[] instance, double[] m_window, double[] m_ranges, int numerics_size,int instance_size, int window_size, double[] result_distance, int[] result_index);

JNIEXPORT void JNICALL Java_hsa_1jni_hsa_1jni_WekaHSAContext_00024KnnNativeContext_knn
  (JNIEnv *env, jobject obj, jdoubleArray instance, jdoubleArray window, jdoubleArray ranges, jint numerics_size, jint instance_size, jint window_size, jdoubleArray distances, jintArray indexes)
{
	jboolean is_copy;
	double* window_ptr =(double*) env->GetPrimitiveArrayCritical(window, &is_copy);
	double* instance_ptr =(double*) env->GetPrimitiveArrayCritical(instance, &is_copy);
	double* ranges_ptr =(double*) env->GetPrimitiveArrayCritical(ranges, &is_copy);
	double* distances_ptr = (double*) env->GetPrimitiveArrayCritical(distances, &is_copy);
	int* indexes_ptr = (int*) env->GetPrimitiveArrayCritical(indexes, &is_copy);
	{
		g_algorithms.m_square_distance.calculate(instance_ptr,window_ptr, window_size, ranges_ptr,instance_size, numerics_size, distances_ptr );

		g_algorithms.m_merge_sort.sort(distances_ptr, indexes_ptr, window_size);
	}

	env->ReleasePrimitiveArrayCritical(window, window_ptr, 0);
	env->ReleasePrimitiveArrayCritical(instance, instance_ptr, 0);
	env->ReleasePrimitiveArrayCritical(ranges, ranges_ptr, 0);
	env->ReleasePrimitiveArrayCritical(distances, distances_ptr, 0);
	env->ReleasePrimitiveArrayCritical(indexes, indexes_ptr, 0);

}



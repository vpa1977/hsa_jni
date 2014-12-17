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
		g_algorithms.m_per_attribute_distances.distance(instance_ptr,window_ptr, ranges_ptr,
		  numerics_size, instance_size, window_size,current_size, distances_ptr );
		//g_algorithms.m_square_distance.calculate(instance_ptr, window_ptr, window_size,
			///	ranges_ptr, instance_size, numerics_size, distances_ptr);
		g_algorithms.m_merge_sort.sort(distances_ptr, indexes_ptr, current_size);
	}

	env->ReleasePrimitiveArrayCritical(window, window_ptr, 0);
	env->ReleasePrimitiveArrayCritical(instance, instance_ptr, 0);
	env->ReleasePrimitiveArrayCritical(ranges, ranges_ptr, 0);
	env->ReleasePrimitiveArrayCritical(distances, distances_ptr, 0);
	env->ReleasePrimitiveArrayCritical(indexes, indexes_ptr, 0);

}

/**
 * Compute SGD for single instance.
 */
void processSGD(size_t instance_size, double * instance_values, double* instance_indices_ptr, double weights_ptr, bool isNominal, int classIndex, double multiplier)
{

}

#define HINGE 0

double dloss(double z)
{
//	if (m_loss == HINGE) {
		return (z < 1) ? 1 : 0;
//	}

/*	if (m_loss == LOGLOSS) {
		// log loss
		if (z < 0) {
			return 1.0 / (Math.exp(z) + 1.0);
		} else {
			double t = Math.exp(-z);
			return t / (t + 1);
		}
	}

	// squared loss
	return z;
*/
}

double product(double* values, int* indices, double* weights, int size, int classIndex)
{
	double result = 0;
	for (int i = 0 ; i < size ; ++i)
		if (indices[i] != classIndex)
			result += values[i] * weights[indices[i]];
	return (result);
}

/*
 * Class:     hsa_jni_hsa_jni_WekaHSAContext_SGD
 * Method:    UpdateWeights
 * Signature: (Lhsa_jni/hsa_jni/InstanceBatch;[D)V
 */
JNIEXPORT void JNICALL Java_hsa_1jni_hsa_1jni_WekaHSAContext_00024SGD_UpdateWeights
  (JNIEnv * env, jobject this_object, jobject batch_instance, jdoubleArray weights)
{

	static jclass batch_instance_class = env->FindClass("hsa_jni/hsa_jni/InstanceBatch");
	static jmethodID isNominalID = env->GetMethodID(batch_instance_class,"isNominal", "()Z" );
	static jmethodID classIndexID = env->GetMethodID(batch_instance_class, "classIndex", "()I" );
	static jmethodID sizeID = env->GetMethodID( batch_instance_class,"size", "()I" );
	static jmethodID valuesID = env->GetMethodID( batch_instance_class, "values", "()[[D" );
	static jmethodID indicesID = env->GetMethodID( batch_instance_class, "indices", "()[[I" );
	static jmethodID classValuesID = env->GetMethodID( batch_instance_class, "classValues", "()[D" );
	static jmethodID multplierID = env->GetMethodID( batch_instance_class, "getMultiplier", "()D" );

	static jfieldID biasID = env->GetFieldID(batch_instance_class, "m_bias", "D");
	static jfieldID timeID = env->GetFieldID(batch_instance_class, "m_t", "D");
	static jfieldID learningRateID = env->GetFieldID(batch_instance_class, "m_learningRate", "D");

	double multplier = env->CallDoubleMethod(batch_instance,multplierID );
	bool isNominal = env->CallBooleanMethod(batch_instance,isNominalID);
	int classIndex = env->CallIntMethod(batch_instance, classIndexID);
	int size = env->CallIntMethod(batch_instance, sizeID);
	double m_bias = env->GetDoubleField(batch_instance, biasID);
	double m_learningRate = env->GetDoubleField(batch_instance, learningRateID);
	int m_t = env->GetDoubleField(batch_instance, timeID);
	int m_loss =0;


	jboolean is_copy;




	jobjectArray values = (jobjectArray)env->CallObjectMethod(batch_instance, valuesID);
	jobjectArray indices =(jobjectArray) env->CallObjectMethod(batch_instance, indicesID);
	jdoubleArray class_values = (jdoubleArray) env->CallObjectMethod(batch_instance, classValuesID);

	double *class_values_ptr = (double*) env->GetPrimitiveArrayCritical(class_values, &is_copy);
	double* weights_ptr = (double*) env->GetPrimitiveArrayCritical(weights, &is_copy);

	static std::vector<double> wx;
	wx.resize(size);

	for (int i = 0 ;i < size ; ++i)
	{
		jintArray instance_indices = (jintArray)env->GetObjectArrayElement(indices, i);
		jdoubleArray  instance_values = (jdoubleArray)env->GetObjectArrayElement(values, i);

		int instance_size = env->GetArrayLength(instance_values);

		double* instance_values_ptr = (double*)env->GetPrimitiveArrayCritical(instance_values, &is_copy);
		int* instance_indices_ptr = (int*)env->GetPrimitiveArrayCritical(instance_indices, &is_copy);

		wx[i] = product(instance_values_ptr, instance_indices_ptr, weights_ptr, instance_size,classIndex);

		env->ReleasePrimitiveArrayCritical( instance_indices, instance_indices_ptr, 0);
		env->ReleasePrimitiveArrayCritical( instance_values, instance_values_ptr, 0);
	}


	for (int i = 0 ;i < size; ++i)
	{
		jintArray instance_indices = (jintArray)env->GetObjectArrayElement(indices, i);
		jdoubleArray  instance_values = (jdoubleArray)env->GetObjectArrayElement(values, i);
		int instance_size = env->GetArrayLength(instance_values);

		double* instance_values_ptr = (double*)env->GetPrimitiveArrayCritical(instance_values, &is_copy);
		int* instance_indices_ptr = (int*)env->GetPrimitiveArrayCritical(instance_indices, &is_copy);

		//
		double y;
		double z;
		if (isNominal) {
			y = (class_values_ptr[i] == 0) ? -1 : 1;
			z = y * (wx[i] + m_bias);
		} else {
			y = class_values_ptr[i];
			z = y - (wx[i] + m_bias);
			y = 1;
		}

		// Only need to do the following if the loss is non-zero
		if (m_loss != HINGE || (z < 1)) {

			// Compute Factor for updates
			double factor = m_learningRate * y * dloss(z);

			// Update coefficients for attributes
			for (int i = 0 ;i < instance_size ; ++i)
			{
				if (instance_indices_ptr[i] == classIndex)
					continue;
				weights_ptr[instance_indices_ptr[i]] += factor * instance_values_ptr[i];
			}
			// update the bias
			m_bias += factor;
		}
		m_t++;

		env->ReleasePrimitiveArrayCritical( instance_indices, instance_indices_ptr, 0);
		env->ReleasePrimitiveArrayCritical( instance_values, instance_values_ptr, 0);

	}


	env->SetDoubleField(batch_instance, biasID, m_bias);
	env->SetDoubleField(batch_instance, learningRateID, m_learningRate);
	env->SetDoubleField(batch_instance, timeID, m_t);

	//

	env->ReleasePrimitiveArrayCritical(class_values, class_values_ptr, 0);
	env->ReleasePrimitiveArrayCritical( weights, weights_ptr, 0);
}



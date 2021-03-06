/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_moa_gpu_NaiveKnn */

#ifndef _Included_org_moa_gpu_NaiveKnn
#define _Included_org_moa_gpu_NaiveKnn
#ifdef __cplusplus
extern "C" {
#endif
#undef org_moa_gpu_NaiveKnn_serialVersionUID
#define org_moa_gpu_NaiveKnn_serialVersionUID 1i64
/*
 * Class:     org_moa_gpu_NaiveKnn
 * Method:    getVotesForDenseInstance
 * Signature: (Lorg/moa/gpu/bridge/DenseOffHeapBuffer;)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_org_moa_gpu_NaiveKnn_getVotesForDenseInstance
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_moa_gpu_NaiveKnn
 * Method:    initNative
 * Signature: (III[IIII)V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_NaiveKnn_initNative
  (JNIEnv *, jobject, jint, jint, jint, jintArray, jint, jint, jint);

/*
 * Class:     org_moa_gpu_NaiveKnn
 * Method:    train
 * Signature: (Lorg/moa/gpu/bridge/DenseOffHeapBuffer;)V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_NaiveKnn_train
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_moa_gpu_NaiveKnn
 * Method:    dispose
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_NaiveKnn_dispose
  (JNIEnv *, jobject);

#ifdef __cplusplus
}
#endif
#endif

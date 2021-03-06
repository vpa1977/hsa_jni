/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_moa_gpu_SparseSGD */

#ifndef _Included_org_moa_gpu_SparseSGD
#define _Included_org_moa_gpu_SparseSGD
#ifdef __cplusplus
extern "C" {
#endif
#undef org_moa_gpu_SparseSGD_serialVersionUID
#define org_moa_gpu_SparseSGD_serialVersionUID 1i64
#undef org_moa_gpu_SparseSGD_QUEUE_SIZE
#define org_moa_gpu_SparseSGD_QUEUE_SIZE 30L
#undef org_moa_gpu_SparseSGD_HINGE
#define org_moa_gpu_SparseSGD_HINGE 0L
#undef org_moa_gpu_SparseSGD_LOGLOSS
#define org_moa_gpu_SparseSGD_LOGLOSS 1L
#undef org_moa_gpu_SparseSGD_SQUAREDLOSS
#define org_moa_gpu_SparseSGD_SQUAREDLOSS 2L
/*
 * Class:     org_moa_gpu_SparseSGD
 * Method:    getVotesForSparseInstance
 * Signature: (Lorg/moa/gpu/bridge/DenseOffHeapBuffer;)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_org_moa_gpu_SparseSGD_getVotesForSparseInstance
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_moa_gpu_SparseSGD
 * Method:    trainNative
 * Signature: (Lorg/moa/gpu/bridge/SparseOffHeapBuffer;)V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_SparseSGD_trainNative
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_moa_gpu_SparseSGD
 * Method:    initNative
 * Signature: (IIIZDD)V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_SparseSGD_initNative
  (JNIEnv *, jobject, jint, jint, jint, jboolean, jdouble, jdouble);

/*
 * Class:     org_moa_gpu_SparseSGD
 * Method:    dispose
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_SparseSGD_dispose
  (JNIEnv *, jobject);

#ifdef __cplusplus
}
#endif
#endif

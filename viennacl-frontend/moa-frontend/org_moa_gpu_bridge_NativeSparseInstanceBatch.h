/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_moa_gpu_bridge_NativeSparseInstanceBatch */

#ifndef _Included_org_moa_gpu_bridge_NativeSparseInstanceBatch
#define _Included_org_moa_gpu_bridge_NativeSparseInstanceBatch
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_moa_gpu_bridge_NativeSparseInstanceBatch
 * Method:    add
 * Signature: (Lorg/moa/gpu/bridge/NativeInstance;)Z
 */
JNIEXPORT jboolean JNICALL Java_org_moa_gpu_bridge_NativeSparseInstanceBatch_add
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_moa_gpu_bridge_NativeSparseInstanceBatch
 * Method:    clear
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeSparseInstanceBatch_clear
  (JNIEnv *, jobject);

/*
 * Class:     org_moa_gpu_bridge_NativeSparseInstanceBatch
 * Method:    init
 * Signature: (Lweka/core/Instances;I)V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeSparseInstanceBatch_init
  (JNIEnv *, jobject, jobject, jint);

/*
 * Class:     org_moa_gpu_bridge_NativeSparseInstanceBatch
 * Method:    release
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeSparseInstanceBatch_release
  (JNIEnv *, jobject);

/*
 * Class:     org_moa_gpu_bridge_NativeSparseInstanceBatch
 * Method:    commit
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeSparseInstanceBatch_commit
  (JNIEnv *, jobject);

#ifdef __cplusplus
}
#endif
#endif

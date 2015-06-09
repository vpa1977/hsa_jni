/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_moa_gpu_bridge_NativeDenseInstanceBatch */

#ifndef _Included_org_moa_gpu_bridge_NativeDenseInstanceBatch
#define _Included_org_moa_gpu_bridge_NativeDenseInstanceBatch
#ifdef __cplusplus
extern "C" {
#endif
#undef org_moa_gpu_bridge_NativeDenseInstanceBatch_SLIDING_WINDOW
#define org_moa_gpu_bridge_NativeDenseInstanceBatch_SLIDING_WINDOW 1L
#undef org_moa_gpu_bridge_NativeDenseInstanceBatch_BUFFER
#define org_moa_gpu_bridge_NativeDenseInstanceBatch_BUFFER 2L
/*
 * Class:     org_moa_gpu_bridge_NativeDenseInstanceBatch
 * Method:    add
 * Signature: (Lorg/moa/gpu/bridge/NativeInstance;)Z
 */
JNIEXPORT jboolean JNICALL Java_org_moa_gpu_bridge_NativeDenseInstanceBatch_add
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_moa_gpu_bridge_NativeDenseInstanceBatch
 * Method:    clear
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeDenseInstanceBatch_clear
  (JNIEnv *, jobject);

/*
 * Class:     org_moa_gpu_bridge_NativeDenseInstanceBatch
 * Method:    init
 * Signature: (Lweka/core/Instances;II)V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeDenseInstanceBatch_init
  (JNIEnv *, jobject, jobject, jint, jint);

/*
 * Class:     org_moa_gpu_bridge_NativeDenseInstanceBatch
 * Method:    release
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeDenseInstanceBatch_release
  (JNIEnv *, jobject);

#ifdef __cplusplus
}
#endif
#endif
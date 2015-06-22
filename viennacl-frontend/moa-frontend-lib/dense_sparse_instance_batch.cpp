#include "library.hpp"
#include "instance_interface.hpp"
#include "org_moa_gpu_bridge_NativeDenseInstanceBatch.h"
#include "viennacl/backend/mem_handle.hpp"
#include "native_instance_batch.hpp"


const char* const theClazz = "org/moa/gpu/bridge/NativeDenseInstanceBatch";

/*
 * Class:     org_moa_gpu_bridge_NativeDenseInstanceBatch
 * Method:    add
 * Signature: (Lorg/moa/gpu/bridge/NativeInstance;)V
 */
JNIEXPORT jboolean JNICALL Java_org_moa_gpu_bridge_NativeDenseInstanceBatch_add(JNIEnv * env, jobject instance_batch, jobject native_instance)
{
	jboolean full = 0;
	static jclass _class = env->FindClass(theClazz);
	static jfieldID _context_field = env->GetFieldID(_class, "m_native_context", "J");
	dense_instance_batch* batch = (dense_instance_batch*)env->GetLongField(instance_batch, _context_field);

	static jclass _instance_class = env->FindClass("org/moa/gpu/bridge/NativeDenseInstance");
	static jfieldID _instance_context_field = env->GetFieldID(_instance_class, "m_native_context", "J");
	dense_storage* storage = (dense_storage*)env->GetLongField(native_instance, _instance_context_field);

	if (batch->add(storage))
		full = 1;

	return full;
}

/*
 * Class:     org_moa_gpu_bridge_NativeDenseInstanceBatch
 * Method:    clear
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeDenseInstanceBatch_clear
  (JNIEnv * env, jobject instance_batch)
{
	static jclass _class = env->FindClass(theClazz);
	static jfieldID _context_field = env->GetFieldID(_class, "m_native_context", "J");
	dense_instance_batch* batch = (dense_instance_batch*)env->GetLongField(instance_batch, _context_field);
	batch->clear();
}

/*
 * Class:     org_moa_gpu_bridge_NativeDenseInstanceBatch
 * Method:    init
 * Signature: (Lweka/core/Instances;I)V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeDenseInstanceBatch_init (JNIEnv * env, jobject instance_batch, jobject instances, jint num_rows)
{
	static jclass _class = env->FindClass(theClazz);
	static jfieldID _context_field = env->GetFieldID(_class, "m_native_context", "J");
	dataset_interface dataset(env,instances);
	dense_instance_batch* batch = new dense_instance_batch(num_rows, dataset.get_num_attributes()-1,get_global_context());
	env->SetLongField(instance_batch, _context_field, (jlong)batch);
}

/*
 * Class:     org_moa_gpu_bridge_NativeDenseInstanceBatch
 * Method:    release
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeDenseInstanceBatch_release(JNIEnv * env, jobject instance_batch)
{
	static jclass _class = env->FindClass(theClazz);
	static jfieldID _context_field = env->GetFieldID(_class, "m_native_context", "J");
	dense_instance_batch* batch = (dense_instance_batch*)env->GetLongField(instance_batch, _context_field);
	delete batch;

}

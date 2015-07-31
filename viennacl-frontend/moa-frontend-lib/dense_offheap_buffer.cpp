
#ifdef VIENNACL_WITH_OPENCL
#include <CL/cl.h>
#endif
#include "library.hpp"
#include "org_moa_gpu_bridge_DenseOffHeapBuffer.h"

const char* const DENSE_OFFHEAP_BUFFER = "org/moa/gpu/bridge/DenseOffHeapBuffer";

/*
* Class:     org_moa_gpu_bridge_DenseOffHeapBuffer
* Method:    allocate
* Signature: (J)J
*/
JNIEXPORT jlong JNICALL Java_org_moa_gpu_bridge_DenseOffHeapBuffer_allocate
(JNIEnv *env, jobject obj, jlong size_in_bytes)
{
	cl_int err;
	const cl_context& cl_ctx = get_global_context().opencl_context().handle().get();
	void * ptr = clSVMAlloc(cl_ctx, CL_MEM_READ_WRITE, size_in_bytes, 0);
	cl_command_queue queue = get_global_context().opencl_context().get_queue(get_global_context().opencl_context().current_device().id(), DATA_TRANSFER_QUEUE).handle().get();
	err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, ptr, size_in_bytes, 0, NULL, NULL);
	VIENNACL_ERR_CHECK(err);
	return (jlong)ptr;
}


/*
* Class:     org_moa_gpu_bridge_DenseOffHeapBuffer
* Method:    release
* Signature: (J)V
*/
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_DenseOffHeapBuffer_release
(JNIEnv *, jobject, jlong handle)
{
	void* ptr = (void*)handle;
	const cl_context& cl_ctx = get_global_context().opencl_context().handle().get();
	clSVMFree(cl_ctx, ptr);
}

/*
* Class:     org_moa_gpu_bridge_DenseOffHeapBuffer
* Method:    begin
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_DenseOffHeapBuffer_begin
(JNIEnv *env, jobject buffer)
{
/*	static jclass clazz = env->FindClass(DENSE_OFFHEAP_BUFFER);
	static jfieldID field = env->GetFieldID(clazz, "m_buffer", "J");
	static jfieldID size_field = env->GetFieldID(clazz, "m_size", "J");
	void * ptr = (void*)env->GetLongField(buffer, field);
	cl_int err;
	jlong size = env->GetLongField(buffer, size_field);
	cl_command_queue queue = get_global_context().opencl_context().get_queue(get_global_context().opencl_context().current_device().id(), DATA_TRANSFER_QUEUE).handle().get();
	err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, ptr, size, 0, NULL, NULL);
	VIENNACL_ERR_CHECK(err);
*/
}

/*
* Class:     org_moa_gpu_bridge_DenseOffHeapBuffer
* Method:    commit
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_DenseOffHeapBuffer_commit
(JNIEnv * env, jobject buffer)
{
	static jclass clazz = env->FindClass(DENSE_OFFHEAP_BUFFER);
	static jfieldID field = env->GetFieldID(clazz, "m_buffer", "J");
	static jfieldID classes_field = env->GetFieldID(clazz, "m_class_buffer", "J");
	static jfieldID size_field = env->GetFieldID(clazz, "m_size", "J");
	void * ptr = (void*)env->GetLongField(buffer, field);
	void * class_ptr = (void*)env->GetLongField(buffer, classes_field);
	cl_int err;

	cl_command_queue queue = get_global_context().opencl_context().get_queue(get_global_context().opencl_context().current_device().id(), DATA_TRANSFER_QUEUE).handle().get();
	err = clEnqueueSVMUnmap(queue, ptr, 0, 0, NULL);
	VIENNACL_ERR_CHECK(err);
	err = clEnqueueSVMUnmap(queue, class_ptr, 0, 0, NULL);
	VIENNACL_ERR_CHECK(err);
	clFinish(queue);

}
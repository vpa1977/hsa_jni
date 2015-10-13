
#ifdef VIENNACL_WITH_OPENCL
#include <CL/cl.h>
#endif
#include "library.hpp"
#include "org_moa_gpu_bridge_DenseOffHeapBuffer.h"

#include "offheap_helpers.hpp"

const char* const DENSE_OFFHEAP_BUFFER = "org/moa/gpu/bridge/DenseOffHeapBuffer";

/*
* Class:     org_moa_gpu_bridge_DenseOffHeapBuffer
* Method:    allocate
* Signature: (J)J
*/
JNIEXPORT jlong JNICALL Java_org_moa_gpu_bridge_DenseOffHeapBuffer_allocate
(JNIEnv *env, jobject obj, jlong size_in_bytes)
{
#ifdef VIENNACL_WITH_HSA
    void* ptr= malloc(size_in_bytes);
    return (jlong)ptr;
#else    
	cl_int err;
	viennacl::ocl::context& ctx = get_global_context().opencl_context();
	cl_device_id device_id = ctx.current_device().id();
	const cl_context& cl_ctx = ctx.handle().get();
	void * ptr = clSVMAlloc(cl_ctx, CL_MEM_READ_WRITE, size_in_bytes, 0);
	viennacl::ocl::command_queue& cmd_queue = ctx.get_queue(device_id, DATA_TRANSFER_QUEUE);
	cl_command_queue queue = cmd_queue.handle().get();
	err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, ptr, size_in_bytes, 0, NULL, NULL);
	VIENNACL_ERR_CHECK(err);
	return (jlong)ptr;
#endif        
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
#ifdef VIENNACL_WITH_HSA
        free(ptr);
#else        
	const cl_context& cl_ctx = get_global_context().opencl_context().handle().get();
	clSVMFree(cl_ctx, ptr);
#endif        
}

/*
* Class:     org_moa_gpu_bridge_DenseOffHeapBuffer
* Method:    begin
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_DenseOffHeapBuffer_begin
(JNIEnv *env, jobject buffer)
{
#ifdef VIENNACL_WITH_HSA    
#else
	static jclass clazz = env->FindClass(DENSE_OFFHEAP_BUFFER);
	static jfieldID field = env->GetFieldID(clazz, "m_buffer", "J");
	static jfieldID size_field = env->GetFieldID(clazz, "m_size", "J");
	void * ptr = (void*)env->GetLongField(buffer, field);
	cl_int err;
	jlong size = env->GetLongField(buffer, size_field);
	cl_command_queue queue = get_global_context().opencl_context().get_queue(get_global_context().opencl_context().current_device().id(), DATA_TRANSFER_QUEUE).handle().get();
	err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, ptr, size, 0, NULL, NULL);
	VIENNACL_ERR_CHECK(err);
#endif
}

/*
* Class:     org_moa_gpu_bridge_DenseOffHeapBuffer
* Method:    commit
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_DenseOffHeapBuffer_commit
(JNIEnv * env, jobject buffer)
{
#ifdef VIENNACL_WITH_HSA
#else
	static jclass clazz = env->FindClass(DENSE_OFFHEAP_BUFFER);
	static jfieldID field = env->GetFieldID(clazz, "m_buffer", "J");
	static jfieldID classes_field = env->GetFieldID(clazz, "m_class_buffer", "J");
//	static jfieldID size_field = env->GetFieldID(clazz, "m_size", "J");
	void * ptr = (void*)env->GetLongField(buffer, field);
	void * class_ptr = (void*)env->GetLongField(buffer, classes_field);
	cl_int err;

	cl_command_queue queue = get_global_context().opencl_context().get_queue(get_global_context().opencl_context().current_device().id(), DATA_TRANSFER_QUEUE).handle().get();
	err = clEnqueueSVMUnmap(queue, ptr, 0, 0, NULL);
	VIENNACL_ERR_CHECK(err);
	err = clEnqueueSVMUnmap(queue, class_ptr, 0, 0, NULL);
	VIENNACL_ERR_CHECK(err);
	clFinish(queue);
#endif        
}
#include "org_moa_gpu_bridge_SparseOffHeapBuffer.h"
#ifdef VIENNACL_WITH_OPENCL
#include <CL/cl.h>
#endif
#include "library.hpp"


#ifdef VIENNACL_WITH_HSA
#include <hsa.h>
#endif


const char* const SPARSE_OFFHEAP_BUFFER = "org/moa/gpu/bridge/SparseOffHeapBuffer";
/*
* Class:     org_moa_gpu_bridge_SparseOffHeapBuffer
* Method:    allocate
* Signature: (J)J
*/
JNIEXPORT jlong JNICALL Java_org_moa_gpu_bridge_SparseOffHeapBuffer_allocate
(JNIEnv *env, jobject obj, jlong size_in_bytes)
{
#ifdef VIENNACL_WITH_HSA
	void* ptr = malloc(size_in_bytes);
//	hsa_memory_register(ptr, size_in_bytes);
	return (jlong)ptr;
#else
	cl_int err;
	const cl_context& cl_ctx = get_global_context().opencl_context().handle().get();
	void * ptr = clSVMAlloc(cl_ctx, CL_MEM_READ_WRITE, size_in_bytes, 0);
	if (ptr == NULL)
	{
		std::cout << "memory allocation for " << size_in_bytes << " failed " << std::endl;
	}
	cl_command_queue queue = get_global_context().opencl_context().get_queue(get_global_context().opencl_context().current_device().id(), DATA_TRANSFER_QUEUE).handle().get();
	err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, ptr, size_in_bytes, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		std::cout << "cl error memory allocation for " << err << " failed " << std::endl;
	}
	VIENNACL_ERR_CHECK(err);
	return (jlong)ptr;
#endif
}

/*
* Class:     org_moa_gpu_bridge_SparseOffHeapBuffer
* Method:    release
* Signature: (J)V
*/
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_SparseOffHeapBuffer_release
(JNIEnv *, jobject, jlong handle)
{
#ifdef VIENNACL_WITH_HSA
	void* ptr = (void*)handle;
    free(ptr);
#else
	void* ptr = (void*)handle;
	const cl_context& cl_ctx = get_global_context().opencl_context().handle().get();
	clSVMFree(cl_ctx, ptr);
#endif        
        
}


JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_SparseOffHeapBuffer_nativeBegin
(JNIEnv *, jobject, jlong handle, jlong size_in_bytes)
{
#ifdef VIENNACL_WITH_HSA
#else
	cl_int err;
	void * ptr = (void*)handle;
	cl_command_queue queue = get_global_context().opencl_context().get_queue(get_global_context().opencl_context().current_device().id(), DATA_TRANSFER_QUEUE).handle().get();
	err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, ptr, size_in_bytes, 0, NULL, NULL);
	VIENNACL_ERR_CHECK(err);
#endif        

}

static void svmUnmap(JNIEnv* env, jobject obj, cl_command_queue queue, jfieldID field)
{
	cl_int err;
	void * ptr = (void*)env->GetLongField(obj, field);
	err = clEnqueueSVMUnmap(queue, ptr, 0, 0, NULL);
	VIENNACL_ERR_CHECK(err);
}

/*
* Class:     org_moa_gpu_bridge_SparseOffHeapBuffer
* Method:    nativeCommit
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_SparseOffHeapBuffer_nativeCommit
(JNIEnv * env, jobject buffer)
{
#ifdef VIENNACL_WITH_HSA    
#else    
	static jclass clazz = env->FindClass(SPARSE_OFFHEAP_BUFFER);

	static jfieldID m_row_jumper = env->GetFieldID(clazz, "m_row_jumper", "J");
	static jfieldID m_column_data = env->GetFieldID(clazz, "m_column_data", "J");
	static jfieldID m_element_data = env->GetFieldID(clazz, "m_element_data", "J");
	static jfieldID classes_field = env->GetFieldID(clazz, "m_class_buffer", "J");
	
	

	cl_command_queue queue = get_global_context().opencl_context().get_queue(get_global_context().opencl_context().current_device().id(), DATA_TRANSFER_QUEUE).handle().get();
	svmUnmap(env, buffer, queue, m_row_jumper);
	svmUnmap(env, buffer, queue, m_column_data);
	svmUnmap(env, buffer, queue, m_element_data);
	svmUnmap(env, buffer, queue, classes_field);

	clFinish(queue);
#endif        

}

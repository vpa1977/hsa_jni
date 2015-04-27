/*
 * MOA frontend to viennacl ml library
 *
 *  Created on: 24/04/2015
 *      Author: bsp
 */

#include "library.hpp"

viennacl::context g_context;

viennacl::context& get_global_context()
{
	return g_context;
}

void write_pointer(JNIEnv* env , void * ptr, jbyteArray dst)
{
	size_t len = sizeof(void*);
	size_t arr_len = env->GetArrayLength(dst);
	assert(arr_len > len);
	env->SetByteArrayRegion(dst,0, len, reinterpret_cast<jbyte*>(&ptr));
}

void read_pointer(JNIEnv* env, jbyteArray src, void** ptr)
{
	env->GetByteArrayRegion(src, 0, sizeof(void*), reinterpret_cast<jbyte*>(ptr));
}


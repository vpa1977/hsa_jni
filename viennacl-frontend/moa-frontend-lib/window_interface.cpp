/*
 * window_interface.cpp
 *
 *  Created on: 27/04/2015
 *      Author: bsp
 */
#include <assert.h>
#include "window_interface.hpp"

std::vector<instance_interface> get_window(JNIEnv* env, jobject window)
{
	static jclass window_class = env->FindClass("org/moa/gpu/Window");
	static jmethodID test = env->GetMethodID(window_class, "clear","()V" );

	static jmethodID get_method_id = env->GetMethodID(window_class, "get", "()[Lweka/core/Instance;");
	assert(get_method_id != 0);
	jobjectArray array = (jobjectArray)env->CallObjectMethod(window, get_method_id);
	size_t len = env->GetArrayLength(array);
	std::vector<instance_interface> result;
	for (size_t i = 0 ; i < len ; ++i)
	{
		jobject instance = env->GetObjectArrayElement(array, i);
		result.push_back(instance_interface(env,instance));
	}
	return result;
}



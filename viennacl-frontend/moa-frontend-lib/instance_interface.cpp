/*
 * instance_interface.cpp
 *
 *  Created on: 27/04/2015
 *      Author: bsp
 */

#include "instance_interface.hpp"

#include <assert.h>

instance_interface::instance_interface(JNIEnv* env, jobject inst) : env_(env), instance_(inst)
{
	static jclass instance_class = env->FindClass("weka/core/Instance");
	assert(instance_class != 0);
	static jmethodID get_method = env->GetMethodID(instance_class, "value", "(I)D");
	assert(get_method != 0);
	get_method_ = get_method;
	static jmethodID class_index_method = env->GetMethodID(instance_class, "classIndex", "()I");
	class_index_method_= class_index_method;

	static jmethodID is_missing_method = env->GetMethodID(instance_class, "isMissing","(I)Z");
	is_missing_method_ = is_missing_method;
}

double instance_interface::get_attribute(int index)
{
	return env_->CallDoubleMethod(instance_, get_method_, index);
}

int instance_interface::get_class_index()
{
	return env_->CallIntMethod(instance_, class_index_method_);
}

bool instance_interface::is_missing( int index)
{
	return env_->CallBooleanMethod(instance_, is_missing_method_, index);
}



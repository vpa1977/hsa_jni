/*
 * instance_interface.hpp
 *
 *  Created on: 27/04/2015
 *      Author: bsp
 */

#ifndef INSTANCE_INTERFACE_HPP_
#define INSTANCE_INTERFACE_HPP_

#include <jni.h>

class instance_interface
{
public:
	instance_interface(JNIEnv* env,jobject inst);
	double get_attribute(int index);
	int get_class_index();
	bool is_missing( int index);
private:
	JNIEnv* env_;
	jobject instance_;
private:
	jmethodID get_method_;
	jmethodID class_index_method_;
	jmethodID is_missing_method_;

};


#endif /* INSTANCE_INTERFACE_HPP_ */

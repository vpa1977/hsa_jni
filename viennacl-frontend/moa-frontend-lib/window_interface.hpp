/*
 * window_interface.hpp
 * Interface to the Window class
 *  Created on: 27/04/2015
 *      Author: bsp
 */

#ifndef WINDOW_INTERFACE_HPP_
#define WINDOW_INTERFACE_HPP_

#include <jni.h>
#include <vector>
#include "instance_interface.hpp"

std::vector<instance_interface> get_window(JNIEnv* env, jobject window);


#endif /* WINDOW_INTERFACE_HPP_ */

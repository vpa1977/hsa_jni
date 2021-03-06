/*
 * library.hpp
 *
 *  Created on: 26/04/2015
 *      Author: bsp
 */

#ifndef LIBRARY_HPP_
#define LIBRARY_HPP_

#define EXECUTION_QUEUE 0
#define DATA_TRANSFER_QUEUE 1
#define DEVICE_QUEUE 2

#include <jni.h>
#include "viennacl/context.hpp"

viennacl::context& get_global_context();
void write_pointer(JNIEnv* env , void * ptr, jbyteArray dst);
void read_pointer(JNIEnv* env, jbyteArray src, void **ptr_dst);
void fill_sparse(viennacl::vector<double>& vcl_vector, jlong values, jlong indices, jlong ind_len, jlong total_len );
void fill_dense(viennacl::vector<double>& vcl_vector, jlong values, jlong total_len );

cl_mem write_with_data_queue(void* source, size_t len);
void write_with_data_queue(cl_mem dst, void* source, size_t len);


#endif /* LIBRARY_HPP_ */

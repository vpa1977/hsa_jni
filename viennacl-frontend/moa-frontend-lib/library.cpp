/*
 * MOA frontend to viennacl ml library
 *
 *  Created on: 24/04/2015
 *      Author: bsp
 */

#include "library.hpp"

#include <boost/numeric/ublas/vector.hpp>
#include <viennacl/vector.hpp>

 //viennacl::context g_context(viennacl::MAIN_MEMORY);// =   viennacl::ocl::current_context();


viennacl::context& get_global_context()
{
	static viennacl::context g_context =   viennacl::ocl::current_context();
	static bool init = false;
	if (!init)
	{
		viennacl::ocl::context* ctx = g_context.opencl_pcontext();
		ctx->build_options("-cl-std=CL2.0 -D CL_VERSION_2_0");
	}
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


void fill_sparse(viennacl::vector<double>& vcl_vector, jlong values, jlong indices, jlong ind_len, jlong total_len )
{
	boost::numeric::ublas::vector<double> cpu_instance(total_len);
	long * ind = (long*)indices;
	double* val = (double*)values;
	for (int i = 0; i < ind_len ; ++i)
	{
		cpu_instance(ind[i]) = val[i];
	}

	viennacl::copy(cpu_instance.begin(),cpu_instance.end(), vcl_vector.begin());
}


void fill_dense(viennacl::vector<double>& vcl_vector, jlong values, jlong total_len )
{
	boost::numeric::ublas::vector<double> cpu_instance(total_len);
	double* val = (double*)values;
	for (int i = 0; i < total_len ; ++i)
	{
		cpu_instance(i) = val[i];
	}

	viennacl::copy(cpu_instance.begin(),cpu_instance.end(), vcl_vector.begin());
}



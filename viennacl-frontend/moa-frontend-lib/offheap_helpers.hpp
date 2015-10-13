#ifndef OFFHEAP_HELPERS_HPP
#define OFFHEAP_HELPERS_HPP

#include <CL/cl.h>
#include <jni.h>
#include <viennacl/vector.hpp>
#include <viennacl/compressed_matrix.hpp>
#include "library.hpp"

struct dense_offheap_buffer {
    typedef double value_type;

    dense_offheap_buffer(JNIEnv *env, jobject buffer) {
        const char* const DENSE_OFFHEAP_BUFFER = "org/moa/gpu/bridge/DenseOffHeapBuffer";
        static jclass clazz = env->FindClass(DENSE_OFFHEAP_BUFFER);
        static jfieldID field = env->GetFieldID(clazz, "m_buffer", "J");
        static jfieldID classes_field = env->GetFieldID(clazz, "m_class_buffer", "J");
        static jfieldID size_field = env->GetFieldID(clazz, "m_size", "J");
        static jfieldID rows_field = env->GetFieldID(clazz, "m_rows", "J");
        
        cl_int err;
        m_size = (long)env->GetLongField(buffer, size_field);
        long rows = (long)env->GetLongField(buffer, rows_field);
        m_class_size = rows * sizeof (value_type);
        void* class_data = (void*) env->GetLongField(buffer, classes_field);
        void* data = (void*) env->GetLongField(buffer, field);

        m_class_vector.resize(rows);
        memcpy(&m_class_vector[0], class_data, rows * sizeof (value_type));
#ifdef VIENNACL_WITH_HSA
        m_class_mem = (value_type*)class_data;
        m_value_mem = (value_type*) data;
#else
        const cl_context& ctx = get_global_context().opencl_context().handle().get();
        m_class_mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, rows * sizeof (value_type), class_data, &err);
        VIENNACL_ERR_CHECK(err);
        m_value_mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, m_size, data, &err);
        VIENNACL_ERR_CHECK(err);
        
#endif        
    }

    ~dense_offheap_buffer() {
#ifdef VIENNACL_WITH_HSA
#else
        clReleaseMemObject(m_class_mem);
        clReleaseMemObject(m_value_mem);
#endif        
    }

    std::vector<double>& class_values_std() {
        return m_class_vector;
    }

    viennacl::vector<double> class_values() {
#ifdef VIENNACL_WITH_HSA        
      return viennacl::vector<value_type>(m_class_mem,viennacl::HSA_MEMORY, m_class_size / sizeof (value_type), 0, 1);
#else
      return viennacl::vector<value_type>(m_class_mem, m_class_size / sizeof (value_type), 0, 1);
#endif        
    }

    viennacl::vector<double> values() {
#ifdef VIENNACL_WITH_HSA
     return viennacl::vector<value_type>(m_value_mem,viennacl::HSA_MEMORY, m_size / sizeof (value_type), 0, 1);
#else
      return viennacl::vector<value_type>(m_value_mem, m_size / sizeof (value_type), 0, 1);
#endif        
    }
    std::vector<double> m_class_vector;
#ifdef VIENNACL_WITH_HSA
    value_type* m_class_mem;
    value_type* m_value_mem;

#else
    cl_mem m_class_mem;
    cl_mem m_value_mem;
#endif    
    long m_size;
    long m_class_size;
};

struct sparse_offheap_buffer {
    typedef double value_type;

    sparse_offheap_buffer(JNIEnv *env, jobject buffer) {
        const char* const SPARSE_OFFHEAP_BUFFER = "org/moa/gpu/bridge/SparseOffHeapBuffer";
        static jclass clazz = env->FindClass(SPARSE_OFFHEAP_BUFFER);
        static jfieldID classes_field = env->GetFieldID(clazz, "m_class_buffer", "J");
        static jfieldID columns_field = env->GetFieldID(clazz, "m_columns", "J");
        static jfieldID element_count_field = env->GetFieldID(clazz, "m_element_count", "J");
        static jfieldID rows_field = env->GetFieldID(clazz, "m_rows", "J");

        static jfieldID row_jumper_field = env->GetFieldID(clazz, "m_row_jumper", "J");
        static jfieldID column_data_field = env->GetFieldID(clazz, "m_column_data", "J");
        static jfieldID element_data_field = env->GetFieldID(clazz, "m_element_data", "J");

        m_rows = (long) env->GetLongField(buffer, rows_field);
        m_cols = (long) env->GetLongField(buffer, columns_field);
        m_elements = (long) env->GetLongField(buffer, element_count_field);

        void* class_data = (void*) env->GetLongField(buffer, classes_field);
        void* element_data = (void*) env->GetLongField(buffer, element_data_field);
        void* row_jumper = (void*) env->GetLongField(buffer, row_jumper_field);
        void* column_data = (void*) env->GetLongField(buffer, column_data_field);
#ifdef VIENNACL_WITH_HSA
        m_class_mem = class_data;
        m_element_mem = element_data;
        m_row_mem = row_jumper;
        m_col_mem = column_data;
#else
        const cl_context& ctx = get_global_context().opencl_context().handle().get();
        cl_int err;
        m_class_mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, m_rows * sizeof (value_type), class_data, &err);
        VIENNACL_ERR_CHECK(err);

        m_element_mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, m_elements * sizeof (value_type), element_data, &err);
        VIENNACL_ERR_CHECK(err);
        m_row_mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (m_rows + 1) * sizeof (cl_uint), row_jumper, &err);
        VIENNACL_ERR_CHECK(err);
        m_col_mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, m_elements * sizeof (cl_uint), column_data, &err);
        VIENNACL_ERR_CHECK(err);
#endif                

    }

    ~sparse_offheap_buffer() {
#ifdef VIENNACL_WITH_HSA
#else
        clReleaseMemObject(m_class_mem);
        clReleaseMemObject(m_element_mem);
        clReleaseMemObject(m_row_mem);
        clReleaseMemObject(m_col_mem);
#endif                
    }

    viennacl::vector<double> class_values() {
#ifdef VIENNACL_WITH_HSA
        return viennacl::vector<value_type>((value_type*) m_class_mem, viennacl::HSA_MEMORY, m_rows, 0, 1);
#else
        return viennacl::vector<value_type>(m_class_mem, m_rows, 0, 1);
#endif            
    }

    viennacl::compressed_matrix<value_type> values() {
#ifdef VIENNAC_WITH_HSA
			return viennacl::compressed_matrix<value_type>(m_row_mem, m_col_mem, m_element_mem, m_rows, m_cols, m_elements);
#else
      return viennacl::compressed_matrix<value_type>(m_row_mem, m_col_mem, m_element_mem, m_rows, m_cols, m_elements, get_global_context());
#endif
    }
#ifdef VIENNACL_WITH_HSA
    void * m_class_mem;
    void * m_element_mem;
    void * m_row_mem;
    void * m_col_mem;

#else        
    cl_mem m_class_mem;
    cl_mem m_element_mem;
    cl_mem m_row_mem;
    cl_mem m_col_mem;
#endif        

    long m_rows;
    long m_cols;
    long m_elements;
    long m_class_size;

};


#endif
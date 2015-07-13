#include "library.hpp"
#include "instance_interface.hpp"
#include "org_moa_gpu_bridge_NativeSparseInstanceBatch.h"
#include "viennacl/backend/mem_handle.hpp"
#include "native_instance_batch.hpp"



void sparse_instance_batch::commit()
{
	static std::vector<int> row_jumper;
	row_jumper.resize(m_num_rows+1);
	static std::vector<double> class_data;
	class_data.resize(m_num_rows);
	// fill row jumper
	int accumulator = 0;
	for (size_t i = 0; i < m_individual_instances.size() ; ++i)
	{
		sparse_storage& p_row = m_individual_instances.at(i);
		row_jumper.at(i) = accumulator;
		accumulator += p_row.m_indices.size();
		class_data.at(i)= p_row.m_class_value;
	}
	row_jumper.at(m_num_rows) = accumulator;

	static std::vector<int> column_data;
	column_data.resize( accumulator );
	static std::vector<double> element_data;
	element_data.resize( accumulator);
	for (size_t i = 0; i < m_individual_instances.size() ; ++i)
	{
		sparse_storage& p_row = m_individual_instances.at(i);
		std::copy(p_row.m_indices.begin(), p_row.m_indices.end(), column_data.begin() + row_jumper.at(i));
		std::copy(p_row.m_values.begin(), p_row.m_values.end(), element_data.begin() + row_jumper.at(i));
	}
	m_instance_values.set((const void*)&row_jumper[0],(const void*)&column_data[0],(const double*) &element_data[0], m_num_rows, m_num_columns, element_data.size());

#ifdef NATIVE_INSTANCE_VALIDATE_FILL
	viennacl::compressed_matrix<double> test_matrix(m_num_rows, m_num_columns, get_global_context());
	test_matrix.clear();
	for (size_t i = 0; i < m_individual_instances.size() ; ++i)
	{
		sparse_storage& p_row = m_individual_instances.at(i);
		for (int col = 0; col < p_row.m_indices.size(); ++col)
		{
			test_matrix(i, p_row.m_indices[col]) = p_row.m_values.at(col);
		}

	}

	viennacl::backend::mem_handle::ram_handle_type ram =  test_matrix.handle2().ram_handle();
	int* good_columns = (int*)ram.get();
	int* bad_columns = (int*)m_instance_values.handle2().ram_handle().get();
	for (int i = 0 ;i < accumulator ; ++i)
		if (good_columns[i]!= bad_columns[i])
			printf("break now");

#endif
	viennacl::copy(class_data, m_class_values);
}



/*
 * Class:     org_moa_gpu_bridge_NativeSparseInstanceBatch
 * Method:    add
 * Signature: (Lorg/moa/gpu/bridge/NativeInstance;)V
 */
JNIEXPORT jboolean JNICALL Java_org_moa_gpu_bridge_NativeSparseInstanceBatch_add(JNIEnv * env, jobject instance_batch, jobject native_instance)
{
	jboolean full = 0;
	static jclass _class = env->FindClass("org/moa/gpu/bridge/NativeSparseInstanceBatch");
	static jfieldID _context_field = env->GetFieldID(_class, "m_native_context", "J");
	sparse_instance_batch* batch = (sparse_instance_batch*)env->GetLongField(instance_batch, _context_field);

	static jclass _instance_class = env->FindClass("org/moa/gpu/bridge/NativeSparseInstance");
	static jfieldID _instance_context_field = env->GetFieldID(_instance_class, "m_native_context", "J");
	sparse_storage* storage = (sparse_storage*)env->GetLongField(native_instance, _instance_context_field);

	if (batch->add(storage))
		full = 1;

	return full;
}

/*
 * Class:     org_moa_gpu_bridge_NativeSparseInstanceBatch
 * Method:    clear
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeSparseInstanceBatch_clear
  (JNIEnv * env, jobject instance_batch)
{
	static jclass _class = env->FindClass("org/moa/gpu/bridge/NativeSparseInstanceBatch");
	static jfieldID _context_field = env->GetFieldID(_class, "m_native_context", "J");
	sparse_instance_batch* batch = (sparse_instance_batch*)env->GetLongField(instance_batch, _context_field);
	batch->clear();
}

/*
 * Class:     org_moa_gpu_bridge_NativeSparseInstanceBatch
 * Method:    init
 * Signature: (Lweka/core/Instances;I)V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeSparseInstanceBatch_init (JNIEnv * env, jobject instance_batch, jobject instances, jint num_rows)
{
	static jclass _class = env->FindClass("org/moa/gpu/bridge/NativeSparseInstanceBatch");
	static jfieldID _context_field = env->GetFieldID(_class, "m_native_context", "J");
	dataset_interface dataset(env,instances);
	sparse_instance_batch* batch = new sparse_instance_batch(num_rows, dataset.get_num_attributes()-1,get_global_context());
	env->SetLongField(instance_batch, _context_field, (jlong)batch);
}

/*
 * Class:     org_moa_gpu_bridge_NativeSparseInstanceBatch
 * Method:    release
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_gpu_bridge_NativeSparseInstanceBatch_release(JNIEnv * env, jobject instance_batch)
{
	static jclass _class = env->FindClass("org/moa/gpu/bridge/NativeSparseInstanceBatch");
	static jfieldID _context_field = env->GetFieldID(_class, "m_native_context", "J");
	sparse_instance_batch* batch = (sparse_instance_batch*)env->GetLongField(instance_batch, _context_field);
	delete batch;

}

package org.moa.gpu;

import org.moa.gpu.util.DirectMemory;

import weka.core.Instance;

public class SimpleDirectMemoryBatchInstances implements BatchInstances {
	
	private int m_size; // size of the window. 
	private int m_column_count; // number of columns
	
	private long m_classes;
	private long m_rows;
	private long m_column_indices;
	private long m_number_of_elements;

	private int m_index;
	private boolean m_full;

	public SimpleDirectMemoryBatchInstances(int size)
	{
		m_size = size;
		m_rows = DirectMemory.allocate(size * DirectMemory.LONG_SIZE);
		assert(m_rows != 0);
		DirectMemory.set(m_rows, size* DirectMemory.LONG_SIZE,(byte) 0);
		
		m_number_of_elements = DirectMemory.allocate(size * DirectMemory.LONG_SIZE);
		assert(m_number_of_elements != 0);
		
		
		m_column_indices = DirectMemory.allocate(size * DirectMemory.LONG_SIZE);
		assert(m_column_indices != 0);
		DirectMemory.set(m_column_indices, size* DirectMemory.LONG_SIZE,(byte) 0);
		
		m_classes = DirectMemory.allocate( size * DirectMemory.DOUBLE_SIZE);
		assert(m_classes != 0);
		clear();
	}
	
	protected void finalize() throws Throwable 
	{
		clear();
		DirectMemory.free( m_rows );
		DirectMemory.free( m_classes );
		DirectMemory.free( m_column_indices );
		DirectMemory.free( m_number_of_elements);
	};

	@Override
	public void clear() {
		for (int i = 0 ; i < m_size ; ++i)
		{
			
			long handle = DirectMemory.read(m_rows, i);
			if (handle!=0)
			{
				assert(m_full);
				DirectMemory.free( handle);
			}
			handle = DirectMemory.read(m_column_indices, i);
			if (handle!=0)
			{
				assert(m_full);
				DirectMemory.free( handle);
			}

		}
		DirectMemory.set(m_rows, m_size* DirectMemory.LONG_SIZE,(byte) 0);
		DirectMemory.set(m_column_indices, m_size* DirectMemory.LONG_SIZE,(byte) 0);
		DirectMemory.set(m_number_of_elements, m_size* DirectMemory.LONG_SIZE,(byte) 0);
		DirectMemory.set(m_classes, m_size* DirectMemory.LONG_SIZE,(byte) 0);

		m_index = 0;
		m_full = false;
	}

	@Override
	public void add(Instance inst) {
		
		long old_handle = DirectMemory.read(m_rows, m_index);
		if (old_handle != 0)
		{
			assert( m_full );
			System.out.println("before free old handle - row data- index "+ m_index);
			DirectMemory.free(old_handle);
			System.out.println("after free old handle - row data");
		}
		long old_handle_columns = DirectMemory.read(m_column_indices, m_index);
		if (old_handle_columns!= 0)
		{
			assert( m_full );
			assert( old_handle != 0);
			System.out.println("before free old handle - column indices - index "+ m_index);
			DirectMemory.free(old_handle_columns);
			System.out.println("after free old handle - column indices");
		}
		m_column_count = inst.numAttributes() -1;
		AccessibleSparseInstance sparse = new AccessibleSparseInstance(inst);
		int[] indices = sparse.getIndexes();
		double[] values = sparse.getValues();
		
		assert(indices.length > 1);// 1 attribute is a class value
		
		long attribute_size =  sparse.classValue() == 0 ? indices.length : indices.length -1;
		
		long row_data = DirectMemory.allocate(attribute_size * DirectMemory.DOUBLE_SIZE);
		assert(row_data != 0);
		
		long row_column_indices = DirectMemory.allocate(attribute_size * DirectMemory.LONG_SIZE);
		assert(row_column_indices != 0);
		
		DirectMemory.write(m_number_of_elements, m_index, attribute_size);
		int idx = 0;
		for (int i = 0 ;i < values.length; ++i)
		{
			if (indices[i] == inst.classIndex())
				continue;
			long offset = indices[i] > inst.classIndex() ? indices[i] - 1 : indices[i];
			DirectMemory.write(row_column_indices, idx, offset);
			DirectMemory.write(row_data, idx, values[i]);
			idx ++;
		}
		
		DirectMemory.write(m_rows, m_index, row_data);
		DirectMemory.write(m_column_indices, m_index, row_column_indices);
		DirectMemory.write(m_classes, m_index, sparse.classValue());
		
		m_index++;
		if (m_index >= m_size )
		{
			m_index = 0;
			m_full = true;
		}
		
	}

	@Override
	public Instance[] get() {
		return null;
	}

	@Override
	public void add(Instance inst, long t) {
		add(inst);
		
	}

	@Override
	public boolean full() {
		return m_full;
	}

	
	public long rows()
	{
		return m_rows;
	}

	public long rowNumberOfElements() 
	{
		return m_number_of_elements;
	}
	
	public long columnIndices()
	{
		return m_column_indices;
	}
	
	public long classes()
	{
		return m_classes;
	}


	@Override
	public int getRowCount() {
		return m_size;
	}

	@Override
	public int getColumnCount() {
		return m_column_count;
	}
}

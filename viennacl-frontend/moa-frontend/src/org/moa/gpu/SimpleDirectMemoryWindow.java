package org.moa.gpu;

import java.util.ArrayList;

import org.moa.gpu.util.DirectMemory;

import weka.core.Instance;

public class SimpleDirectMemoryWindow implements Window {
	
	private int m_size; // size of the window. 
	private int m_column_count; // number of columns
	
	private long m_rows;
	private long m_classes;
	private long m_element_count;

	private int m_index;
	private boolean m_full;

	public SimpleDirectMemoryWindow(int size)
	{
		m_size = size;
		m_rows = DirectMemory.allocate(size * DirectMemory.LONG_SIZE);
		DirectMemory.set(m_rows, size* DirectMemory.LONG_SIZE,(byte) 0);
		m_element_count = DirectMemory.allocate(size * DirectMemory.LONG_SIZE);
		DirectMemory.set(m_element_count, size* DirectMemory.LONG_SIZE,(byte) 0);
		m_classes = DirectMemory.allocate( size * DirectMemory.DOUBLE_SIZE);
		clear();
	}
	
	protected void finalize() throws Throwable 
	{
		clear();
		DirectMemory.free( m_rows );
		DirectMemory.free( m_classes );
		DirectMemory.free( m_element_count );
	};

	@Override
	public void clear() {
		for (int i = 0 ; i < m_size ; ++i)
		{
			long handle = DirectMemory.read(m_rows, i);
			if (handle!=0)
				DirectMemory.free( handle);
			handle = DirectMemory.read(m_element_count, i);
			if (handle!=0)
				DirectMemory.free( handle);

			DirectMemory.write(m_rows,  i,  (long)0);
			DirectMemory.write(m_classes, i, (long)0);
			DirectMemory.write(m_element_count,i, (long)0);
		}
		m_index = 0;
		m_full = false;
	}

	@Override
	public void add(Instance inst) {
		
		long oldHandle = DirectMemory.read(m_rows, m_index);
		if (oldHandle != 0)
		{
			assert( m_full );
			DirectMemory.free(oldHandle);
		}
		oldHandle = DirectMemory.read(m_element_count, m_index);
		if (oldHandle!= 0)
		{
			assert( m_full );
			DirectMemory.free(oldHandle);
		}
		m_column_count = inst.numAttributes() -1;
		AccessibleSparseInstance sparse = new AccessibleSparseInstance(inst);
		int[] indices = sparse.getIndexes();
		double[] values = sparse.getValues();
		assert(indices.length > 0);
		
		long rowHandle = DirectMemory.allocate(indices.length * DirectMemory.DOUBLE_SIZE);
		assert(rowHandle != 0);
		
		long rowElementHandle = DirectMemory.allocate((indices.length+1) * DirectMemory.LONG_SIZE);
		assert(rowElementHandle != 0);
		
		DirectMemory.write(rowElementHandle, 0, (long)indices.length);
		for (int i = 0 ;i < values.length; ++i)
		{
			if (i == inst.classIndex())
				continue;
			
			DirectMemory.write(rowHandle, i, values[i]);
			long offset = indices[i];
			if (offset > inst.classIndex())
					--offset; 
			DirectMemory.write(rowElementHandle, i, offset);
		}
		DirectMemory.write(m_rows, m_index, rowHandle);
		DirectMemory.write(m_element_count, m_index, rowElementHandle);
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

	@Override
	public double[] getValues() {
		return null;
	}

	@Override
	public int[] getRowIndices() {
		return null;
	}

	@Override
	public int[] getColumnIndices() {
		return null;
	}

	@Override
	public double[] getClassValues() {
		return null;
	}
	
	public long rowHandle()
	{
		return m_rows;
	}
	
	
	public long getElementCount()
	{
		return m_element_count;
	}
	
	public long classesHandle()
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

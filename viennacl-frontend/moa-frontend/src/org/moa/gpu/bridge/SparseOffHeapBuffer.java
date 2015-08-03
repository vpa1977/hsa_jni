package org.moa.gpu.bridge;

import org.moa.gpu.util.DirectMemory;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

/**
 * Implementation of the CSR matrix offheap buffer
 */
public class SparseOffHeapBuffer {
	private long m_rows;
	private long m_columns;
	private long m_element_count;
	private long m_class_buffer;
	private long m_row_jumper;
	private long m_column_data;
	private long m_element_data;
	private Instance[] m_instances;
	
	
	
	public SparseOffHeapBuffer(int rows, int columns)
	{
		m_rows = rows;
		m_columns = columns;
		m_class_buffer = allocate(rows * DirectMemory.DOUBLE_SIZE);
		m_row_jumper = allocate( (rows+1) * DirectMemory.INT_SIZE);
		m_instances = new Instance[rows];
		m_column_data = 0;
		m_element_data = 0;
		m_element_count = 0;
	}
	
	public void release() 
	{
		release(m_class_buffer);
		release(m_row_jumper);
		release(m_element_data);
		release(m_column_data);
		m_element_data = 0;
		m_column_data = 0;
		m_class_buffer = 0;
		m_row_jumper = 0;
	}
	
	protected void finalize() throws Throwable 
	{
		release();
		super.finalize();
	}
	
	public void set(Instance inst, int pos)
	{
		if (pos >= m_rows)
			throw new ArrayIndexOutOfBoundsException(pos);
		m_instances[pos] = inst;
		DirectMemory.write(m_class_buffer + pos * DirectMemory.DOUBLE_SIZE, inst.classValue());
	}
	
	
	public void commit()
	{
		int pos = 0;
		int accumulator = 0;
		for (Instance inst : m_instances) 
		{
			DirectMemory.writeInt(m_row_jumper + pos ,accumulator);
			accumulator += inst.classIsMissing() ? inst.numValues() : (inst.numValues() - 1);
			pos+= DirectMemory.INT_SIZE;
		}
		DirectMemory.writeInt(m_row_jumper + pos, accumulator);
		m_element_count = accumulator;
		
		m_column_data = allocate( accumulator * DirectMemory.INT_SIZE );
		m_element_data = allocate( accumulator * DirectMemory.DOUBLE_SIZE);
		
		int row = 0;
		
		for (Instance inst : m_instances) 
		{
			int classIndex = inst.classIndex();
			int row_offset = DirectMemory.readInt(m_row_jumper + row * DirectMemory.INT_SIZE);
			long column_data = m_column_data + row_offset * DirectMemory.INT_SIZE;
			long element_data = m_element_data + row_offset * DirectMemory.DOUBLE_SIZE; 
			long pos_column = 0;
			long pos_element = 0;
			for (int attr = 0; attr < inst.numValues(); ++attr )
			{
				int index = inst.index(attr);
				if (index == classIndex)
					continue;
				if (index > classIndex)
					--index; 
				double value = inst.valueSparse(attr);
				DirectMemory.writeInt(column_data +pos_column, index);
				DirectMemory.write  (element_data + pos_element , value);
				pos_column += DirectMemory.INT_SIZE;
				pos_element += DirectMemory.DOUBLE_SIZE;
			}
			++row;
		}
		nativeCommit();
	}
	
	
	public Instance read(Instances dataset, int pos)
	{
		double classValue = DirectMemory.read(m_class_buffer + pos * DirectMemory.DOUBLE_SIZE);
		int row_offset = DirectMemory.readInt(m_row_jumper + pos * DirectMemory.INT_SIZE);
		int next_row_offset = DirectMemory.readInt(m_row_jumper + (pos+1) * DirectMemory.INT_SIZE);
		double[] values = new double[ next_row_offset - row_offset];
		int[] indices = new int[ next_row_offset - row_offset];
		SparseInstance inst = new SparseInstance(dataset.numAttributes());
		
		inst.setDataset(dataset);
		for (int i = 0 ;i < dataset.numAttributes(); ++i)
			inst.setValue(i, 0);
		inst.setClassValue(classValue);
		
		for (int offset = 0; offset < indices.length ; ++offset)
		{
			values[offset] = DirectMemory.read(m_element_data + (row_offset + offset) * DirectMemory.DOUBLE_SIZE);
			indices[offset] = DirectMemory.readInt(m_column_data +(row_offset + offset) * DirectMemory.INT_SIZE);
			if (indices[offset] >= inst.classIndex())
				++indices[offset];
			inst.setValue(indices[offset], values[offset]);
		}
		
		return inst;
	}
	
	public void begin() 
	{
		release(m_column_data);
		release(m_element_data);
		m_column_data = 0;
		m_element_data = 0;
		nativeBegin(m_class_buffer, m_rows * DirectMemory.DOUBLE_SIZE);
		nativeBegin(m_row_jumper, (m_rows+1) * DirectMemory.INT_SIZE);
	}
	
	public native long  allocate(long size);
	public native void  release(long handle);
	
	public native void nativeBegin(long buffer, long size);
	
	public native void nativeCommit();
	
	public long classes() 
	{
		return m_class_buffer;
	}
}

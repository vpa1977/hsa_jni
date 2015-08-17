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
        private long m_max_element_count;
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
                m_max_element_count = 0;
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
			accumulator += inst.numValues();
			pos+= DirectMemory.INT_SIZE;
		}
		DirectMemory.writeInt(m_row_jumper + pos, accumulator);
                if (m_max_element_count < accumulator)
                {
                    m_max_element_count = accumulator;
                    release(m_column_data);
                    release(m_element_data);
                    m_column_data = allocate( accumulator * DirectMemory.INT_SIZE );
                    m_element_data = allocate( accumulator * DirectMemory.DOUBLE_SIZE);
                }
                else
                {
                    nativeBegin(m_column_data,accumulator * DirectMemory.INT_SIZE );
                    nativeBegin(m_element_data,accumulator * DirectMemory.DOUBLE_SIZE );
                }
                
		m_element_count = accumulator;
		int row = 0;
		
		for (Instance inst : m_instances) 
		{
			int row_offset = DirectMemory.readInt(m_row_jumper + row * DirectMemory.INT_SIZE);
			long column_data = m_column_data + row_offset * DirectMemory.INT_SIZE;
			long element_data = m_element_data + row_offset * DirectMemory.DOUBLE_SIZE; 
			SparseInstanceAccess ins = new SparseInstanceAccess(inst);
			double[] element_data_arr = ins.getValues();
			int[] index_data_arr = ins.getIndices();
			DirectMemory.writeArray(element_data, 0, element_data_arr);
			DirectMemory.writeArray(column_data, 0, index_data_arr);
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
			inst.setValue(indices[offset], values[offset]);
		}
		
		return inst;
	}
	
	public void begin() 
	{
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

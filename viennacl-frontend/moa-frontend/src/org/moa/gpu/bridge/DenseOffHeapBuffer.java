package org.moa.gpu.bridge;

import org.moa.gpu.util.DirectMemory;

import weka.core.Instance;

public class DenseOffHeapBuffer {
	
	private long m_size;
	private long m_step;
	private long m_buffer;
	private int m_rows;
	private long m_class_buffer;
	
	public DenseOffHeapBuffer(int rows, int numAttributes)
	{
		m_rows = rows;
		m_size = rows * numAttributes * DirectMemory.DOUBLE_SIZE;
		m_step = numAttributes * DirectMemory.DOUBLE_SIZE;
		m_buffer = allocate(m_size);
		m_class_buffer = allocate(rows * DirectMemory.DOUBLE_SIZE);
	}
	public void release() 
	{
		release(m_class_buffer);
		release(m_buffer);
		m_class_buffer = 0;
		m_buffer = 0;
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
		long writeIndex = pos * m_step;
		for (int attr = 0; attr < inst.numAttributes(); ++attr)
		{
			if (inst.classIndex() == attr)
				continue;
			double value = inst.value(attr);
			writeAttr(writeIndex, value);
			writeIndex += DirectMemory.DOUBLE_SIZE;
		}
		DirectMemory.write(m_class_buffer + pos * DirectMemory.DOUBLE_SIZE, inst.classValue());
	}
	
	
	public void read(Instance flyweight, int pos)
	{
		long writeIndex = pos * m_step;
		for (int attr = 0; attr < flyweight.numAttributes(); ++attr)
		{
			if (flyweight.classIndex() == attr)
				continue;
			double value = readAttr(writeIndex);
			flyweight.setValue(attr, value);
			writeIndex += DirectMemory.DOUBLE_SIZE;
		}
		double classValue = DirectMemory.read(m_class_buffer + pos * DirectMemory.DOUBLE_SIZE);
		flyweight.setClassValue(classValue);
	}
	
	
	private void writeAttr(long index, double value)
	{
		DirectMemory.write(m_buffer + index,  value);
	}
	
	private double readAttr(long index)
	{
		return DirectMemory.read(m_buffer + index);
	}
	
	public native long  allocate(long size);
	public native void  release(long handle);
	public native void begin();
	public native void commit();
	
	public  long data()
	{
		return m_buffer;
	}
	
	public long classes() 
	{
		return m_class_buffer;
	}
	

}

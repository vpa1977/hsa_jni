package org.moa.gpu.bridge;

import java.util.ArrayList;

import weka.core.Instances;

public class NativeDenseWindow {
	public NativeDenseWindow(Instances dataset, int rows)
	{
		create(dataset,rows);
	}
	
	public void create(Instances dataset, int rows)
	{
		m_list.ensureCapacity(rows);
		init(dataset, rows);
	}
	
	
	public boolean addInstance(NativeInstance inst)
	{
		
		boolean canAdd =  add(inst);
		if (canAdd)
			m_list.add(inst);
		return canAdd;
		
	}
	
	public void clearBatch()
	{
		clear();
		m_list.clear();
	}
	
	private native boolean add(NativeInstance ins);
	private native void clear();
	private native void init(Instances dataset, int rows);
	private native void release();
	public native void commit();
	
	@Override
	protected void finalize() throws Throwable {
		// TODO Auto-generated method stub
		super.finalize();
		release();
	}
	private long m_native_context;
	private ArrayList<NativeInstance> m_list = new ArrayList<NativeInstance>();
}

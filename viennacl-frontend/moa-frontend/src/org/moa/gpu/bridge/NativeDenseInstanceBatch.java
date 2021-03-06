package org.moa.gpu.bridge;

import java.util.ArrayList;

import weka.core.Instances;
/** 
 * Java interface to the batch of instances. 
 * The batch rows are updated in a circular fashion 
 * @author bsp
 *
 */
public class NativeDenseInstanceBatch implements NativeInstanceBatch {
	
	public NativeDenseInstanceBatch(Instances dataset, int rows)
	{
		throw new RuntimeException("This class is obsolete. Do not use");
		//create(dataset,rows);
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
	public native void release();
	public native void commit();
	
	private long m_native_context;
	private ArrayList<NativeInstance> m_list = new ArrayList<NativeInstance>();
	
}

package org.moa.gpu.bridge;

import weka.core.Instances;
/** 
 * Java interface to the batch of instances. 
 * The batch rows are updated in a circular fashion 
 * @author bsp
 *
 */
public class NativeSparseInstanceBatch implements NativeInstanceBatch {
	public NativeSparseInstanceBatch(Instances dataset, int rows)
	{
		throw new RuntimeException("This class is obsolete. Do not use");
		//init(dataset, rows);
	}
	
	
	
	/* (non-Javadoc)
	 * @see org.moa.gpu.bridge.NativeInstanceBatch#addInstance(org.moa.gpu.bridge.NativeInstance)
	 */
	@Override
	public boolean addInstance(NativeInstance inst)
	{
		if (! (inst instanceof NativeSparseInstance))
			throw new RuntimeException("Only NativeSparseInstance is supported");
		boolean canAdd =  add(inst);
		return canAdd;
		
	}
	
	/* (non-Javadoc)
	 * @see org.moa.gpu.bridge.NativeInstanceBatch#clearBatch()
	 */
	@Override
	public void clearBatch()
	{
		clear();
	}
	
	private native boolean add(NativeInstance ins);
	private native void clear();
	private native void init(Instances dataset, int rows);
	public native void release();
	public native void commit();
	
	@Override
	protected void finalize() throws Throwable {
		// TODO Auto-generated method stub
		super.finalize();
		clear();
	}
	private long m_native_context;
}

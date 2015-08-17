package org.moa.gpu.bridge;

public interface NativeInstanceBatch {

	public abstract boolean addInstance(NativeInstance inst);

	public abstract void clearBatch();
	
	public abstract void commit();

	public abstract void release();

}
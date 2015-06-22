package org.moa.gpu.bridge;

import weka.core.Instances;

public interface NativeInstanceBatch {

	public abstract boolean addInstance(NativeInstance inst);

	public abstract void clearBatch();

}
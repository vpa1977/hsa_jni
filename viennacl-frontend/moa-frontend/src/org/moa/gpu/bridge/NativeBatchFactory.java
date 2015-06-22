package org.moa.gpu.bridge;

import weka.core.Instances;

public interface NativeBatchFactory {
	public NativeInstanceBatch create(Instances dataset, int rows);
}

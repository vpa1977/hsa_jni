package org.moa.gpu;

import org.moa.gpu.bridge.NativeSparseInstance;
import org.moa.gpu.bridge.SparseInstanceAccess;

import weka.core.Instance;

public class NativeSparseInstanceStreamGenerator extends SparseInstanceStreamGenerator {
	public NativeSparseInstanceStreamGenerator()
	{
		throw new RuntimeException("This class is obsolete. Do not use");
	}
	public Instance nextInstance() {
		NativeSparseInstance nativeInstance = new NativeSparseInstance((SparseInstanceAccess )super.nextInstance()); 
		return nativeInstance;
	}
}

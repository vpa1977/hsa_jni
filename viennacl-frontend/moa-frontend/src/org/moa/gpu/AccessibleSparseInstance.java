package org.moa.gpu;

import weka.core.Instance;
import weka.core.SparseInstance;

public class AccessibleSparseInstance extends SparseInstance {
	public AccessibleSparseInstance(Instance i) {
		super(i);
		setDataset(i.dataset());
	}

	public int[] getIndexes() {
		return m_Indices;
	}

	public double[] getValues() {
		return m_AttValues;
	}

}
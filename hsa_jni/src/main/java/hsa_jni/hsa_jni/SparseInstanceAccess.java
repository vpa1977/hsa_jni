package hsa_jni.hsa_jni;

import weka.core.Instance;
import weka.core.SparseInstance;

public class SparseInstanceAccess extends SparseInstance {
	public SparseInstanceAccess(Instance i) {
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
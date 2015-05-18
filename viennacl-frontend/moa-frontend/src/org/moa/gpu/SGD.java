package org.moa.gpu;

import org.moa.gpu.util.DirectMemory;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.SparseInstance;
import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;

/** 
 * SGD implementation backed by the native library.
 * @author bsp
 *
 */
public class SGD extends AbstractClassifier implements NativeClassifier {
	
	private BatchInstances m_window;
	private boolean m_init;
	private int m_loss;
	protected double m_learning_rate = 0.0001;

    /** The regularization parameter */
    protected double m_lambda = 0.0001;

	public SGD(BatchInstances w, int loss)
	{
		m_window = w;
		m_loss = loss;
		m_init = false;
	}
	
	private native double[] getVotesForSparseInstance(long values,long indices, long indice_len,long total_len, BatchInstances w);
	private native double[] getVotesForDenseInstance(long values,long total_len, BatchInstances w);
	
	/** 
	 * retrain classifier on the sliding window
	 * @param w
	 */
	private native void trainNative(BatchInstances w);
	
	
	/** 
	 * initialize native context
	 * the native context pointer will be stored in m_native_context
	 * which should provide sufficient storage space
	 * @param size
	 * @param loss
	 * @param nominal
	 */
	private native void initNative(int num_attr, int batch_size, int loss, boolean nominal, double learning_rate, double lambda);
	/** 
	 * dispose native context and all natively allocated structures
	 */
	private native void dispose(); 
	
	protected void finalize() throws Throwable { dispose(); };
	
	
	

	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		int size = inst.numAttributes() - 1;
		if (inst instanceof SparseInstance)
		{
			// TODO: optimize - pre-allocate handles 
			AccessibleSparseInstance accessibleInstance = new AccessibleSparseInstance(inst);
			double[] values = accessibleInstance.getValues();
			int[] indices = accessibleInstance.getIndexes();
			int len  = indices.length-1;
			long valuesHandle = DirectMemory.allocate(DirectMemory.DOUBLE_SIZE * len);
			long indicesHandle = DirectMemory.allocate(DirectMemory.LONG_SIZE * len);
			int idx = 0;
			for (int i = 0 ; i < indices.length ; ++i)
			{
				if (indices[i] == inst.classIndex())
					continue;
				int offset = indices[i] > inst.classIndex() ? indices[i] -1 : indices[i];
				DirectMemory.write(valuesHandle, idx,  values[i]);
				DirectMemory.write(indicesHandle, idx, offset);
				idx++;
			}
			double[] result = getVotesForSparseInstance(valuesHandle, indicesHandle, len, size, m_window);
			DirectMemory.free(valuesHandle);
			DirectMemory.free(indicesHandle);
			return result;
		}
		else
		if (inst instanceof DenseInstance)
		{
			// TODO: optimize - reuse valuesHandle for dense instance
			long valuesHandle = DirectMemory.allocate(DirectMemory.DOUBLE_SIZE * size);
			for (int i = 0;i < inst.numAttributes() ; i ++ )
			{
				if (i == inst.classIndex())
					continue;
				int offset = i > inst.classIndex() ? i -1 : i;
				DirectMemory.write(valuesHandle, offset,inst.value(i));
			}
			double[] result= getVotesForDenseInstance(valuesHandle, size, m_window);
			DirectMemory.free(valuesHandle);
			return result;
			
		}
		else
			throw new RuntimeException("Unsupported instance type");
	}

	

	@Override
	public void resetLearningImpl() {
		m_window.clear();
		
		
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		m_window.add(inst);
		if (!m_init)
		{
			initNative(inst.numAttributes() -1, m_window.getRowCount(),m_loss,   inst.classAttribute().isNominal(), m_learning_rate, m_lambda);
			m_init = true;
		}
		if (m_window.full())
		{
			trainNative(m_window);
			m_window.clear();
		}
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		out.append("SGD implementation backed by viennacl ");
	}
	  
	private long m_native_context; 

}

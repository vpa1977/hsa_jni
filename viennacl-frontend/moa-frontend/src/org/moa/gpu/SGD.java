package org.moa.gpu;

import weka.core.Instance;
import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;

/** 
 * SGD implementation backed by the native library.
 * @author bsp
 *
 */
public class SGD extends AbstractClassifier implements NativeClassifier {
	
	private Window m_window;
	private boolean m_init;
	private int m_loss;

	public SGD(Window w, int loss)
	{
		m_window = w;
		m_loss = loss;
		m_init = false;
	}
	
	private native double[] getVotesForInstance(Instance inst, Window w);
	
	/** 
	 * retrain classifier on the sliding window
	 * @param w
	 */
	private native void trainNative(Window w);
	
	
	/** 
	 * initialize native context
	 * the native context pointer will be stored in m_native_context
	 * which should provide sufficient storage space
	 * @param size
	 * @param loss
	 * @param nominal
	 */
	private native void initNative(int size, int loss, boolean nominal);
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
		return getVotesForInstance(inst, m_window);
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
			initNative(inst.numAttributes() -1, m_loss, inst.classAttribute().isNominal());
			m_init = true;
		}
		if (m_window.full())
			trainNative(m_window);
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
	
	private boolean m_nominal;
	private static final int CONTEXT_SIZE = 128;  
	private byte[] m_native_context = new byte[CONTEXT_SIZE]; 

}

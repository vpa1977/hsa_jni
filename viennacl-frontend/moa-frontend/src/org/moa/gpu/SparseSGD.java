package org.moa.gpu;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import org.moa.gpu.bridge.NativeClassifier;
import org.moa.gpu.bridge.NativeInstance;
import org.moa.gpu.bridge.NativeInstanceBatch;
import org.moa.gpu.bridge.NativeSparseInstance;
import org.moa.gpu.bridge.NativeSparseInstanceBatch;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;
import weka.core.Instance;

/** 
 * SGD implementation backed by the native library.
 * @author bsp
 *
 */
public class SparseSGD extends AbstractClassifier implements NativeClassifier {
	
	private static final int QUEUE_SIZE = 10;
	private ThreadPoolExecutor m_copy_thread;
	private ThreadPoolExecutor m_train_thread;
	
	private NativeInstanceBatch m_native_batch;
	
	
    /** The regularization parameter */
    protected double m_lambda = 0.0001;

    public FloatOption lambdaRegularizationOption = new FloatOption("lambdaRegularization",
            'l', "Lambda regularization parameter .",
            0.0001, 0.00, Integer.MAX_VALUE);

    public FloatOption learningRateOption = new FloatOption("learningRate",
            'r', "Learning rate parameter.",
            0.0001, 0.00, Integer.MAX_VALUE);
    
    public MultiChoiceOption lossFunctionOption = new MultiChoiceOption(
            "lossFunction", 'o', "The loss function to use.", new String[]{
                "HINGE", "LOGLOSS", "SQUAREDLOSS"}, new String[]{
                "Hinge loss (SVM)",
                "Log loss (logistic regression)",
                "Squared loss (regression)"}, 0);
    
    
    public IntOption learningBatchSize = new IntOption("learningBatchSize", 'b', "Learning batch size", 1024, 2, Integer.MAX_VALUE);
    

    protected static final int HINGE = 0;

    protected static final int LOGLOSS = 1;

    protected static final int SQUAREDLOSS = 2;

    /** The current loss function to minimize */
    protected int m_loss = HINGE;

	protected double m_learning_rate = 0.0001;
	private int m_batch_size;
	private boolean m_native_init = false;

	
	private native double[] getVotesForSparseInstance(NativeSparseInstance inst );
	
	
	/** 
	 * Train on the next batch
	 * @param w
	 */
	private native void trainNative(NativeInstanceBatch w);
	
	
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
	
	protected void finalize() throws Throwable {
		m_copy_thread.shutdown();
		m_train_thread.shutdown();

		dispose();
		
		};
	
	
	

	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		try {
			m_copy_thread.submit(() -> {}).get();
			m_train_thread.submit(() -> {}).get();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ExecutionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		if (inst instanceof NativeSparseInstance)
			return getVotesForSparseInstance((NativeSparseInstance)inst);
		throw new RuntimeException("Unsupported instance type");
		
	}

	

	@Override
	public void resetLearningImpl() {
        m_lambda = this.lambdaRegularizationOption.getValue();
        m_learning_rate = this.learningRateOption.getValue();
        m_loss = this.lossFunctionOption.getChosenIndex();
        m_batch_size = learningBatchSize.getValue();
        m_native_batch = null;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if (!(inst instanceof NativeInstance)) 
			throw new RuntimeException("Only NativeInstance is supported");
		
		if (!m_native_init)
		{
			m_copy_thread = new ThreadPoolExecutor(1, 1, 0, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<Runnable>(QUEUE_SIZE));
			m_train_thread = new ThreadPoolExecutor(1, 1, 0, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<Runnable>(QUEUE_SIZE));
			initNative(inst.numAttributes() -1, m_batch_size,m_loss,   inst.classAttribute().isNominal(), m_learning_rate, m_lambda);
			m_native_init = true;
			
		}
		
		if (m_native_batch == null)
		{
			m_native_batch = new NativeSparseInstanceBatch(inst.dataset(), m_batch_size);
		}
		boolean full = m_native_batch.addInstance((NativeInstance) inst);
		if (full)
		{
			final NativeInstanceBatch batch = m_native_batch;
			m_copy_thread.submit(() -> {
				batch.commit();
				m_train_thread.submit(() -> {trainNative(batch); });
			});
			
			m_native_batch = new NativeSparseInstanceBatch(inst.dataset(), m_batch_size);
			m_native_batch.addInstance((NativeInstance) inst);
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

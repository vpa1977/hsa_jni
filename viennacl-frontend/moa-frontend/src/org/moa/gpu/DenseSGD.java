package org.moa.gpu;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import org.moa.gpu.bridge.DenseOffHeapBuffer;
import org.moa.gpu.bridge.NativeClassifier;
import org.moa.gpu.bridge.NativeInstance;
import org.moa.gpu.bridge.NativeInstanceBatch;
import org.moa.gpu.bridge.NativeDenseInstanceBatch;
import org.moa.gpu.bridge.NativeDenseInstance;
import org.moa.gpu.util.DirectMemory;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.SparseInstance;
import moa.classifiers.AbstractClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.options.ClassOption;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;

/** 
 * SGD implementation backed by the native library.
 * @author bsp
 *
 */
public class DenseSGD extends AbstractClassifier implements NativeClassifier {
	
	private DenseOffHeapBuffer m_native_batch;
	private static final int QUEUE_SIZE = 10;
	private ThreadPoolExecutor m_copy_thread;
	private ThreadPoolExecutor m_train_thread;

	
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
    
    
    public IntOption learningBatchSize = new IntOption("learningBatchSize", 'b', "Learning batch size", 256, 1, Integer.MAX_VALUE);
    

    protected static final int HINGE = 0;

    protected static final int LOGLOSS = 1;

    protected static final int SQUAREDLOSS = 2;

    /** The current loss function to minimize */
    protected int m_loss = HINGE;

	protected double m_learning_rate = 0.0001;
	private int m_batch_size;
	private boolean m_native_init = false;
	private int m_row;

	
	private native double[] getVotesForDenseInstance(DenseOffHeapBuffer inst );
	
	
	/** 
	 * Train on the next batch
	 * @param w
	 */
	private native void trainNative(DenseOffHeapBuffer w);
	
	
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
			m_copy_thread.submit( () -> {}).get();
			m_train_thread.submit( () -> {}).get();
			
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ExecutionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		if (inst == null)
			return null;
		DenseOffHeapBuffer buffer = new DenseOffHeapBuffer(1,  inst.numAttributes()-1);
		buffer.begin();
		buffer.set(inst, 0);
		buffer.commit();
		double[] res =  getVotesForDenseInstance(buffer);
		buffer.release();
		return res;
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
		if (!m_native_init)
		{
			m_copy_thread = new ThreadPoolExecutor(1, 1, 0, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<Runnable>(QUEUE_SIZE));
			m_train_thread = new ThreadPoolExecutor(1, 1, 0, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<Runnable>(QUEUE_SIZE));
			initNative(inst.numAttributes() -1, m_batch_size,m_loss,   inst.classAttribute().isNominal(), m_learning_rate, m_lambda);
			m_native_init = true;
		}
		
		if (m_native_batch == null)
		{
			m_native_batch = new DenseOffHeapBuffer(m_batch_size, inst.dataset().numAttributes()-1);
			m_row = 0;
		}
		m_native_batch.set(inst, m_row);
		m_row ++;
		if (m_row == m_batch_size)
		{
			m_row = 0;
			final DenseOffHeapBuffer batch = m_native_batch;
		
			m_copy_thread.submit( () -> 
			{
				batch.commit();
				m_train_thread.submit( () -> {trainNative(batch); batch.release(); });
			}
			);
		
			m_native_batch =  new DenseOffHeapBuffer(m_batch_size, inst.dataset().numAttributes()-1);;
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

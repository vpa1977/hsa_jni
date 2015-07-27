package org.moa.gpu;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import org.moa.gpu.bridge.NativeClassifier;
import org.moa.gpu.bridge.NativeInstance;
import org.moa.gpu.bridge.NativeInstanceBatch;
import org.moa.gpu.bridge.NativeDenseInstanceBatch;
import org.moa.gpu.bridge.NativeDenseWindow;
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
public class NaiveKnn extends AbstractClassifier implements NativeClassifier {
	
	private NativeDenseWindow m_native_batch;
	private static final int QUEUE_SIZE = 10;
	private ThreadPoolExecutor m_copy_thread;
	private ThreadPoolExecutor m_train_thread;

    public IntOption slidingWindowSize = new IntOption("slidingWindowSize", 'b', "Sliding Window Size", 1024, 2, Integer.MAX_VALUE);
    
	private int m_batch_size;
	private boolean m_native_init = false;

	
	private native double[] getVotesForDenseInstance(NativeDenseInstance inst );
	
	
	/** 
	 * Train on the next batch
	 * @param w
	 */
	private native void trainNative(NativeDenseWindow w);
	
	
	/** 
	 * initialize native context
	 * the native context pointer will be stored in m_native_context
	 * which should provide sufficient storage space
	 * @param size
	 * @param loss
	 * @param nominal
	 */
	private native void initNative(int num_attr, int batch_size);
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
		if (inst instanceof NativeDenseInstance)
			return getVotesForDenseInstance((NativeDenseInstance)inst);
		throw new RuntimeException("Unsupported instance type");
	}

	

	@Override
	public void resetLearningImpl() {
        m_batch_size = slidingWindowSize.getValue();
        m_native_batch = null;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if (!(inst instanceof NativeDenseInstance)) 
			throw new RuntimeException("Only NativeDenseInstance is supported");
		
		if (!m_native_init)
		{
			m_copy_thread = new ThreadPoolExecutor(1, 1, 0, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<Runnable>(QUEUE_SIZE));
			m_train_thread = new ThreadPoolExecutor(1, 1, 0, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<Runnable>(QUEUE_SIZE));
			initNative(inst.numAttributes() -1, m_batch_size);
			m_native_init = true;
		}
		
		if (m_native_batch == null)
		{
			m_native_batch = new NativeDenseWindow(inst.dataset(), m_batch_size);
		}
		boolean full = m_native_batch.addInstance((NativeInstance) inst);
		if (full)
		{
			final NativeDenseWindow batch = m_native_batch;
			m_copy_thread.submit( () -> 
			{
				batch.commit();
				m_train_thread.submit( () -> {trainNative(batch);});
			}
			);
			m_native_batch =  new NativeDenseWindow(inst.dataset(), m_batch_size);
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
		out.append("Naive KNN implementation backed by viennacl ");
	}
	  
	private long m_native_context; 

}

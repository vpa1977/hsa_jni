package org.moa.gpu;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import org.moa.gpu.bridge.NativeClassifier;
import org.moa.gpu.bridge.NativeDenseInstance;
import org.moa.gpu.bridge.NativeDenseWindow;
import org.moa.gpu.bridge.NativeInstance;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;
import org.moa.gpu.bridge.DenseOffHeapBuffer;
import weka.classifiers.rules.ZeroR;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Naive implementation backed by the native library.
 *
 * @author bsp
 *
 */
public class NaiveKnn extends AbstractClassifier implements NativeClassifier {

    private int m_batch_size;
    private boolean m_native_init = false;
    private SlidingWindow m_sliding_window;
    private int m_num_classes;
    private int m_class_index;
    private int m_k;
    private ZeroR m_default_classifier = new ZeroR();

    public IntOption neighboursNumber = new IntOption("neighbourNumber", 'n', "Number of neighbours to use", 16, 1, Integer.MAX_VALUE);
    public IntOption slidingWindowSize = new IntOption("slidingWindowSize", 'b', "Sliding Window Size", 1024, 2, Integer.MAX_VALUE);
    public MultiChoiceOption distanceWeightingOption = new MultiChoiceOption("distanceWeighting", 'w', "Distance Weighting", 
            new String[]{"none", "similarity", "inverse"}, 
            new String[]{"No distance weighting", "Weight by 1-distance", "Weight by 1/distance"}, 0);

   
    

    private native double[] getVotesForDenseInstance(DenseOffHeapBuffer buffer, DenseOffHeapBuffer model);

     /**
     * initialize native context the native context pointer will be stored in
     * m_native_context which should provide sufficient storage space
     *
     * @param size
     * @param loss
     * @param nominal
     */
    private native void initNative(int num_attr, int batch_size, int class_index, int[] attributeTypes);

    /**
     * dispose native context and all natively allocated structures
     */
    private native void dispose();

    protected void finalize() throws Throwable {
        dispose();
    }
	
	

	@Override
    public boolean isRandomizable() {
        // TODO Auto-generated method stub
        return false;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (inst == null) {
            return null;
        }
        DenseOffHeapBuffer buffer = new DenseOffHeapBuffer(1, inst.numAttributes());
        buffer.begin();
        buffer.set(inst, 0);
        buffer.commit();
        double[] res = getVotesForDenseInstance(buffer, m_model);
        buffer.release();
        return res;
    }

    @Override
    public void resetLearningImpl() {
        m_batch_size = slidingWindowSize.getValue();
        shutdown();
    }
    
    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (!(inst instanceof NativeDenseInstance)) {
            throw new RuntimeException("Only NativeDenseInstance is supported");
        }

        if (!m_native_init) {
            initNative(inst.numAttributes(), m_batch_size, inst.classIndex(), attributeTypes(inst.dataset()));
            m_sliding_window = new SlidingWindow(m_batch_size, inst.dataset().numAttributes());
            m_native_init = true;
        }
        m_sliding_window.update(inst);
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
    
    private void shutdown()
    {
        if (m_native_init)
        {
            m_sliding_window.dispose();
            dispose();
            m_native_init = false;
        }
    }
    
    private int[] attributeTypes(Instances dataset)
    {
        int[] attributeTypes = new int[dataset.numAttributes()];
         for (int i = 0 ;i < attributeTypes.length; ++i)
         {
             if(dataset.attribute(i).isNumeric())
                 attributeTypes[i] = 0;
             else
             if(dataset.attribute(i).isNominal())
                 attributeTypes[i] = 1;
             else
                 attributeTypes[i] = 2;
         }
        return attributeTypes;
    }

    private long m_native_context;

}

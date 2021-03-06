package org.moa.gpu;

import org.moa.gpu.bridge.DenseOffHeapBuffer;
import org.moa.gpu.bridge.NativeClassifier;
import org.moa.gpu.bridge.NativeDenseInstance;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;
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
    private int m_k;
    private int m_distance_weighting;
    private ZeroR m_default_classifier = new ZeroR();

    public IntOption neighboursNumber = new IntOption("neighbourNumber", 'n', "Number of neighbours to use", 16, 1, Integer.MAX_VALUE);
    public IntOption slidingWindowSize = new IntOption("slidingWindowSize", 'b', "Sliding Window Size", 1024, 2, Integer.MAX_VALUE);
    public MultiChoiceOption distanceWeightingOption = new MultiChoiceOption("distanceWeighting", 'w', "Distance Weighting", 
            new String[]{"none", "similarity", "inverse"}, 
            new String[]{"No distance weighting", "Weight by 1-distance", "Weight by 1/distance"}, 0);

   
    

    private native double[] getVotesForDenseInstance(DenseOffHeapBuffer buffer);

     /**
     * initialize native context the native context pointer will be stored in
     * m_native_context which should provide sufficient storage space
     *
     */
    private native void initNative(int num_attr, int batch_size, int class_index, int[] attributeTypes, int num_classes, int k, int distance_weighting);
    
    
    private native void train(DenseOffHeapBuffer model);

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
        if (!m_sliding_window.isReady())
        {
			try {
				return m_default_classifier.distributionForInstance(inst);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			return null;
        }
        
        DenseOffHeapBuffer buffer = new DenseOffHeapBuffer(1, inst.numAttributes());
        buffer.begin();
        buffer.set(inst, 0);
        buffer.commit();
        double[] res = getVotesForDenseInstance(buffer);
        buffer.release();
        return res;
    }

    @Override
    public void resetLearningImpl() {
        m_batch_size = slidingWindowSize.getValue();
        m_k = neighboursNumber.getValue();
        m_distance_weighting = distanceWeightingOption.getChosenIndex();
        shutdown();
    }
    
    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (!(inst instanceof NativeDenseInstance)) {
            throw new RuntimeException("Only NativeDenseInstance is supported");
        }

        if (!m_native_init) {
            initNative(inst.numAttributes(), m_batch_size, inst.classIndex(), 
            		attributeTypes(inst.dataset()),  inst.dataset().numClasses(), 
            		m_k, m_distance_weighting);
            m_sliding_window = new SlidingWindow(m_batch_size, inst.dataset().numAttributes());
            try {
				m_default_classifier.buildClassifier(inst.dataset());
			} catch (Exception e) {	}
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

package hsa_jni.hsa_jni;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;

import org.stream_gpu.knn.kdtree.KDTreeWindow;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class KdTreeKnnGpuClassifier extends  AbstractClassifier {
	
    /** no weighting. */
    public static final int WEIGHT_NONE = 1;
    /** weight by 1/distance. */
    public static final int WEIGHT_INVERSE = 2;
    /** weight by 1-distance. */
    public static final int WEIGHT_SIMILARITY = 4;

	
	private int m_k;
	private int m_class_type;
	private int m_num_classes;
	private int m_distance_weighting;
	private int m_window_size;
	private int m_num_attributes_used;
	
	private WekaHSAContext m_context;
	private KDTreeWindow m_window;
	
	public KdTreeKnnGpuClassifier(WekaHSAContext context, int size, int k)
	{
		m_context = context;
		m_k = k;
		m_window_size = size;
		
	}
	
	private void init(Instances model)
	{
		m_class_type = model.classAttribute().type();
		m_num_classes = model.numClasses();
		m_distance_weighting = WEIGHT_NONE;
		m_num_attributes_used = 0;
        for (int i = 0; i < model.numAttributes(); i++) {
          if ((i != model.classIndex()) && 
    	  (model.attribute(i).isNominal() || model.attribute(i).isNumeric())) {
        	  m_num_attributes_used += 1;
          }
        }
		
	}
	
	
	public double[] getVotesForInstance(Instance arg0) {
		double [] distribution = new double [m_num_classes];
		for (int i = 0 ;i < distribution.length ; ++i)
			distribution[i] = 1/m_num_classes;
		return distribution;
	}

	
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void getModelDescription(StringBuilder arg0, int arg1) {
		// TODO Auto-generated method stub
		
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	  @Override
    public void resetLearningImpl() {
		  m_window = null;
    }

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if (m_window == null){
			m_window = new KDTreeWindow(m_context, m_window_size, inst.dataset());
		}
		m_window.add(inst);
	}
	
	 protected double [] makeDistribution(int[] indices, double[] distances)
		      throws Exception {

		      double total = 0, weight;
		      double [] distribution = new double [m_num_classes];
		      
		      // Set up a correction to the estimator
		      if (m_class_type == Attribute.NOMINAL) {
		        for(int i = 0; i < m_num_classes; i++) {
		        	distribution[i] = 1.0 / Math.max(1,m_window_size);
		        }
		        total = (double)m_num_classes / Math.max(1,m_window_size);
		      }

		      for(int i=0; i < m_k; i++) {
		        // Collect class counts
		        Instance current = null;//m_native_context.instances()[indices[i]];
		        switch (m_distance_weighting) {
		          case WEIGHT_INVERSE:
		            weight = 1.0 / (Math.sqrt(distances[i]/m_num_attributes_used) + 0.001); // to avoid div by zero
		            break;
		          case WEIGHT_SIMILARITY:
		            weight = 1.0 - Math.sqrt(distances[i]/m_num_attributes_used);
		            break;
		          default:                                 // WEIGHT_NONE:
		            weight = 1.0;
		            break;
		        }
		        weight *= current.weight();
		        try {
		          switch (m_class_type) {
		            case Attribute.NOMINAL:
		              distribution[(int)current.classValue()] += weight;
		              break;
		            case Attribute.NUMERIC:
		              distribution[0] += current.classValue() * weight;
		              break;
		          }
		        } catch (Exception ex) {
		          throw new Error("Data has no class attribute!");
		        }
		        total += weight;      
		      }

		      // Normalise distribution
		      if (total > 0) {
		        Utils.normalize(distribution, total);
		      }
		      return distribution;
		    }

}

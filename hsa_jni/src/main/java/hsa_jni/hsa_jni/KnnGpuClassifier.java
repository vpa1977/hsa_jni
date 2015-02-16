package hsa_jni.hsa_jni;

import weka.core.Attribute;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;

public class KnnGpuClassifier extends  AbstractClassifier {
	
    /** no weighting. */
    public static final int WEIGHT_NONE = 1;
    /** weight by 1/distance. */
    public static final int WEIGHT_INVERSE = 2;
    /** weight by 1-distance. */
    public static final int WEIGHT_SIMILARITY = 4;

	
	private WekaHSAContext.KnnNativeContext m_native_context;
	private int m_k;
	private int m_class_type;
	private int m_num_classes;
	private int m_distance_weighting;
	private int m_window_size;
	private int m_num_attributes_used;
	
	private double[] m_tmp_distances;
	private int[] m_tmp_indexes;
	private WekaHSAContext m_context;
	
	public KnnGpuClassifier(WekaHSAContext context, int size, int k)
	{
		m_context = context;
		m_k = k;
		m_window_size = size;
		m_tmp_distances = new double[size];
		m_tmp_indexes = new int[size];
	}
	
	private void init(Instances model)
	{
		m_native_context = m_context.create(model, m_window_size);
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
	
	void printK(Instance a)
	{
		System.out.println("Target:" +a);
		EuclideanDistance dist = new EuclideanDistance(m_native_context.instances()[0].dataset());
		for (int i = 0 ;i < m_native_context.instances().length ;++i)
			dist.update(m_native_context.instances()[i]);
		for(int i = 0;i < 32 ; ++i)
		{
			System.out.println("Src:" + m_native_context.instances()[i] + " distance " + Math.sqrt(findDistance(i)) + " euc "+ dist.distance(a,m_native_context.instances()[i] ));
		}
	}
	
	double findDistance(int idx)
	{
		for (int i = 0 ;i < m_tmp_indexes.length ; ++i)
			if (m_tmp_indexes[i]  == idx)
				return m_tmp_distances[i];
		return 0;
	}

	
	public double[] getVotesForInstance(Instance arg0) {
		m_native_context.computeKnn(arg0, m_tmp_distances, m_tmp_indexes);
	//	printK(arg0);
		//1.000000 4.000000 4.000000 4.000000 9.000000 
		try {
			return makeDistribution(m_tmp_indexes, m_tmp_distances);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
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
		  if (m_native_context != null)
			  m_native_context.reset();
    }

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if (m_native_context == null)
			init(inst.dataset());
		m_native_context.addInstance(inst);
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
		        Instance current = m_native_context.instances()[indices[i]];
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

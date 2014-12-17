package hsa_jni.hsa_jni;

import java.util.ArrayList;
import java.util.Arrays;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/** 
 * 
 * Provides JNI interface to the 
 * instance data 
 *
 */
public class WekaHSAContext {
	
	static {
		System.loadLibrary("weka_jni");
	}
	
	KnnNativeContext m_context;
	
	public WekaHSAContext() {
		init();		
	}
	
	
	public WekaHSAContext.KnnNativeContext create(Instances model, int size)
	{
		return new KnnNativeContext(model, size);
	}
	
	public WekaHSAContext.SGD createNativeSGD()
	{
		return new SGD();
	}
	
	public class SGD 
	{
		public native void UpdateWeights(InstanceBatch batch, double[] weights);
	}
	
	
	/** 
	 * Provides sliding window backed by native implementation 
	 * In-Memory layout 
	 *  Numeric Attribute 1 [ Instance1 .. InstanceN ... Instance WindowSize] ... Numeric Attribute N [1..N .. Instance Window Size] Nominal Attribute1 [] ... Nominal Attribute N   
	 * @author bsp
	 *
	 */
	public class KnnNativeContext
	{
		
		public KnnNativeContext(Instances model, int size)
		{
			m_model = model;
			m_instances = new Instance[size];
			
			ArrayList<Integer> nominals = new  ArrayList<Integer>();
			ArrayList<Integer> numerics = new ArrayList<Integer>();

			int class_index = m_model.classIndex();
			for (int i = 0 ; i < m_model.numAttributes() ; i ++ )
			{
				if (i == class_index)
					continue;
				
				Attribute attr = m_model.attribute(i);
				if (attr.isNominal())
					nominals.add(i);
				if (attr.isNumeric())
					numerics.add(i);
			}
			m_numerics = numerics.toArray(new Integer[numerics.size()]);
			m_nominals = nominals.toArray(new Integer[nominals.size()]);
			m_window = new double[ size * (m_numerics.length + m_nominals.length)];
			m_ranges = new double[ m_numerics.length * 2];
			
			for (int i = 0 ;i < m_numerics.length; ++i)
			{
				m_ranges[i*2] = Double.MAX_VALUE;
				m_ranges[i*2 +1]= Double.MIN_VALUE;
			}
			m_index = 0;
			m_current_size = 0;
			m_is_changed = false;
		}
		
		public void computeKnn(Instance test, double[] distances, int[] indices)
		{
			rescanAll();
			
			double [] testArr = new double[m_nominals.length + m_numerics.length];
			int offset = 0;
			double[] ranges = new double[ m_ranges.length];
			System.arraycopy(m_ranges, 0, ranges, 0, ranges.length);
			for (int i = 0; i < m_numerics.length; i ++ ) 
			{
				int idx = m_numerics[i];
				testArr[offset]  = test.value(idx);
				if (ranges[i*2] > testArr[offset])
					ranges[i*2] = testArr[offset];
				else if (ranges[i*2+1] < testArr[offset])
					ranges[i*2+1] = testArr[offset];
				++offset;
			}
			
			for (int i = 0; i < m_nominals.length; i ++ ) 
				testArr[offset++] = test.value(m_nominals[i]);
			
			knn (testArr, 
				m_window, // window
				 ranges, // value ranges
				 m_numerics.length, // numerics
	     	     m_nominals.length + m_numerics.length, // full instance size
	     	     m_instances.length, // full window size 
	     	     m_current_size,
			     distances, // distances array 
			     indices // indexes array
			     );
			
		}
		
		public void rescanAll()
		{
			if (m_is_changed)
				rescanRanges(m_window, m_ranges, m_numerics.length, m_numerics.length+ m_nominals.length,m_instances.length, m_current_size);
			m_is_changed = false;
		}
		

		
		public void addInstance(Instance inst)
		{
			m_is_changed = true;
			m_instances[m_index] = inst;
			int i;
			for (i = 0; i < m_numerics.length; i ++ ) 
			{
				int idx = m_numerics[i];
				int offset = i * m_instances.length+m_index;
				m_window[offset] = inst.value(idx);
				if (m_ranges[i*2] > m_window[offset])
					m_ranges[i*2] = m_window[offset];
				else if (m_ranges[i*2+1] < m_window[offset])
					m_ranges[i*2+1] = m_window[offset];
			}
			
			for (; i < m_nominals.length + m_numerics.length; i ++ ) 
			{
				int offset = i * m_instances.length+m_index;
				m_window[offset] = inst.value(m_nominals[i - m_numerics.length]);
			}
			
			++m_index;
			if (m_current_size < m_instances.length)
				++m_current_size;
			if (m_index == m_instances.length)
				m_index = 0;
		}
		

		private native void knn(double[] instance, double[] m_window, double[] m_ranges, 
								int numerics_size,
								int instance_size,
								int window_size,
								int current_size,
								double[] result_distance, 
								int[] result_index);
		public native void rescanRanges(double[] m_window, double[] m_ranges, int numerics_size,int instance_size, 
								int window_size, 
								int current_size);
		
		public Instances m_model;
		public Instance[] m_instances;
		public double[] m_window;
		public double[] m_ranges;
		private Integer[] m_numerics;
		private Integer[] m_nominals;
		private int m_index;
		private int m_current_size;
		private boolean m_is_changed;
		
		private double[] m_tmp_distances;
		private int[] m_tmp_indexes;
		
		
		public void rescanRangesSeq(double[] m_window, double[] m_ranges, int numerics_size,int instance_size, int window_size, int attribute, int is_max )
		{
			double result;
			if (is_max > 0)
				result = Double.MIN_VALUE;
			else
				result = Double.MAX_VALUE;
			int offset = attribute;
			for (int i = 0 ;i < window_size ; ++i)
			{
				double test = m_window[ i * instance_size + offset];
				if (is_max > 0)
				{
					if (test > result)
						result = test;
				}
				else
				{
					if (test < result)
						result = test;
				}
			}
			if (is_max>0)
				m_ranges[ offset * 2+1] = result;
			else
				m_ranges[ offset * 2] = result;
			
				
		}
		
		public void rescanAllSeq()
		{
			for (int i = 0 ;i < m_numerics.length; ++i)
			{
				rescanRangesSeq(m_window, m_ranges, m_numerics.length, m_numerics.length+ m_nominals.length,m_current_size, i, 1);
				rescanRangesSeq(m_window, m_ranges, m_numerics.length, m_numerics.length+ m_nominals.length,m_current_size, i, 0);
			}
		}

		public void reset() {
			Arrays.fill(m_instances, null);
			Arrays.fill(m_window, 0);
			Arrays.fill(m_ranges, 0);
			m_current_size = 0;
		}

		public Instance[] instances() {
			return m_instances;
		}

	
	}
	

	/** 
	 * load gpu kernels
	 */
	public native void init();
	
	
	
	
}

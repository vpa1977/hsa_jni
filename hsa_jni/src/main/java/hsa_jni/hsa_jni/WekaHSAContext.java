package hsa_jni.hsa_jni;

import java.util.ArrayList;

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
			m_tmp_distances = new double [size];
			m_tmp_indexes = new int[size];
			
			for (int i = 0 ;i < m_numerics.length; ++i)
			{
				m_ranges[i*2] = Double.MAX_VALUE;
				m_ranges[i*2 +1]= Double.MIN_VALUE;
			}
			m_index = 0;
			m_current_size = 0;
		}
		
		public void computeKnn(Instance test)
		{
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
			
			knn (testArr,m_window, ranges,m_numerics.length,
					m_nominals.length + m_numerics.length, 
					m_instances.length, 
					m_tmp_distances, m_tmp_indexes);
			
		}
		
		public void addInstance(Instance inst)
		{
			boolean need_rescan = false;
			
			m_instances[m_index] = inst;
			int offset = m_index * (m_nominals.length + m_numerics.length);
			for (int i = 0; i < m_numerics.length; i ++ ) 
			{
				int idx = m_numerics[i];
				double val  = m_window[offset];
				if (m_ranges[i*2] == val)
					rescanRanges(m_window, m_ranges,m_numerics.length, m_numerics.length + m_nominals.length, m_current_size, i,0);
				else if (m_ranges[i*2+1] == val) 
					rescanRanges(m_window, m_ranges,m_numerics.length, m_numerics.length + m_nominals.length, m_current_size, i,1);
				m_window[offset] = inst.value(idx);
				if (m_ranges[i*2] > m_window[offset])
					m_ranges[i*2] = m_window[offset];
				else if (m_ranges[i*2+1] < m_window[offset])
					m_ranges[i*2+1] = m_window[offset];
				++offset;
			}
			
			for (int i = 0; i < m_nominals.length; i ++ ) 
			{
				m_window[offset++] = inst.value(m_nominals[i]);
			}
			
			++m_index;
			if (m_current_size < m_instances.length)
				++m_current_size;
			if (m_index == m_instances.length)
				m_index = 0;
		}
		
		private native void knn(double[] instance, double[] m_window, double[] m_ranges, int numerics_size,int instance_size, int window_size, double[] result_distance, int[] result_index);
		private native void rescanRanges(double[] m_window, double[] m_ranges, int numerics_size,int instance_size, int window_size, int attribute, int is_max );
		
		public Instances m_model;
		public Instance[] m_instances;
		public double[] m_window;
		public double[] m_ranges;
		private Integer[] m_numerics;
		private Integer[] m_nominals;
		private int m_index;
		private int m_current_size;
		
		private double[] m_tmp_distances;
		private int[] m_tmp_indexes;
	}
	

	/** 
	 * load gpu kernels
	 */
	public native void init();
	
	
	
	
}

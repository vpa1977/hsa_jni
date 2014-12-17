package hsa_jni.hsa_jni;

import weka.core.Instance;
import weka.core.Instances;

public class InstanceBatch {
	
	private int[][] m_indices;
	private double[][] m_values;
	private double[] m_class_value;
	private double m_multiplier;
	private int m_index;
	

	private Instances m_dataset;

	public InstanceBatch(Instances dataset, int size)
	{
		m_indices = new int[size][];
		m_values = new double[size][];
		m_class_value = new double[size];
		m_index = 0;
		m_dataset = dataset;
	}
	
	public void setDataset(Instances dataset)
	{
		m_dataset = dataset;
	}
	
	public boolean isNominal() 
	{
		return m_dataset.classAttribute().isNominal();
	}
	
	public int classIndex() 
	{
		return m_dataset.classIndex();
	}
	
	public boolean add(Instance  i )
	{
		SparseInstanceAccess sparseAccess;
		if (i instanceof SparseInstanceAccess)
			sparseAccess = (SparseInstanceAccess)i;
		else
			sparseAccess = new SparseInstanceAccess( i );
		
		m_indices[ m_index ] = sparseAccess.getIndexes();
		m_values[ m_index ] = sparseAccess.getValues();
		m_class_value[m_index] = sparseAccess.classValue();
		++m_index;
		if (m_index >= m_indices.length)
			return false;
		return true;
	}
	
	public int size()
	{
		return m_index;
	}
	
	public double[][] values() 
	{
		return m_values;
	}
	
	public int[][] indices() 
	{
		return m_indices;
	}
	
	public double[] classValues()
	{
		return m_class_value;
	}
	
	public void commit()
	{
		m_index = 0;
	}

	/* 
		Those do not really belong here, but 
	*/
	public void setMultiplier(double multiplier) {
		m_multiplier = multiplier;
	}
	
	public double getMultiplier() 
	{
		return m_multiplier;
	}
	
	public double m_t;
	public double m_bias;
	public double m_learningRate;
}

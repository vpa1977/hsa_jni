package weka.core.neighboursearch;

import java.io.Serializable;

import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NormalizableDistance;
import weka.filters.unsupervised.attribute.RandomProjection;

public class RandomProjectionFold implements Serializable, IProjection {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private RandomProjection[] m_projections;
	private Instances[] m_data;
	private boolean m_copy;
	private NormalizableDistance[] m_distances;
	public RandomProjectionFold(Instances dataset, int num_curves, int num_dimensions) throws Exception
	{
		m_projections = new RandomProjection[num_curves];
		m_data = new Instances[num_curves];
		m_distances = new NormalizableDistance[num_curves];
		int projection_attributes = Math.min(num_dimensions, dataset.numAttributes()*2/3);
		for (int i = 0 ; i < m_projections.length ; ++i)
		{
			m_projections[i] = new RandomProjection() {
				public void setReplaceMissingValues(boolean val)
				{
					super.setReplaceMissingValues(true);
					m_replaceMissing = new weka.filters.unsupervised.attribute.ReplaceMissingValues();
				}
			};
			m_projections[i].setReplaceMissingValues(true);
			m_projections[i].setNumberOfAttributes(projection_attributes);
			m_projections[i].setInputFormat(dataset);

		}
	}
		
	
	/* (non-Javadoc)
	 * @see weka.core.neighboursearch.IProjection#project(weka.core.Instances)
	 */
	@Override
	public Instances[] project(Instances dataset) throws Exception
	{
		for (int curve = 0; curve < m_projections.length ; ++curve)
		{
			m_data[curve] = null;
			m_distances[curve] = null;
		}
		
		for (int i = 0 ; i < dataset.size() ; ++i) 
		{
			Instance inst = dataset.instance(i);
			for (int curve = 0; curve < m_projections.length ; ++curve)
			{
				Instance out;
				m_projections[curve].input(inst);
				m_projections[curve].batchFinished();
				if (m_data[curve] == null)
					m_data[curve] = m_projections[curve].getOutputFormat();
				m_data[curve].add(m_projections[curve].output());
			}
		}
		for (int i = 0 ;i < m_projections.length; ++i)
		{
			m_distances[i] = new EuclideanDistance(m_data[i]);
		}
		return m_data;
	}
	
	/* (non-Javadoc)
	 * @see weka.core.neighboursearch.IProjection#getProjection()
	 */
	@Override
	public Instances[] getProjection()
	{
		return m_data;
	}
	
	public NormalizableDistance getDistance(int i)
	{
		return m_distances[i];
	}
	

	public double[][] getRanges(int i)
	{
		try {
			return m_distances[i].getRanges();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
		
	/* (non-Javadoc)
	 * @see weka.core.neighboursearch.IProjection#project(int, weka.core.Instance)
	 */
	@Override
	public Instance project(int curve, Instance src) throws Exception
	{
		m_projections[curve].input(src);
		m_projections[curve].batchFinished();
		if (m_data[curve] == null)
			m_data[curve] =m_projections[curve].getOutputFormat(); 
		return m_projections[curve].output();
	}
}

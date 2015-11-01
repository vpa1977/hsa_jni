package weka.core.neighboursearch;

import java.io.Serializable;
import java.util.ArrayList;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NormalizableDistance;
import weka.filters.unsupervised.attribute.RandomProjection;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class DiagonalProjectionFold implements Serializable, IProjection {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private DiagonalProjection[] m_projections;
	private Instances m_source_data;
	private NormalizableDistance m_source_ranges;
	private Instances[] m_data;
	private boolean m_copy;
	private int m_projection_attribute_count;
	private NormalizableDistance[] m_distances;
	private ReplaceMissingValues m_replace_missing_filter;
	public DiagonalProjectionFold(Instances dataset, int num_curves, int num_dimensions) throws Exception
	{
		m_projections = new DiagonalProjection[num_curves];
		m_data = new Instances[num_curves];
		m_distances = new NormalizableDistance[num_curves];
		m_projection_attribute_count = Math.min(num_dimensions, dataset.numAttributes()*2/3);
		for (int i = 0 ; i < m_projections.length ; ++i)
		{
			m_projections[i] = new DiagonalProjection(dataset.classIndex(), dataset.numAttributes(), m_projection_attribute_count);
		}
		m_replace_missing_filter = new ReplaceMissingValues();
		m_replace_missing_filter.setInputFormat(dataset);

	}
		
	
	/* (non-Javadoc)
	 * @see weka.core.neighboursearch.IProjection#project(weka.core.Instances)
	 */
	@Override
	public Instances[] project(Instances dataset) throws Exception
	{
		ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
		for (int i = 0 ;i < m_projection_attribute_count; ++i)
			attInfo.add(new Attribute(""+i));
		attInfo.add(new Attribute("class"));
		for (int curve = 0; curve < m_projections.length ; ++curve)
		{
			m_data[curve] = new Instances("proj"+ curve, attInfo, 0);
			m_data[curve].setClassIndex(attInfo.size()-1);
			m_distances[curve] = null;
		}
		
		m_source_data = dataset;
		m_source_ranges = new EuclideanDistance(m_source_data);
		
		for (int i = 0 ; i < dataset.size() ; ++i) 
		{
			Instance inst = dataset.instance(i);
			m_replace_missing_filter.input(inst);
			m_replace_missing_filter.batchFinished();
			Instance out  = m_replace_missing_filter.output();
			
			for (int curve = 0; curve < m_projections.length ; ++curve)
			{
				double[] values = new double[m_projection_attribute_count +1];
				values[values.length  - 1] = out.classValue();
				for (int att = 0; att < m_projection_attribute_count ; ++att)
					values[att] =m_projections[curve].project(att, m_source_ranges.getRanges(), out);
				m_data[curve].add( new DenseInstance(1.0, values));
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
		m_replace_missing_filter.input(src);
		m_replace_missing_filter.batchFinished();
		Instance out  = m_replace_missing_filter.output();
		
		double[] values = new double[m_projection_attribute_count +1];
		values[values.length  - 1] = out.classValue();
		for (int att = 0; att < m_projection_attribute_count ; ++att)
			values[att] =m_projections[curve].project(att, m_source_ranges.getRanges(), out);
		Instance inst =new DenseInstance(1.0, values);
		inst.setDataset(m_data[curve]);
		return  inst;
	}


	@Override
	public Instance[] addToProjections(Instance src) throws Exception {
		throw new RuntimeException("Not implemented");
	}
}

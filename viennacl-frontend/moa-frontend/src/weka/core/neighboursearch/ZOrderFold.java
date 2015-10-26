package weka.core.neighboursearch;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

import weka.core.Instance;
import weka.core.NormalizableDistance;

public class ZOrderFold  implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private int m_num_attributes;
	private int m_num_dimensions;
	// curve|dimension
	private ArrayList<Integer>[][] m_fold;
	
	public ZOrderFold(int class_index,int num_attributes, int num_dimensions, int num_curves)
	{
		Random rnd = new Random();
		m_num_attributes= num_attributes-1;
		m_num_dimensions = num_dimensions;
		m_fold = new ArrayList[num_curves][m_num_dimensions];
		for (int curve = 0; curve < num_curves; ++curve)
		{
			int slice = m_num_attributes / m_num_dimensions;
			ArrayList<Integer> indices = new ArrayList<Integer>();
			for (int i = 0 ;i < m_num_attributes+1; ++i)
				if (i != class_index) indices.add(i);
			
			for (int i = 0; i < m_num_dimensions; ++i)
			{
				m_fold[curve][i]= new ArrayList<Integer>();
				int count = Math.min(slice, indices.size());
				for (int j = 0; j < count; ++j)
				{
					
					m_fold[curve][i].add(indices.remove(rnd.nextInt(indices.size())));
				}
				if (indices.size() < slice)
					m_fold[curve][i].addAll(indices);
			}
		}
	}
	
	public double fold(int curve, double[][] ranges, int dim, Instance attribute)
	{
		ArrayList<Integer> list = m_fold[curve][dim];
		double acc =0;
		for (Integer i : list)
		{
			acc += (attribute.value(i) - ranges[i][NormalizableDistance.R_MIN])/ranges[i][NormalizableDistance.R_WIDTH];
		}
		return acc;
	}

	public ArrayList<Integer> mapping(int curve, int dim) {
		return m_fold[curve][dim];
	}
}

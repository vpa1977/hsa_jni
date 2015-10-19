package weka.core.neighboursearch;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

import weka.core.Instance;

public class ZOrderFold  implements Serializable{
	private int m_num_attributes;
	private int m_num_dimensions;
	private ArrayList<Integer>[] m_fold;
	
	public ZOrderFold(int class_index,int num_attributes, int num_dimensions)
	{
		Random rnd = new Random();
		m_num_attributes= num_attributes-1;
		m_num_dimensions = num_dimensions;
		m_fold = new ArrayList[m_num_dimensions];
		int slice = m_num_attributes / m_num_dimensions;
		ArrayList<Integer> indices = new ArrayList<Integer>();
		for (int i = 0 ;i < m_num_attributes+1; ++i)
			if (i != class_index) indices.add(i);
		
		for (int i = 0; i < m_num_dimensions; ++i)
		{
			m_fold[i]= new ArrayList<Integer>();
			int count = Math.min(slice, indices.size());
			for (int j = 0; j < count; ++j)
			{
				
				m_fold[i].add(indices.remove(rnd.nextInt(indices.size())));
			}
			if (indices.size() < slice)
				m_fold[i].addAll(indices);
		}
	}
	
	public double fold(int dim, Instance attribute)
	{
		ArrayList<Integer> list = m_fold[dim];
		double acc =0;
		for (Integer i : list)
		{
			acc += attribute.value(i);
		}
		return acc;
	}

	public ArrayList<Integer> mapping(int i) {
		return m_fold[i];
	}
}

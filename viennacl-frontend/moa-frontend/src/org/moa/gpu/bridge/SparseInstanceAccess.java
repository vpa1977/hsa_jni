package org.moa.gpu.bridge;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

public class SparseInstanceAccess extends SparseInstance {
	public SparseInstanceAccess(Instance i) {
		super(i);
		setDataset(i.dataset());
	}
	
	/** 
	 * Create a sparse instance from the dense one usef pre-defined attribute mapping 
	 * @param size
	 * @param instance
	 * @param att_mapping
	 */
	 public SparseInstanceAccess(Instances parent, Instance instance, HashMap<Integer, Integer> att_mapping) {
	    m_Weight = instance.weight();
	    m_Dataset = parent;
	    m_NumAttributes = parent.numAttributes();
	    m_AttValues = new double[instance.numAttributes()];
	    m_Indices = new int[instance.numAttributes()];
	    
	    class Pair {
	    	Pair(double val, int index) 
	    	{ m_val = val; m_index = index; }
	    	double m_val;
	    	int m_index;
	    }
	    ArrayList<Pair> list = new ArrayList<Pair>();
	    for (int i = 0; i < instance.numAttributes(); i++) 
	    {
	    	list.add(new Pair(instance.value(i), att_mapping.get(i)));
	    }
	    Collections.sort(list, new Comparator<Pair>(){
			@Override
			public int compare(Pair p1, Pair p2) {
				return p1.m_index - p2.m_index;
			}
	    	
	    });
	    
	    for (int i = 0; i < instance.numAttributes(); i++) 
	    {
		    m_AttValues[i] = list.get(i).m_val;
		    m_Indices[i] = list.get(i).m_index;
	    }
	}

	public int[] getIndices() {
		return m_Indices;
	}

	public double[] getValues() {
		return m_AttValues;
	}

}
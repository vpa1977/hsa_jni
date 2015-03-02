package org.moa.streams;

import java.util.LinkedList;
import java.util.List;
import java.util.function.Predicate;

import weka.core.Instance;
import weka.core.Instances;

/** 
 * Implementation of 
 * @MISC{Douglas96runningmax/min,
    	author = {Scott C. Douglas},
    	title = {Running Max/Min Calculation Using a Pruned Ordered List},
    	year = {1996}
    	url = {http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777}
	}
	
 * Given sliding window [1 2 3 2 2] 3 4 6
 * We can ignore everything left of true max [3] and keep only [2 2] as the max-candidate list
 * When 1 leaves window we have 1 [2 3 2 2 3] 4 6 and the max-candidate becomes [3] - since [2 2] will never 
 * become max
 * Same applies for the minimum calculation. 
 *  
 *  Similiar algorithm was also proposed by Zhao et. al. (who for some reason does not cite Douglas)
 *  though Zhao also explores time-based sliding windows. 
 *  @incollection{
year={2007},
isbn={978-3-540-71702-7},
booktitle={Advances in Databases: Concepts, Systems and Applications},
volume={4443},
series={Lecture Notes in Computer Science},
editor={Kotagiri, Ramamohanarao and Krishna, P.Radha and Mohania, Mukesh and Nantajeewarawat, Ekawit},
doi={10.1007/978-3-540-71703-4_55},
title={Evaluating MAX and MIN over Sliding Windows with Various Size Using the Exemplary Sketch},
url={http://dx.doi.org/10.1007/978-3-540-71703-4_55},
publisher={Springer Berlin Heidelberg},
author={Zhao, Jiakui and Yang, Dongqing and Cui, Bin and Chen, Lijun and Gao, Jun},
pages={652-663},
language={English}
}

 *  
 */
public class MaxList implements MinMaxWindow {
	private Instances m_dataset;
	private double[] m_max;
	private double[] m_min;
	private List<Double>[] m_max_list;
	private List<Double>[] m_min_list;
	private int m_min_len;
	
	public MaxList(Instances dataset)
	{
		m_min_len = 0;
		
		m_max_list = new LinkedList[dataset.numAttributes()];
		m_min_list = new LinkedList[dataset.numAttributes()];
		m_max = new double[dataset.numAttributes()];
		m_min = new double[dataset.numAttributes()];
		for (int i = 0 ; i < dataset.numAttributes(); ++i)
		{
			m_max[i] = -Double.POSITIVE_INFINITY;
			m_min[i] = Double.POSITIVE_INFINITY;
			m_max_list[i]= new LinkedList<Double>();
			m_min_list[i]= new LinkedList<Double>();
		}
		m_dataset = dataset;
	}
	
	
	public void update(Instance inst, Instance to_remove)
	{
		for (int i = 0 ;i < m_dataset.numAttributes() ; ++i)
		{
			if (!inst.isMissing(i)) // handle new value
			{	
				final double value = inst.value(i);
				// update max
				if (value > m_max[i]) // new max entered sliding window. prune max candidate list
				{
					m_max[i] = value;
					m_max_list[i].clear();
				}
				else 
				{
					m_min_list[i].removeIf(new Predicate<Double>(){	
						public boolean test(Double t) {
									return t < value;
								}
					});
/*					final LinkedList<Double> d = new LinkedList<Double>();
					m_max_list[i].forEach((x)-> { if (x > value) d.add(x); });
					m_max_list[i] = d;*/
					m_max_list[i].add(value);
					if (m_min_len < m_min_list[i].size())
						m_min_len = m_min_list[i].size();

				}
				
				// update min
				if (value < m_min[i]) // new min entered sliding window. prune min candidate list
				{
					m_min[i] = value;
					m_min_list[i].clear();
				} 
				else 
				{   // a new candidate entered if it is less than true min, 
					// purge conditionally
/*					final LinkedList<Double> d = new LinkedList<Double>();
					m_min_list[i].forEach( (x) ->  { if (x < value) d.add(x);} );
					m_min_list[i] = d;*/
					m_min_list[i].removeIf(new Predicate<Double>(){	
						public boolean test(Double t) {
									return t > value;
								}
						
					});
					m_min_list[i].add(value);
					if (m_min_len < m_min_list[i].size())
						m_min_len = m_min_list[i].size();
				}
			}
			
			if (to_remove != null && !to_remove.isMissing(i)) // 
			{
				final double value = to_remove.value(i);
				if (value == m_max[i])
					m_max[i] = m_max_list[i].remove(0);
				if (value == m_min[i])
					m_min[i] = m_min_list[i].remove(0);
			}
		}
	}


	public double max(int i) {
		return m_max[i];
	}

	public double min(int i) {
		return m_min[i];
	}


	@Override
	public int length() {
		return m_max.length;
	}


	@Override
	public double width(int i) {
		return Math.abs( max(i) - min(i));
	}

}

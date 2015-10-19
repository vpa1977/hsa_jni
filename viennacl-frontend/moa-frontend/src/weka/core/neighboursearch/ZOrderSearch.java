/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.core.neighboursearch;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;

/**
 * Z-Order curve search implementation, implemented in line with to http://cs.uef.fi/sipu/pub/2015_MSc_Sami_Sieranoja.pdf
 * 
 */
public class ZOrderSearch extends NearestNeighbourSearch {

	private ZOrder m_order;
	private ArrayList<ZOrderInstance>[] m_instance_list;
	private double[] m_Distances;
	private boolean m_SkipIdentical;
	/** 
	 * Number of z-order curves to generate
	 */
	private int m_NumSearchCurves = 3;
	/** 
	 * Number of dimensions to reduce
	 */
	private int m_NumDimensions = 3;
	/** 
	 * Perform dimensionality reduction
	 */
	private boolean m_UseReduction = true;

	/**
	 * Constructor.
	 */
	public ZOrderSearch() {
		super();
	}

	/**
	 * Constructor.
	 * 
	 * @param insts
	 *            The set of instances that constitute the neighbourhood.
	 */
	public ZOrderSearch(Instances insts) {
		super(insts);
		m_order = createOrder(insts);
		m_DistanceFunction.setInstances(insts);
		m_instance_list = new ArrayList[m_NumSearchCurves];
		for (int i = 0; i < m_NumSearchCurves; ++i)
			m_instance_list[i] = m_order.createZOrder(insts, i);
	}

	private ZOrder createOrder(Instances insts) {
		return m_UseReduction
				? new ZOrder(insts.numAttributes(), m_NumSearchCurves, insts.classIndex(), m_NumDimensions)
				: new ZOrder(insts.numAttributes(), m_NumSearchCurves);
	}

	@Override
	public Instance nearestNeighbour(Instance target) throws Exception {

		double min = Double.MAX_VALUE;
		Instance result = null;
		for (int i = 0; i < m_NumSearchCurves; ++i) {
			ZOrderInstance instance = new ZOrderInstance(m_order.interleave(target, i), target);
			int position = Collections.binarySearch(m_instance_list[i], instance);
			int other = position < m_instance_list[i].size() ? position + 1 : position - 1;
			double d1 = m_DistanceFunction.distance(target, m_instance_list[i].get(position).m_instance);
			if (d1 < min) {
				min = d1;
				result = m_instance_list[i].get(position).m_instance;
			}
			double d2 = m_DistanceFunction.distance(target, m_instance_list[i].get(other).m_instance);
			if (d2 < min) {
				min = d2;
				result = m_instance_list[i].get(other).m_instance;
			}
		}
		return result;
	}

	@Override
	public Instances kNearestNeighbours(Instance target, int kNN) throws Exception {
		ArrayList<Instance> candidates = new ArrayList<Instance>();
		for (int i = 0; i < m_NumSearchCurves; ++i) {
			ZOrderInstance instance = new ZOrderInstance(m_order.interleave(target, i), target);
			int position = Collections.binarySearch(m_instance_list[i], instance);
			int min = Math.max(0, position - kNN);
			int max = Math.min(position + kNN, m_instance_list[i].size() - 1);
			for (int index = min; index <= max; ++index) {
				candidates.add(m_instance_list[i].get(index).m_instance);
			}
		}

		MyHeap heap = new MyHeap(kNN);
		double distance;
		int firstkNN = 0;
		for (int i = 0; i < candidates.size(); ++i) {
			Instance inst = candidates.get(i);
			if (target == inst) // for hold-one-out cross-validation
				continue;
			if (m_Stats != null)
				m_Stats.incrPointCount();
			if (firstkNN < kNN) {
				distance = m_DistanceFunction.distance(target, inst, Double.POSITIVE_INFINITY, m_Stats);
				if (distance == 0.0 && m_SkipIdentical)
					if (i < candidates.size())
						continue;
					else
						heap.put(i, distance);
				heap.put(i, distance);
				firstkNN++;
			} else {
				MyHeapElement temp = heap.peek();
				distance = m_DistanceFunction.distance(target, candidates.get(i), (float) temp.distance, m_Stats);
				if (distance == 0.0F && m_SkipIdentical)
					continue;
				if (distance < temp.distance) {
					heap.putBySubstitute(i, distance);
				} else if (distance == temp.distance) {
					heap.putKthNearest(i, distance);
				}

			}
		}

		Instances neighbours = new Instances(m_Instances, (heap.size() + heap.noOfKthNearest()));
		m_Distances = new double[heap.size() + heap.noOfKthNearest()];
		int[] indices = new int[heap.size() + heap.noOfKthNearest()];
		int i = 1;
		MyHeapElement h;
		while (heap.noOfKthNearest() > 0) {
			h = heap.getKthNearest();
			indices[indices.length - i] = h.index;
			m_Distances[indices.length - i] = (float) h.distance;
			i++;
		}
		while (heap.size() > 0) {
			h = heap.get();
			indices[indices.length - i] = h.index;
			m_Distances[indices.length - i] = (float) h.distance;
			i++;
		}

		m_DistanceFunction.postProcessDistances(m_Distances);

		for (int k = 0; k < indices.length; k++) {
			neighbours.add(candidates.get(indices[k]));
		}

		if (m_Stats != null)
			m_Stats.searchFinish();

		return neighbours;
	}

	@Override
	public double[] getDistances() throws Exception {
		return m_Distances;
	}

	@Override
	public void update(Instance ins) throws Exception {
		for (int i = 0; i < m_NumSearchCurves; ++i) {
			BigInteger newInt = m_order.interleave(ins, i);
			ZOrderInstance zo = new ZOrderInstance(newInt, ins);
			m_instance_list[i].add(zo);
		}
		m_DistanceFunction.update(ins);
	}

	/**
	 * Sets the instances.
	 * 
	 * @param insts
	 *            the instances to use
	 * @throws Exception
	 *             if setting fails
	 */
	public void setInstances(Instances insts) throws Exception {
		super.setInstances(insts);
		m_order = createOrder(insts);
		m_DistanceFunction.setInstances(insts);
		m_instance_list = new ArrayList[m_NumSearchCurves];
		for (int i = 0; i < m_NumSearchCurves; ++i)
			m_instance_list[i] = m_order.createZOrder(insts, i);
	}

	@Override
	public String getRevision() {
		return "1.0";
	}
	
	  /**
	   * Returns a string describing this nearest neighbour search algorithm.
	   * 
	   * @return 		a description of the algorithm for displaying in the 
	   * 			explorer/experimenter gui
	   */
	  public String globalInfo() {
	    return 
	        "Approximate nearest neighbours search using z-order function";
	  }

	  /**
	   * Returns an enumeration describing the available options.
	   *
	   * @return 		an enumeration of all the available options.
	   */
	  public Enumeration listOptions() {
	    Vector<Option> newVector = new Vector<Option>();
	    newVector.add(new Option(
	    		"\tSkip identical instances (distances equal to zero).\n",
	    		"S", 1,"-S"));

	    newVector.add(new Option(
	    		"\tNumber of search curves.\n",
	    		"C", 3,"-C"));

	    newVector.add(new Option(
	    		"\tNumber of z-order curve dimensions.\n",
	    		"D", 3,"-D"));

	    newVector.add(new Option(
	    		"\tPerform dimensionality reduction\n",
	    		"R", 3,"-R"));

	    
	    return newVector.elements();
	  }
	  
	  /**
	   * Parses a given list of options. Valid options are:
	   *
	   <!-- options-start -->
	   <!-- options-end -->
	   *
	   * @param options 	the list of options as an array of strings
	   * @throws Exception 	if an option is not supported
	   */
	  public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		
	    m_UseReduction = Utils.getFlag('R', options);
	    m_SkipIdentical = Utils.getFlag('S', options);
	    if (m_UseReduction)
	    {
	    	m_NumDimensions = Integer.parseInt(Utils.getOption('D', options));
	    }
	    m_NumSearchCurves =  Integer.parseInt(Utils.getOption('C', options));
	  }

	  /**
	   * Gets the current settings.
	   *
	   * @return 		an array of strings suitable for passing to setOptions()
	   */
	  public String [] getOptions() {
	    Vector<String>	result;
	    String[]		options;
	    int			i;
	    
	    result = new Vector<String>();
	    
	    options = super.getOptions();
	    for (i = 0; i < options.length; i++)
	      result.add(options[i]);
	    
	    if (m_SkipIdentical)
	      result.add("-S");
	    if (m_UseReduction)
	      result.add("-R");
	    if (m_UseReduction)
	    {
	    	result.add("-D");
	    	result.add(m_NumDimensions+"");
	    }
	    result.add("-C");
	    result.add(m_NumSearchCurves+"");

	    return result.toArray(new String[result.size()]);

	  }


}

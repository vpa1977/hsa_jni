package org.moa.streams;

import weka.core.Instance;

public class Prediction {
	private Instance m_instance;
	private double[] m_votes;
	private double m_true_class;
	
	public Prediction(Instance inst)
	{
		m_instance = inst;
	}
	
	public Prediction(Instance testInst, double trueClass) {
		m_true_class = trueClass;
	}
	
	public double getTrueClass()
	{
		return m_true_class;
	}

	public Instance instance()
	{
		return m_instance;
	}
	
	public void setVotes(double[] votes)
	{
		m_votes = votes;
	}
	
	public double[] getVotes() 
	{
		return m_votes;
	}

}

package org.stream_gpu.knn.kdtree;

import java.util.Stack;

import org.moa.streams.Prediction;

public class WorkItem 
{
	enum PATH {
		LEFT, 
		RIGHT, 
		BOTH, 
		NONE
	}
	
	
	public WorkItem(int k, Prediction p)
	{
		m_prediction = p;
		m_heap = new Heap(k);
		m_path = new Stack<PATH>();
	}
	
	
	public Prediction m_prediction;
	public Heap m_heap;
	public Stack<PATH> m_path;
	public int m_k;
	
	public PATH getCurrentPath(int level) {
		// TODO Auto-generated method stub
		return null;
	}

	public void setCurrentPath(int m_level, PATH path, double distanceToParents) {
		// TODO Auto-generated method stub
		
	}

	public double getDistanceToParents(int m_level) {
		// TODO Auto-generated method stub
		return 0;
	}

	public void popPath() {
		
		
	}

	public void addPath(int i, PATH none, double distanceToSplitPlane) {
		// TODO Auto-generated method stub
		
	}
}
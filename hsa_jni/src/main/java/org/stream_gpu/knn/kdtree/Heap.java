package org.stream_gpu.knn.kdtree;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.PriorityQueue;

import weka.core.Instance;


public class Heap {
	
	public class Entry {
		public double m_distance;
		public Instance m_instance;
	}
	
	private PriorityQueue<Entry> m_queue;
	private int m_k;
	
	public Heap(int k)
	{
		m_queue = new PriorityQueue<Entry>(k, new Comparator<Entry>() {
			@Override
			public int compare(Entry left, Entry right) {
				return left.m_distance - right.m_distance > 0 ? -1 : 1;
			}
		});
		m_k = k;
	}
	
	public void add(Entry inst)
	{
		m_queue.add(inst);
		while (m_queue.size() > m_k)
			m_queue.remove();
	}

	public int size() {
		return m_queue.size();
	}

	public Entry peek() {
		return m_queue.peek();
	}

	public ArrayList<Entry> toArray() {
		ArrayList<Entry> list = new ArrayList<Entry>();
		list.addAll(m_queue);
		return list;
	}

}

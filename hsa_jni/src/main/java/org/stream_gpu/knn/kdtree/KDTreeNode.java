package org.stream_gpu.knn.kdtree;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.concurrent.Future;
import java.util.function.Predicate;

import org.moa.streams.Prediction;

import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

public class KDTreeNode {

	public static int SPLIT_VALUE = 2048;
	public static int COLLAPSE_VALUE = 1024;

	private Instances m_dataset;
	private boolean m_is_leaf;
	private KDTreeNode m_left = null;
	private double m_max_distance;
	private KDTreeNode m_parent;
	private ValueRange[] m_ranges;
	private KDTreeNode m_right = null;
	private int m_split_index;
	private double m_split_value;
	private int m_size;

	private ArrayList<TreeItem> m_values;
	private KDTreeWindow m_window;

	public KDTreeNode(KDTreeWindow window, Instances dataset, KDTreeNode parent) {
		m_parent = parent;
		m_is_leaf = true;
		m_dataset = dataset;
		m_values = new ArrayList<TreeItem>();
		m_window = window;
		clearRanges();
	}

	public KDTreeNode(Instances dataset, KDTreeNode parent,
			ArrayList<TreeItem> values) {
		m_parent = parent;
		m_is_leaf = true;
		m_dataset = dataset;
		m_values = values;

		rescanRanges();
		if (shouldSplit())
			split();
	}

	public void add(final TreeItem to_add) {
		++m_size;
		if (m_is_leaf) {
			to_add.setOwner(this);
			m_values.add(to_add);
			if (shouldSplit()) // try to clean obsolete instances if split
								// criteria is met
			{
				int i = m_values.size() - 1;
				for (; i >= 0; --i) {
					if (to_add.id() - m_values.get(i).id() > m_window.size())
						break;
				}
				if (i >= 0) {
					int old_size = m_values.size();
					m_values = new ArrayList<TreeItem>(m_values.subList(i,m_values.size()));
					int diff = old_size - m_values.size();
					m_parent.subtractSize(diff);
					m_size -= diff;
				}
			}

			if (shouldSplit())
				split();

		} else {
			if (shouldCollapse())
			{
				m_is_leaf = true;
				m_values = instances();
				m_values.add(to_add);
				m_left = null;
				m_right = null;
			}
			else
			{
				double value = to_add.instance().value(m_split_index);
				if (value <= m_split_value)
					m_left.add(to_add);
				else
					m_right.add(to_add);
			}
		}

	}

	private void subtractSize(int i) {
		m_size -= i;
	}

	public ArrayList<TreeItem> instances() {
		if (isLeaf())
			return m_values;
		ArrayList<TreeItem> ret = new ArrayList<TreeItem>();
		ret.addAll(m_left.instances());
		ret.addAll(m_right.instances());
		return ret;
	}

	public boolean isLeaf() {
		return m_is_leaf;
	}



	private boolean shouldSplit() {
		return m_values.size() > SPLIT_VALUE;
	}

	private boolean shouldCollapse() {
		return size() < COLLAPSE_VALUE;
	}

	public int size() {
		return m_size;
	}

	public long getLastId() {
		if (m_values.size() > 0)
			return m_values.get(m_values.size() - 1).id();
		return 0;
	}

	private void split() {
		rescanRanges();
		if (m_split_index < 0)
			return;
		Iterator<TreeItem> it = m_values.iterator();
		KDTreeNode left = new KDTreeNode(m_window, m_dataset, this);
		KDTreeNode right = new KDTreeNode(m_window, m_dataset, this);
		while (it.hasNext()) {
			TreeItem inst = it.next();
			if (inst.instance().value(m_split_index) <= m_split_value)
				left.add(inst);
			else if (inst.instance().value(m_split_index) > m_split_value)
				right.add(inst);
		}
		m_left = left;
		if (left.size() == 0)
			throw new RuntimeException("Empty left leaf");
		m_right = right;
		if (right.size() == 0)
			throw new RuntimeException("Empty right leaf");
		m_values.clear();
		m_is_leaf = false;
	}

	public void print(PrintStream ps, int step) {
		for (int i = 0; i < step; i++)
			ps.print(" ");
		ps.print(m_is_leaf + " split dim " + m_split_index + " value "
				+ m_split_value + " size " + size());
		if (m_is_leaf) {
			ps.print(" nodes ");
			Iterator<TreeItem> it = m_values.iterator();
			while (it.hasNext())
				System.out.print(it.next().id() + " ");
			System.out.println();
		} else {
			ps.println();
			for (int i = 0; i < step; i++)
				ps.print(" ");

			ps.println("left: ");
			m_left.print(ps, step + 1);
			for (int i = 0; i < step; i++)
				ps.print(" ");
			ps.println("right: ");
			m_right.print(ps, step + 1);
		}
	}

	private void clearRanges() {
		m_split_index = -1;
		m_split_value = Double.MAX_VALUE;
		m_max_distance = Double.MIN_VALUE;
		m_ranges = new ValueRange[m_dataset.numAttributes()];
		for (int i = 0; i < m_ranges.length; i++)
			m_ranges[i] = new ValueRange();
	}

	private void rescanRanges() {
		clearRanges();
		Iterator<TreeItem> it = m_values.iterator();
		while (it.hasNext())
			updateRanges(it.next());

	}

	private void updateRanges(TreeItem inst) {
		for (int i = 0; i < inst.instance().numAttributes(); i++) {
			double value = inst.instance().value(i);
			boolean updated = false;
			if (m_ranges[i].m_max < value) {
				m_ranges[i].m_max = value;
				updated = true;
			}
			if (m_ranges[i].m_min > value) {
				m_ranges[i].m_min = value;
				updated = true;
			}
			if (updated) {
				double dist = Math.abs(m_ranges[i].m_max - m_ranges[i].m_min);
				if (dist > m_max_distance && dist > 0) {
					m_max_distance = dist;
					m_split_index = i;
					m_split_value = m_ranges[i].m_min + dist * 0.5;
				}
			}
		}

	}

	public void printRanges() {
		System.out.println("Ranges: index " + m_split_index + " value "
				+ m_split_value + " distance " + m_max_distance);
		int i = 0;
		for (ValueRange r : m_ranges) {
			System.out.println((i++) + " " + r.m_min + " <x< " + r.m_max);
		}
	}

	public int getSplitIndex() {
		return m_split_index;
	}

	public double getSplitValue() {
		return m_split_value;
	}

	public KDTreeNode left() {
		return m_left;
	}

	public KDTreeNode right() {
		return m_right;
	}

	public void schedule(Prediction instance) {
		// TODO Auto-generated method stub
		
	}

	/*
	 * private Future<> scheduleSearch(Instance in) {
	 * 
	 * }
	 * 
	 * public Future<> getVotes(KDTreeWindow win, Instance in) { if (isLeaf())
	 * return scheduleSearch(in); else { KDTreeNode nearer, further; boolean
	 * targetInLeft = in.value(getSplitIndex()) <= getSplitValue(); if
	 * (targetInLeft) { nearer = left(); further = right(); } else { nearer =
	 * left(); further = right(); } }
	 * 
	 * if (node.isLeaf()) { findNearestForNode(m_gpu_model, heap, instance,
	 * node, k); } else { KDTreeNode nearer, further;
	 * 
	 * boolean targetInLeft = instance.value(node.getSplitIndex()) <=
	 * node.getSplitValue(); if (targetInLeft) { nearer = node.left(); further =
	 * node.right(); } else { nearer = node.left(); further = node.right(); }
	 * findNearest(nearer,instance, k, heap, distanceToParents);
	 * 
	 * if (heap.size() < k) { // if haven't found the first k double
	 * distanceToSplitPlane = distanceToParents +
	 * m_distance.sqDifference(node.getSplitIndex(),
	 * instance.wekaInstance().value(node.getSplitIndex()),
	 * node.getSplitValue()); findNearest(further,instance, k, heap,
	 * distanceToSplitPlane); return; } else { // else see if ball centered at
	 * query intersects with the // other // side. double distanceToSplitPlane =
	 * distanceToParents + m_distance.sqDifference(node.getSplitIndex(),
	 * instance.wekaInstance().value(node.getSplitIndex()),
	 * node.getSplitValue()); if (heap.peek().distance() >=
	 * distanceToSplitPlane) { findNearest(further,instance, k, heap,
	 * distanceToSplitPlane); } }// end else } return null; }
	 */

}
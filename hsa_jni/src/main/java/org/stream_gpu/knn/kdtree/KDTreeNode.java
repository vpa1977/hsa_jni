package org.stream_gpu.knn.kdtree;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Iterator;

import org.moa.streams.EuclideanDistance;
import org.moa.streams.MaxList;
import org.moa.streams.MinMaxWindow;

import weka.core.Instance;
import weka.core.Instances;

public class KDTreeNode {

	public static int SPLIT_VALUE = 2048;
	public static int COLLAPSE_VALUE = 1024;

	private Instances m_dataset;
	private boolean m_is_leaf;
	private KDTreeNode m_left = null;
	private KDTreeNode m_right = null;
	private KDTreeNode m_parent;
	private int m_split_index;
	private double m_split_value;
	private int m_size;
	private int m_level;
	private EuclideanDistance m_node_distance;
	
	
	private ArrayList<WorkItem> m_scheduled_work;

	private ArrayList<TreeItem> m_values;
	private KDTreeWindow m_window;

	public KDTreeNode(KDTreeWindow window, Instances dataset, KDTreeNode parent) {
		m_parent = parent;
		m_level = parent == null ? 0 : parent.m_level+1;
		System.out.println("Level "+ m_level);
		
		m_is_leaf = true;
		m_dataset = dataset;
		m_values = new ArrayList<TreeItem>();
		m_window = window;
		m_scheduled_work = new ArrayList<WorkItem>();
		m_node_distance = new EuclideanDistance(dataset);
		
		
	}

	public KDTreeNode(Instances dataset, KDTreeNode parent,
			ArrayList<TreeItem> values) {
		m_parent = parent;
		m_level = parent == null ? 0 : parent.m_level+1;
		m_is_leaf = true;
		m_dataset = dataset;
		m_values = values;
		
		m_node_distance = new EuclideanDistance(dataset);
		
		
		for (TreeItem t : values)
			m_node_distance.update(t.instance(), null);
		if (shouldSplit())
			split();
	}

	public void add(final TreeItem to_add) {
		++m_size;
		if (m_is_leaf) {
			m_node_distance.update(to_add.instance(), null);
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
					for (int j = 0 ; j < i; ++i )
						m_node_distance.update( null, m_values.get(j).instance());
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
				m_node_distance = new EuclideanDistance(m_dataset);
				for (TreeItem t : m_values)
					m_node_distance.update(t.instance(), null);
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
		updateSplitValues();
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



	private void updateSplitValues() {
		// TODO Auto-generated method stub
		
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

	public void schedule(WorkItem instance) {
		if (isLeaf())
		{
			m_scheduled_work.add(instance);
			if (m_scheduled_work.size() == m_window.getWorkSize())
				processWork();
			for (WorkItem item : m_scheduled_work)
				m_parent.schedule(item);
		}
		else
		{
			WorkItem.PATH top = instance.getCurrentPath(m_level);
			if (top == WorkItem.PATH.NONE)
				scheduleOnNearest(instance);
			else
			{
				if (top != WorkItem.PATH.BOTH)
					scheduleOnFurtherst(instance, top);
				else
				{
					instance.popPath();
					if (m_parent!= null)
						m_parent.schedule(instance);
					else
						m_window.searchComplete(instance);
				}
			}
			
		}
	}

	/** 
	 * Process furtherst node
	 * @param instance
	 * @param top
	 */
	private void scheduleOnFurtherst(WorkItem instance, WorkItem.PATH top) {
		Instance wekaInstance = instance.m_prediction.instance();
		double distanceToParents = instance.getDistanceToParents(m_level);
		Heap heap = instance.m_heap;
		KDTreeNode next;
		if (top == WorkItem.PATH.LEFT)
			next = right();
		else
			next = left();
			
		double distanceToSplitPlane = distanceToParents 	+ m_window.distance(getSplitIndex(), wekaInstance.value(getSplitIndex()), getSplitValue());
		instance.addPath(m_level +1, WorkItem.PATH.NONE, distanceToSplitPlane);
		if (heap.size() < m_window.getK()) { // if haven't found the first k
			next.schedule(instance);
		} else {
			if (heap.peek().distance() >= distanceToSplitPlane) {
				next.schedule(instance);
			}
			else
			{
				m_window.searchComplete(instance);
			}
		}// end else
	}

	private void scheduleOnNearest(WorkItem instance) {
		KDTreeNode next;
		double distanceToParents = instance.getDistanceToParents(m_level);
		boolean targetInLeft = instance.m_prediction.instance().value(getSplitIndex()) <= getSplitValue();
		if (targetInLeft) {
			instance.setCurrentPath(m_level, WorkItem.PATH.LEFT, distanceToParents);
			next = left();
		} else {
			instance.setCurrentPath(m_level, WorkItem.PATH.RIGHT, distanceToParents);
			next = right();
		}
		next.schedule(instance);
	}
	
	/** 
	 * calculate distances and update heaps of all work-items
	 */
	private void processWork()
	{
		
	}


}
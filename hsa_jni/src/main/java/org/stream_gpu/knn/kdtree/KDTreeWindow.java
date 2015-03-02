package org.stream_gpu.knn.kdtree;

import hsa_jni.hsa_jni.SparseInstanceAccess;
import hsa_jni.hsa_jni.WekaHSAContext;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;

import org.moa.streams.EuclideanDistance;
import org.moa.streams.Prediction;

import weka.core.Instance;
import weka.core.Instances;

public class KDTreeWindow {
	private KDTreeNode m_root;
	private int m_window_size;
	private long m_current_id;
	private int m_k;
	private Queue<Prediction> m_result_queue;
	private ArrayList<TreeItem> m_window;
	private LinkedList<WorkItem> m_pending_queue;
	private EuclideanDistance m_min_max;

	private double m_min_box_rel_width = 1E-2;
	private Instances m_dataset;

	public KDTreeWindow(WekaHSAContext context, int window_size,
			Instances dataset) {
		m_window_size = window_size;
		m_dataset = dataset;
		m_pending_queue = new LinkedList<WorkItem>();
		m_window = new ArrayList<TreeItem>();
		if (m_dataset != null)
			clear();
	}

	public void add(Instance inst) {
		if (m_dataset == null)
		{
			m_dataset = inst.dataset();
			clear();
		}
		TreeItem to_add = new TreeItem(m_current_id++,
				new SparseInstanceAccess(inst));
		TreeItem removed = null;
		m_window.add(to_add);
		if (m_window.size() > m_window_size)
			removed = m_window.remove(0);
		m_min_max.update(to_add.instance(), removed == null ? null :removed.instance());
		m_root.add(to_add);
	}

	public int getInstanceCount() {
		return m_root.size();
	}

	public int size() {
		return m_window_size;
	}

	public void setWorkQueue(Queue<Prediction> result_queue) {
		m_result_queue = result_queue;
	}

	public void evaluate(int k, Prediction prediction) {
		WorkItem newSearch = new WorkItem(k, prediction);
		m_pending_queue.add(newSearch);
		m_root.schedule(newSearch);
	}

	public boolean hasPendingQueries() {
		return m_pending_queue.size() > 0;
	}

	public int getWorkSize() {
		return 1024;
	}

	public int getK() {
		return m_k;
	}

	public void searchComplete(WorkItem instance) {
		m_pending_queue.remove(instance);
		m_result_queue.add(instance.m_prediction);
	}

	public double distance(int splitIndex, double value, double splitValue) {
		return m_min_max.sqDifference(splitIndex, value, splitValue);
	}

	public void clear() {
		if (m_dataset == null)
			return;
		m_root = new KDTreeNode(this, m_dataset, null);
		m_root.SPLIT_VALUE = Math.min(m_window_size / 16, 32528);
		m_root.COLLAPSE_VALUE = Math.min(m_window_size / 32, 4096);
		m_min_max = new EuclideanDistance(m_dataset);
		
	}

}

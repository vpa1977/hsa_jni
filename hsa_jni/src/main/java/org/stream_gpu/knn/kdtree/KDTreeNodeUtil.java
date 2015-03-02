package org.stream_gpu.knn.kdtree;

import org.moa.streams.MinMaxWindow;

public class KDTreeNodeUtil {
	
	private int m_class_index;
	private boolean m_normalize_node_width;

	public KDTreeNodeUtil( int classIndex, boolean normalizeNodeWidth)
	{
		m_class_index = classIndex;
		m_normalize_node_width = normalizeNodeWidth;
	}
	
	
	  /** 
	   * Returns the maximum attribute width of instances/points 
	   * in a KDTreeNode relative to the whole dataset. 
	   * 
	   * @param nodeRanges The attribute ranges of the 
	   * KDTreeNode whose maximum relative width is to be 
	   * determined.
	   * @param universe The attribute ranges of the whole
	   * dataset (training instances + test instances so 
	   * far encountered).
	   * @return The maximum relative width
	   */
	  public double getMaxRelativeNodeWidth(MinMaxWindow nodeRanges, MinMaxWindow universe) {
	    int widest = widestDim(nodeRanges, universe);
	    if(widest < 0)
	    	return 0.0;
	    else
	    	return nodeRanges.max(widest) / universe.max(widest);
	  }
	  
	  /**
	   * Returns the widest dimension/attribute in a 
	   * KDTreeNode (widest after normalizing).
	   * @param nodeRanges The attribute ranges of 
	   * the KDTreeNode.
	   * @param universe The attribute ranges of the 
	   * whole dataset (training instances + test 
	   * instances so far encountered).
	   * @return The index of the widest 
	   * dimension/attribute.
	   */
	  protected int widestDim(MinMaxWindow nodeRanges, MinMaxWindow universe) {
	    final int classIdx = m_class_index;
	    double widest = 0.0;
	    int w = -1;
	    if (m_normalize_node_width) {
	      for (int i = 0; i < nodeRanges.length(); i++) {
	        double newWidest = nodeRanges.width(i)/ universe.width(i);
	        if (newWidest > widest) {
	          if (i == classIdx)
	            continue;
	          widest = newWidest;
	          w = i;
	        }
	      }
	    } else {
	      for (int i = 0; i < nodeRanges.length(); i++) {
	        if (nodeRanges.width(i) > widest) {
	          if (i == classIdx)
	            continue;
	          widest = nodeRanges.width(i);
	          w = i;
	        }
	      }
	    }
	    return w;
	  }
}

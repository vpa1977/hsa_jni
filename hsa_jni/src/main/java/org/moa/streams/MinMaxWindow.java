package org.moa.streams;

import weka.core.Instance;

public interface MinMaxWindow {
	public void update(Instance add, Instance remove);
	public double max(int i);
	public double min(int i);
	public int length();
	public double width(int i);
}
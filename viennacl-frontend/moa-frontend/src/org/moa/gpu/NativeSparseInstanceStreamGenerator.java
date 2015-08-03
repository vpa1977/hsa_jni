package org.moa.gpu;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Random;

import org.moa.gpu.bridge.NativeSparseInstance;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import moa.MOAObject;
import moa.core.InstancesHeader;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.ClassOption;
import moa.options.IntOption;
import moa.options.OptionHandler;
import moa.streams.InstanceStream;
import moa.streams.generators.RandomTreeGenerator;
import moa.tasks.TaskMonitor;

public class NativeSparseInstanceStreamGenerator extends SparseInstanceStreamGenerator {
	public Instance nextInstance() {
		NativeSparseInstance nativeInstance = new NativeSparseInstance((AccessibleSparseInstance )super.nextInstance()); 
		return nativeInstance;
	}
}

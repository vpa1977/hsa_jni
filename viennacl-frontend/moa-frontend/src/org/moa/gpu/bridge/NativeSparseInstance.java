package org.moa.gpu.bridge;

import java.util.Enumeration;

import org.moa.gpu.AccessibleSparseInstance;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class NativeSparseInstance implements NativeInstance {


	public NativeSparseInstance(Instance source)
	{
		m_instance = new AccessibleSparseInstance(source);
		init();
	}

	@Override
	public Attribute attribute(int index) {
		return m_instance.attribute(index);
	}

	@Override
	public Attribute attributeSparse(int indexOfIndex) {
		return m_instance.attributeSparse(indexOfIndex);
	}

	@Override
	public Attribute classAttribute() {
		return m_instance.classAttribute();
	}

	@Override
	public int classIndex() {
		// TODO Auto-generated method stub
		return m_instance.classIndex();
	}

	@Override
	public boolean classIsMissing() {
		// TODO Auto-generated method stub
		return m_instance.classIsMissing();
	}

	@Override
	public double classValue() {
		return m_instance.classValue();
	}

	@Override
	public Instances dataset() {
		return m_instance.dataset();
	}

	@Override
	public void deleteAttributeAt(int position) {
		m_instance.deleteAttributeAt(position);
		writeToNative();
	}


	@Override
	public Enumeration enumerateAttributes() {
		return m_instance.enumerateAttributes();
	}

	@Override
	public boolean equalHeaders(Instance inst) {
		return m_instance.equalHeaders(inst);
	}

	@Override
	public String equalHeadersMsg(Instance inst) {
		// TODO Auto-generated method stub
		return m_instance.equalHeadersMsg(inst);
	}

	@Override
	public boolean hasMissingValue() {
		// TODO Auto-generated method stub
		return m_instance.hasMissingValue();
	}

	@Override
	public int index(int position) {
		return m_instance.index(position);
	}

	@Override
	public void insertAttributeAt(int position) {
		m_instance.insertAttributeAt(position);
		writeToNative();
	}

	@Override
	public boolean isMissing(int attIndex) {
		return m_instance.isMissing(attIndex);
	}

	@Override
	public boolean isMissingSparse(int indexOfIndex) {
		return m_instance.isMissingSparse(indexOfIndex);
	}

	@Override
	public boolean isMissing(Attribute att) {
		return isMissing(att);
	}

	@Override
	public Instance mergeInstance(Instance inst) {
		Instance result = m_instance.mergeInstance(inst);
		writeToNative();
		return result;
	}

	@Override
	public int numAttributes() {
		return m_instance.numAttributes();
	}

	@Override
	public int numClasses() {
		return m_instance.numClasses();
	}

	@Override
	public int numValues() {
		return m_instance.numValues();
	}

	@Override
	public void replaceMissingValues(double[] array) {
		m_instance.replaceMissingValues(array);
		writeToNative();
	}

	@Override
	public void setClassMissing() {
		m_instance.setClassMissing();
		writeToNative();
	}

	@Override
	public void setClassValue(double value) {
		m_instance.setClassValue(value);
		writeToNative();
	}

	@Override
	public void setClassValue(String value) {
		m_instance.setClassValue(value);
		writeToNative();
		
	}

	@Override
	public void setDataset(Instances instances) {
		m_instance.setDataset(instances);
		
	}

	@Override
	public void setMissing(int attIndex) {
		m_instance.setMissing(attIndex);
		writeToNative();
	}

	@Override
	public void setMissing(Attribute att) {
		m_instance.setMissing(att);
		writeToNative();
		
	}

	@Override
	public void setValue(int attIndex, double value) {
		m_instance.setValue(attIndex, value);
		writeToNative();
		
	}

	@Override
	public void setValueSparse(int indexOfIndex, double value) {
		m_instance.setValue(indexOfIndex, value);
		writeToNative();
		
	}

	@Override
	public void setValue(int attIndex, String value) {
		m_instance.setValue(attIndex, value);
		writeToNative();
		
	}

	@Override
	public void setValue(Attribute att, double value) {
		m_instance.setValue(att, value);
		writeToNative();
		
	}

	@Override
	public void setValue(Attribute att, String value) {
		m_instance.setValue(att, value);
		writeToNative();
		
	}

	@Override
	public void setWeight(double weight) {
		m_instance.setWeight(weight);
		writeToNative();
		
	}

	@Override
	public Instances relationalValue(int attIndex) {
		return m_instance.relationalValue(attIndex);
	}

	@Override
	public Instances relationalValue(Attribute att) {
		return m_instance.relationalValue(att);
		}

	@Override
	public String stringValue(int attIndex) {
		return m_instance.stringValue(attIndex);
	}

	@Override
	public String stringValue(Attribute att) {
		return m_instance.stringValue(att);	
		}

	@Override
	public double[] toDoubleArray() {
		return m_instance.toDoubleArray();
	}

	@Override
	public String toStringNoWeight(int afterDecimalPoint) {
		return m_instance.toStringNoWeight(afterDecimalPoint);
	}

	@Override
	public String toStringNoWeight() {
		return m_instance.toStringNoWeight();
		}

	@Override
	public String toStringMaxDecimalDigits(int afterDecimalPoint) {
		
		return m_instance.toStringMaxDecimalDigits(afterDecimalPoint);
	}

	@Override
	public String toString(int attIndex, int afterDecimalPoint) {
		return m_instance.toString(attIndex, afterDecimalPoint);
	}

	@Override
	public String toString(int attIndex) {
		return m_instance.toString(attIndex);
	}

	@Override
	public String toString(Attribute att, int afterDecimalPoint) {
		return m_instance.toString(att, afterDecimalPoint);
	}

	@Override
	public String toString(Attribute att) {
		return m_instance.toString(att);
	}

	@Override
	public double value(int attIndex) {
		return m_instance.value(attIndex);
	}

	@Override
	public double valueSparse(int indexOfIndex) {
		return m_instance.valueSparse(indexOfIndex);
	}

	@Override
	public double value(Attribute att) {
		return m_instance.value(att);
	}

	
	public double weight() {
		return m_instance.weight();
	}

	@Override
	public Object copy() {
		return new NativeSparseInstance(this);
	}
	
	
	@Override
	protected void finalize() throws Throwable {
		super.finalize();
		release();
	}

	private native void writeToNative();
	public native void release();
	private native void init();
	
	private long m_native_context;
	private AccessibleSparseInstance m_instance;

}

package test;

import java.util.ArrayList;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class InstanceCreator {

	public static void main(String[] args) {
		ArrayList<Attribute> attrs = new ArrayList<Attribute>();
		attrs.add( new Attribute("class1"));
		attrs.add( new Attribute("attribute1"));
		Instances test = new Instances("test_set", attrs,1);
		test.setClassIndex(0);
		test.add( new DenseInstance(1, new double[] {1 , 0.1}));
		test.add( new DenseInstance(1, new double[] {1 , 0.2}));
		test.add( new DenseInstance(1, new double[] {1 , 0.3}));
		test.add( new DenseInstance(1, new double[] {0 , 0.6}));
		test.add( new DenseInstance(1, new double[] {0 , 0.7}));
		test.add( new DenseInstance(1, new double[] {0 , 0.8}));
		System.out.println(test);
	}

}

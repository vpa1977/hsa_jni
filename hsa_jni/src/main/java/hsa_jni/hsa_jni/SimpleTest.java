package hsa_jni.hsa_jni;

import java.util.ArrayList;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class SimpleTest {
	public static void main(String[] args)
	{
		//System.out.println(System.getProperty("java.class.path"));
		//System.exit(0);
		WekaHSAContext context = new WekaHSAContext();
		KnnGpuClassifier clazz = new KnnGpuClassifier(context, 2, 2);
		
		ArrayList<Attribute> attinfo = new ArrayList<Attribute>();
		attinfo.add( new Attribute("1"));
		attinfo.add( new Attribute("2"));
		attinfo.add( new Attribute("3"));
		attinfo.add( new Attribute("4"));
		attinfo.add( new Attribute("class"));
		
		Instances dataset = new Instances( "test", attinfo , 1);
		dataset.setClassIndex(0);
		Instance inst = new DenseInstance(1, new double[]{ 1, 1, 1 , 1 ,1});
		Instance inst1 = new DenseInstance(1, new double[]{ 2, 2, 2, 2 , 1});
		inst.setDataset(dataset);
		inst1.setDataset(dataset);
		
		clazz.trainOnInstance(inst);
		clazz.trainOnInstance(inst1);
		
		double[] votes = clazz.getVotesForInstance(inst);
		
		System.out.println("votes are " + votes[0] + " " + votes[1]);
		
		
	}
}

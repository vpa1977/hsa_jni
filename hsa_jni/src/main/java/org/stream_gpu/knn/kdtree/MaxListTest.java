package org.stream_gpu.knn.kdtree;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;

import org.junit.Test;
import org.moa.streams.MaxList;

import weka.core.Attribute;
import weka.core.Debug.Random;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

public class MaxListTest {
	
	
	private Instances createDataSet(String name, int numAttributes, int numInstances)
	{
		ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
		for (int i = 0 ;i < numAttributes ; ++i)
		{
			Attribute att = new Attribute(""+i);
			attInfo.add(att);
		}
		Instances dataset = new Instances("test", attInfo, 10);
		return dataset;
	}
	
	

	@Test
	public void test() {
		Instances dataset = createDataSet("test", 2, 10);
		
		Instance one = new DenseInstance(1, new double[] { 1, 1 } );
		Instance two = new DenseInstance(1, new double[] { 2, 1 } );
		Instance three = new DenseInstance(1, new double[] { 3, 1 } );
		Instance four  = new DenseInstance(1, new double[] { 4, 1 } );
		Instance five  = new DenseInstance(1, new double[] { 5, 1 } );
		
		MaxList maxList = new MaxList(dataset);
		maxList.update(one, null);
		
		assertEquals(maxList.max(0), 1,0.01);
		assertEquals(maxList.min(0), 1,0.01);
		
		maxList.update(two, null);
		
		assertEquals(maxList.max(0), 2,0.01);
		assertEquals(maxList.min(0), 1,0.01);
		
		maxList.update(three, one);
		
		assertEquals(maxList.max(0), 3,0.01);
		assertEquals(maxList.min(0), 2,0.01);
		
		maxList.update(two, two);
		
		assertEquals(maxList.max(0), 3,0.01);
		assertEquals(maxList.min(0), 2,0.01);
	}
	
	@Test
	public void compareSpeed()
	{
		Instances dataset = createDataSet("test", 2, 10);
		EuclideanDistance distance = new EuclideanDistance(dataset);

		MaxList maxList = new MaxList(dataset);
		Random rnd = new Random();
		long t1 = System.currentTimeMillis();
		for (int i = 0 ; i < 1000000 ; ++i)
			distance.update(new DenseInstance(1, new double[] { rnd.nextDouble(), rnd.nextDouble() } ));
		long t2 = System.currentTimeMillis();
		for (int i = 0 ; i < 1000000; ++i)
			maxList.update(new DenseInstance(1, new double[] { rnd.nextDouble(), rnd.nextDouble() } ), null);
		long t3 = System.currentTimeMillis();
		long timeMaxList = t3 - t2;
		long timeUpdate = t2 - t1;
		//assertTrue(timeMaxList < timeUpdate);
		
	}

}

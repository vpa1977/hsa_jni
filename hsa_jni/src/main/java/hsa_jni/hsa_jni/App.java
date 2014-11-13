package hsa_jni.hsa_jni;

import hsa_jni.hsa_jni.WekaHSAContext.KnnNativeContext;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.concurrent.TimeUnit;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws IOException, NoSuchMethodException, SecurityException
    {
    	
		
    	System.out.println(System.getProperty("java.class.path"));

    	System.out.println(System.getProperty("java.class.path"));
    	ArffLoader loader = new ArffLoader();
    	loader.reset();
    	loader.setSource(new File("electNormNew.arff"));
		Instances instances = loader.getStructure();
    	System.out.println("----");
    	
    	
    	WekaHSAContext context = new WekaHSAContext();
    	long start, end;
    	KnnNativeContext knn = context.create( instances, 16*8192);
    	
    	int window_size = 4*65535;
    	int num_attrs = 16;
    	int num_numerics = 10;
    	
    	
    	double[] window = new double[ window_size * num_attrs ];
    	double[] ranges = new double[ num_attrs * 2];
    	
    	//long start = System.currentTimeMillis();
    	
    	for (int i = 0 ; i < 1000 ; i ++ )
    		knn.rescanAllSeq();
    	//long end = System.currentTimeMillis();
    	
    	//double diff = end - start;
    	//System.out.println("Done in "+ (diff/1000) + "val "+ knn.m_ranges[0]);
    	
    	
    	//start = System.currentTimeMillis();
    	
    	for (int i = 0 ; i < 1000 ; i ++ )
    		knn.rescanAll();
    	//end = System.currentTimeMillis();
    	
    	//diff = end - start;
    	//System.out.println("Done in "+ (diff/1000));
    	//System.exit(0);
    	
    	Instance inst;
    	int count = 0;
    	Instance test = loader.getNextInstance(instances);
/*
    	{
    		int iters = knn.m_instances.length;
	    	for (int i = 0 ;i < iters; ++i)
	    		knn.addInstance(test, true);

	    	start = System.currentTimeMillis();
	    	for (int i = 0; i < 1000 ; ++i)
	    		knn.rescanAll();
			end = System.currentTimeMillis();
	    	System.out.println("1000 iterations done in  "+ (end-start));
	    	
	    	start = System.currentTimeMillis();
	    	for (int i = 0; i < 1000 ; ++i)
	    		knn.rescanAllSeq();
			end = System.currentTimeMillis();
	    	System.out.println("1000 iterations done in  "+ (end -start));

    	}
    	 
  */  	
    	
		while ( (inst = loader.getNextInstance(instances)) != null)
		{
			knn.addInstance(inst);
		}
		
		TimeUnit nano = TimeUnit.NANOSECONDS; 
		
		start = System.nanoTime();
	//	for (int i = 0 ;i < 1000 ;  ++i)
	//		knn.computeKnn(test);
		end = System.nanoTime();
    	System.out.println("1000 iterations done in  "+nano.toMillis(end-start));
	/*	
		System.out.println("Check ranges");
		
		long start = System.currentTimeMillis();
		for (int i = 0 ;i < 1000 ;  ++i)
			knn.computeKnn(test);
		long end = System.currentTimeMillis();
    	System.out.println("Time "+ (end -start));
		
		
		
    	/*ArrayList<Attribute> attinfo = new ArrayList<Attribute>();
		attinfo.add( new Attribute("1"));
		attinfo.add( new Attribute("2"));
		attinfo.add( new Attribute("3"));
		attinfo.add( new Attribute("4"));
		attinfo.add( new Attribute("class"));
		
		Instances dataset = new Instances( "test", attinfo , 1);
		dataset.setClassIndex(0);

    
    	
    	WekaHSAContext context = new WekaHSAContext();
    	
    	KnnNativeContext knn = context.create( dataset, 16);
    	
    	for (int i = 0 ; i < 32; ++i)
    	{
    		Instance inst = new DenseInstance(1, new double[]{ 1, 1, 1 , 1 ,1});
    		inst.setDataset(dataset);
    		knn.addInstance(inst);
    	}
    	
    	
    	
		Instance inst = new DenseInstance(1, new double[]{ 0, 0, 0 , 0 ,0});
		inst.setDataset(dataset);
    	
		knn.computeKnn(inst);*/
    	
    	
    	  
    	 
    	/*
    	int count = 0;
    	System.out.println(count);
    	
    	Iterator<Instance> it = data.iterator();
    	while (it.hasNext())
    	{
    		context.AddInstance(it.next());
    		context.AddPureJavaInstance(it.next());
    	}
    		
    	context.dispose();
    	context.disposePureJava();
    	
    	long start = System.nanoTime();
    	it = data.iterator();
    	for (int i = 0; i < 4*8192 ; ++i)
    	{
    		context.AddInstance(it.next());
    	}
    	
    	context.dispose();
    	long end = System.nanoTime();
    	
    	System.out.println("Time "+ (end -start)/1000000);
    	
    	
    	start = System.nanoTime();
    	it = data.iterator();
    	for (int i = 0; i < 4*8192 ; ++i)
    		context.AddPureJavaInstance(it.next());
    	context.commit();
    	context.disposePureJava();
    	end = System.nanoTime();
    	
    	System.out.println("Time2 "+ (end -start)/1000000); 
    	*/
    	
    	

    }
}

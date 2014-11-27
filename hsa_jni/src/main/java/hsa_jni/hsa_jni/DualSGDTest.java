package hsa_jni.hsa_jni;



import weka.classifiers.lazy.IBk;
import weka.core.neighboursearch.LinearNNSearch;
import moa.classifiers.meta.WEKAClassifier;
import moa.streams.InstanceStream;
import moa.streams.generators.RandomTreeGenerator;
import moa.tasks.EvaluatePeriodicHeldOutTest;
import moa.tasks.NullMonitor;

import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.JavaCL;

public class DualSGDTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Throwable {
		
		int k = Integer.parseInt(args[0]);
		int window = Integer.parseInt(args[1]);
		
		int test_size  = Integer.parseInt(args[2]);
		int train_size = Integer.parseInt(args[3]);
		
		
		
		InstanceStream generator = (InstanceStream)Class.forName(args[4]).newInstance();
		
		RandomTreeGenerator rtg = (RandomTreeGenerator) generator;
		rtg.numNominalsOption.setValue(32);
		rtg.numNumericsOption.setValue(32);
		
		
		WekaHSAContext context = new WekaHSAContext();
		
		KnnGpuClassifier classifier = new KnnGpuClassifier(context,window, k);
		
		SGD hsaSGD = new SGD();
		
		moa.classifiers.functions.SGD moaSGD = new moa.classifiers.functions.SGD(); 
		
		System.out.println("Test : window="+ window + " k =" + k);
		System.out.println("	 : test_size="+ test_size + " train_size =" + train_size);
		System.out.println("Stream: "+generator.getClass());// + " "+ generator.getOptions().getAsCLIString());
		
		EvaluatePeriodicHeldOutTest test = new EvaluatePeriodicHeldOutTest();
		test.streamOption.setCurrentObject(generator);
		test.learnerOption.setCurrentObject( moaSGD);
		test.testSizeOption.setValue(test_size);
		test.trainSizeOption.setValue(train_size);
		System.out.println("------Classifier:"+ test.learnerOption.getValueAsCLIString()  );
		Object ret = test.doTask(new NullMonitor(),null);
		System.out.println(ret);
		System.out.println("---------------------------------------------------------------------------");
		
		test = new EvaluatePeriodicHeldOutTest();
		test.learnerOption.setCurrentObject(hsaSGD);
		test.streamOption.setCurrentObject(generator);
		test.testSizeOption.setValue(test_size);
		test.trainSizeOption.setValue(train_size);
		System.out.println("------Classifier:"+ test.learnerOption.getValueAsCLIString()  );
		ret = test.doTask(new NullMonitor(),null);
		System.out.println(ret);
		System.out.println("---------------------------------------------------------------------------");
		
	}

}

package hsa_jni.hsa_jni;



import java.io.FileInputStream;
import java.util.List;

import hsa_jni.hsa_jni.config.Experiments;
import hsa_jni.hsa_jni.config.FirstKnnExperiment;

import javax.xml.bind.JAXB;

import weka.classifiers.lazy.IBk;
import weka.core.neighboursearch.KDTree;
import weka.core.neighboursearch.LinearNNSearch;
import moa.classifiers.meta.WEKAClassifier;
import moa.options.AbstractOptionHandler;
import moa.options.ClassOption;
import moa.streams.InstanceStream;
import moa.streams.generators.RandomTreeGenerator;
import moa.tasks.EvaluatePeriodicHeldOutTest;
import moa.tasks.NullMonitor;

import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.JavaCL;

public class DualKdTreeTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Throwable {
		
		//int k = Integer.parseInt(args[0]);
		//int window = Integer.parseInt(args[1]);
		
	//	int test_size  = Integer.parseInt(args[2]);
		//int train_size = Integer.parseInt(args[3]);
		
		System.out.println(System.getProperty("java.class.path"));
		Experiments experiments = JAXB.unmarshal(new FileInputStream(args[0]),Experiments.class );
		
		List<FirstKnnExperiment> knns = experiments.getKnn();
		
		for (FirstKnnExperiment experiment : knns )
		{
		
			try {
				
				String generatorCLI = experiment.getGenerator();
				
				InstanceStream generator = (InstanceStream)ClassOption.cliStringToObject(generatorCLI,InstanceStream.class, null );
				
				WekaHSAContext context = new WekaHSAContext();
				
				KdTreeKnnGpuClassifier classifier = new KdTreeKnnGpuClassifier(context,experiment.getWindow(), experiment.getK());
				
				IBk kMeans = new IBk(experiment.getK());
				kMeans.setNearestNeighbourSearchAlgorithm(new KDTree());
				kMeans.setWindowSize(experiment.getWindow());
				
				int window = experiment.getWindow();
				int train_size = experiment.getTrainSize();
				int test_size = experiment.getTestSize();
				int k = experiment.getK();
				
				WEKAClassifier wekaClassifier = new WEKAClassifier();
				wekaClassifier.baseLearnerOption.setCurrentObject(kMeans);
				System.out.println("Test : window="+ window + " k =" + k);
				System.out.println("	 : test_size="+ test_size + " train_size =" + train_size);
				System.out.println("Stream: "+ ((AbstractOptionHandler)generator).getCLICreationString(InstanceStream.class));// + " "+ generator.getOptions().getAsCLIString());
				
				
				
				EvaluatePeriodicHeldOutTest test = new EvaluatePeriodicHeldOutTest();
				test.streamOption.setCurrentObject(generator);
				test.learnerOption.setCurrentObject(classifier);
				test.testSizeOption.setValue(test_size);
				test.trainSizeOption.setValue(train_size);
				System.out.println("------Classifier:"+ test.learnerOption.getValueAsCLIString()  );
				Object ret = test.doTask(new NullMonitor(),null);
				System.out.println(ret);
				System.out.println("---------------------------------------------------------------------------");
				
				test = new EvaluatePeriodicHeldOutTest();
				test.learnerOption.setCurrentObject(wekaClassifier);
				test.streamOption.setCurrentObject(generator);
				test.testSizeOption.setValue(test_size);
				test.trainSizeOption.setValue(train_size);
				System.out.println("------Classifier:"+ test.learnerOption.getValueAsCLIString()  );
				ret = test.doTask(new NullMonitor(),null);
				System.out.println(ret);
				System.out.println("---------------------------------------------------------------------------");

			}
			catch (Throwable t )
			{
				t.printStackTrace();
			}
		}
		
	}

}

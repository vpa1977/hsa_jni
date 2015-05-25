

import org.moa.gpu.Context;
import org.moa.gpu.SGD;
import org.moa.gpu.SimpleDirectMemoryBatchInstances;
import org.moa.gpu.SimpleWindow;
import org.moa.gpu.config.*;

import java.io.FileInputStream;
import java.util.List;

import javax.xml.bind.JAXB;

import weka.classifiers.lazy.IBk;
import weka.core.neighboursearch.LinearNNSearch;
import moa.classifiers.meta.WEKAClassifier;
import moa.options.AbstractOptionHandler;
import moa.options.ClassOption;
import moa.streams.InstanceStream;
import moa.streams.generators.RandomTreeGenerator;
import moa.tasks.EvaluatePeriodicHeldOutTest;
import moa.tasks.MeasureStreamSpeed;
import moa.tasks.NullMonitor;

import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.JavaCL;

public class StreamTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Throwable {
		System.out.println( System.getProperty("java.class.path"));
		Context.load();
		
		RandomTreeGenerator d;
		
		
		Experiments experiments = JAXB.unmarshal(new FileInputStream(args[0]),Experiments.class );
		
		List<SGDExperiment> list = experiments.getSgd();
		
		for (SGDExperiment experiment : list )
		{
			try {
				
								
				
				test.SGD moaSGD = new test.SGD();
				String generatorCLI = experiment.getGenerator();
				
				InstanceStream generator = (InstanceStream)ClassOption.cliStringToObject(generatorCLI,InstanceStream.class, null );
				
				MeasureStreamSpeed speed = new MeasureStreamSpeed();
				speed.streamOption.setCurrentObject(generator);
				Object ret = speed.doTask(new NullMonitor(),null);
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


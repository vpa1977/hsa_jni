package test.java;

import static org.junit.Assert.*;

import java.io.FileInputStream;

import org.junit.Test;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.neighboursearch.IProjection;
import weka.core.neighboursearch.RandomProjectionFold;

public class RandomProjectionFoldTest {

	@Test
	public void testCreate() throws Throwable {
	       ArffLoader loader = new ArffLoader();
	       loader.setSource(new FileInputStream("F:\\weka_377\\data\\cpu.arff"));
	       loader.getStructure();
	       Instances dataset = loader.getDataSet();
	       IProjection fold = new RandomProjectionFold(dataset, 100, 100);
	       IProjection fold1 = new RandomProjectionFold(dataset, 2, 2);
	}

}

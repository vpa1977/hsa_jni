package test;

import static org.junit.Assert.*;

import java.lang.annotation.Native;
import java.util.ArrayList;

import org.junit.Test;
import org.moa.gpu.NativeDenseInstanceStreamGenerator;
import org.moa.gpu.SparseInstanceStreamGenerator;
import org.moa.gpu.bridge.DenseOffHeapBuffer;
import org.moa.gpu.bridge.NativeDenseInstance;
import org.moa.gpu.bridge.NativeDenseInstanceBatch;
import org.moa.gpu.bridge.SparseOffHeapBuffer;

import moa.streams.generators.RandomTreeGenerator;
import weka.core.Instance;
import weka.core.SparseInstance;

public class TestOffHeapSparseBuffers {

	@Test
	public void testReadWrite() {
		System.loadLibrary("moa-frontend-lib");
		SparseInstanceStreamGenerator spr = new SparseInstanceStreamGenerator();
		spr.numAttributesOption.setValue(64);
		spr.numMappingsOption.setValue(1);
		spr.prepareForUse();

		ArrayList<Instance> store = new ArrayList<Instance>();
		for (int i = 0; i < 1024; ++i)
			store.add(spr.nextInstance());

		long clock = System.currentTimeMillis();
		for (int l = 0; l < 100000000; ++l) {
			SparseOffHeapBuffer sparseOffHeap = new SparseOffHeapBuffer(1024, spr.getHeader().numAttributes()-1);

			sparseOffHeap.begin();
			for (int i = 0; i < 1024; ++i) {
				sparseOffHeap.set(store.get(i), i);
			}
			sparseOffHeap.commit();
			for (int i = 0; i < 1024; ++i) {
				Instance generated = store.get(i);
				Instance inst = sparseOffHeap.read(spr.getHeader(), i);
				double[] fw_arr = inst.toDoubleArray();
				double[] st_arr = generated.toDoubleArray();
				assertEquals(fw_arr.length, st_arr.length);
				for (int k = 0; k < fw_arr.length; ++k)
					assertEquals("Test on " + k, fw_arr[k], st_arr[k], 0.000001);
			}

			sparseOffHeap.release();
		}
		long end = System.currentTimeMillis();
		double time_msec = end - clock;

		// speed = 1000/speed;
		System.out.println(time_msec);
	}

}

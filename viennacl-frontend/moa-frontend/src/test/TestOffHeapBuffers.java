package test;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Test;
import org.moa.gpu.bridge.DenseOffHeapBuffer;

import moa.streams.generators.RandomTreeGenerator;
import weka.core.Instance;

public class TestOffHeapBuffers {

	@Test
	public void test() {
		System.loadLibrary("moa-frontend-lib");
		RandomTreeGenerator rtg = new RandomTreeGenerator();
		rtg.prepareForUse();
		Instance inst = rtg.nextInstance();
		ArrayList<Instance> store = new ArrayList<Instance>();
		for (int i = 0; i < 1024; ++i)
			store.add(rtg.nextInstance());

		long clock = System.currentTimeMillis();
		for (int l = 0 ;l < 100000000; ++l)
		{
			DenseOffHeapBuffer denseOffHeap = new DenseOffHeapBuffer(1024, inst.numAttributes()-1);
			denseOffHeap.begin();
			for (int i = 0; i < 1024; ++i)
			{
				denseOffHeap.set(store.get(i), i);
			}
		/*	for (int i = 0;i < 1024; ++i)
			{
				Instance generated = store.get(i);
				denseOffHeap.read(inst, i);
				double[] fw_arr = inst.toDoubleArray();
				double[] st_arr = generated.toDoubleArray();
				assertEquals(fw_arr.length, st_arr.length);
				for (int k = 0 ;k < fw_arr.length; ++k)
					assertEquals(fw_arr[k], st_arr[k], 0.000001);
			}*/
			denseOffHeap.commit();
			denseOffHeap.release();
		}
		long end = System.currentTimeMillis();
		double time_msec = end - clock;
		double speed = time_msec / (1024 * 1000);
		//speed = 1000/speed;
		System.out.println(speed);
	}

}

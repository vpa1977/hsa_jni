package test;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;

import org.junit.Test;
import org.moa.gpu.NativeDenseInstanceStreamGenerator;
import org.moa.gpu.bridge.DenseOffHeapBuffer;
import org.moa.gpu.bridge.NativeDenseInstance;
import org.moa.gpu.bridge.NativeDenseInstanceBatch;

import moa.streams.generators.RandomTreeGenerator;
import weka.core.Instance;

public class TestOffHeapDenseBuffers {


	@Test
	public void testReadWrite() {
		System.loadLibrary("moa-frontend-lib");
		RandomTreeGenerator rtg = new RandomTreeGenerator();
		rtg.prepareForUse();
		Instance inst = rtg.nextInstance();
		ArrayList<Instance> store = new ArrayList<Instance>();
		for (int i = 0; i < 1024; ++i)
			store.add(rtg.nextInstance());

		long clock = System.currentTimeMillis();
		for (int l = 0 ;l < 10; ++l)
		{
			DenseOffHeapBuffer denseOffHeap = new DenseOffHeapBuffer(1024, inst.numAttributes()-1);
			denseOffHeap.begin();
			for (int i = 0; i < 1024; ++i)
			{
				denseOffHeap.set(store.get(i), i);
			}
		for (int i = 0;i < 1024; ++i)
			{
				Instance generated = store.get(i);
				denseOffHeap.read(inst, i);
				double[] fw_arr = inst.toDoubleArray();
				double[] st_arr = generated.toDoubleArray();
				assertEquals(fw_arr.length, st_arr.length);
				for (int k = 0 ;k < fw_arr.length; ++k)
					assertEquals(fw_arr[k], st_arr[k], 0.000001);
			}
			denseOffHeap.commit();
			denseOffHeap.release();
		}
		long end = System.currentTimeMillis();
		double time_msec = end - clock;
	
		//speed = 1000/speed;
		System.out.println(time_msec);
	}
	
	

	@Test
	public void testNativeDenseStuff() {
		System.loadLibrary("moa-frontend-lib");
		
		NativeDenseInstanceStreamGenerator stream = new NativeDenseInstanceStreamGenerator();
		RandomTreeGenerator rtg = new RandomTreeGenerator();
		rtg.prepareForUse();
		Instance inst = rtg.nextInstance();
		stream.baseStreamOption.setCurrentObject(rtg);
		stream.prepareForUse();
		
		NativeDenseInstanceBatch nativeDense = new NativeDenseInstanceBatch(stream.getHeader(), 1024);
		for (int i = 0; i < 1024; ++i)
			nativeDense.addInstance((NativeDenseInstance)stream.nextInstance());
		nativeDense.commit();
		nativeDense.release();
		
		long clock = System.currentTimeMillis();
		for (int l = 0 ;l < 1000; ++l)
		{
			nativeDense = new NativeDenseInstanceBatch(stream.getHeader(), 1024);
			for (int i = 0; i < 1024; ++i)
				nativeDense.addInstance((NativeDenseInstance)stream.nextInstance());
			nativeDense.commit();
			nativeDense.release();
		}
		long end = System.currentTimeMillis();
		double time_msec = end - clock;
	
		//speed = 1000/speed;
		System.out.println("Native dense time " + time_msec);
		
		clock = System.currentTimeMillis();
		for (int l = 0 ;l < 1000; ++l)
		{
			DenseOffHeapBuffer denseOffHeap = new DenseOffHeapBuffer(1024, inst.numAttributes()-1);
			denseOffHeap.begin();
			for (int i = 0; i < 1024; ++i)
			{
				denseOffHeap.set(rtg.nextInstance(), i);
			}
			denseOffHeap.commit();
			denseOffHeap.release();
		}
		end = System.currentTimeMillis();
		time_msec = end - clock;
	
		//speed = 1000/speed;
		System.out.println("Offheap " + time_msec);

		
	}

}

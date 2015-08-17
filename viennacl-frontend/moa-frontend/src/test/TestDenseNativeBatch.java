package test;

import org.junit.Test;
import org.moa.gpu.NativeDenseInstanceStreamGenerator;
import org.moa.gpu.NativeSparseInstanceStreamGenerator;
import org.moa.gpu.bridge.NativeDenseInstance;
import org.moa.gpu.bridge.NativeDenseInstanceBatch;
import org.moa.gpu.bridge.NativeSparseInstance;
import org.moa.gpu.bridge.NativeSparseInstanceBatch;

public class TestDenseNativeBatch {

	@Test
	public void test() {
		System.loadLibrary("moa-frontend-lib");
		{
			NativeDenseInstanceStreamGenerator stream = new NativeDenseInstanceStreamGenerator();
			stream.prepareForUse();
			for (int i = 0 ;i < 10000 ; ++i)
			{
				NativeDenseInstanceBatch batch = new NativeDenseInstanceBatch(stream.getHeader(), 1024);
				while (batch.addInstance( (NativeDenseInstance)stream.nextInstance()))
				{}
				batch.commit();
				batch.clearBatch();
				batch.release();
			}
			
		}
		{
			NativeSparseInstanceStreamGenerator stream = new NativeSparseInstanceStreamGenerator();
			stream.prepareForUse();
			for (int i = 0 ;i < 10000 ; ++i)
			{
				NativeSparseInstanceBatch batch = new NativeSparseInstanceBatch(stream.getHeader(), 1024);
				while (batch.addInstance( (NativeSparseInstance)stream.nextInstance()))
				{}
				batch.commit();
				batch.clearBatch();
				batch.release();
			}
			
		}
		
		
	}

}

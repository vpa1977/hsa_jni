package org.moa.gpu.bridge;

import org.moa.gpu.util.DirectMemory;

import weka.core.DenseInstance;
import weka.core.Instance;

public class DenseOffHeapBuffer {

    private long m_size;
    private long m_step;
    private long m_buffer;
    private long m_rows;
    private long m_class_buffer;
    private long m_weights;

    public DenseOffHeapBuffer(int rows, int numAttributes) {
        m_rows = rows;
        m_size = rows * numAttributes * DirectMemory.DOUBLE_SIZE;
        m_step = numAttributes;
        m_buffer = allocate(m_size);
        m_class_buffer = allocate(rows * DirectMemory.DOUBLE_SIZE);
        m_weights = allocate(rows * DirectMemory.DOUBLE_SIZE);
    }

    public void release() {
        release(m_class_buffer);
        release(m_buffer);
        release(m_weights);
        m_class_buffer = 0;
        m_buffer = 0;
        m_weights = 0;
    }

    protected void finalize() throws Throwable {
        release();
        super.finalize();
    }

    private class DenseInstanceAccess extends DenseInstance {

        public DenseInstanceAccess(Instance inst) {
            super(inst);
        }

        public double[] values() {
            return m_AttValues;
        }
    }

    public void set(Instance inst, int pos) {
        if (pos >= m_rows) {
            throw new ArrayIndexOutOfBoundsException(pos);
        }
        long writeIndex = pos * m_step;

        DenseInstanceAccess ins = new DenseInstanceAccess(inst);
        double[] data = ins.values();
        DirectMemory.writeArray(m_buffer, writeIndex, data); // write instances
        DirectMemory.write(m_buffer + (writeIndex + inst.classIndex()) * DirectMemory.DOUBLE_SIZE, 0); // zero out class attribute
        DirectMemory.write(m_class_buffer + pos * DirectMemory.DOUBLE_SIZE, inst.classValue()); // write class
        DirectMemory.write(m_weights+ pos * DirectMemory.DOUBLE_SIZE, inst.weight() );
    }

    public void read(Instance flyweight, int pos) {
        long writeIndex = pos * m_step * DirectMemory.DOUBLE_SIZE;
        for (int attr = 0; attr < flyweight.numAttributes(); ++attr) {
            double value = readAttr(writeIndex);
            flyweight.setValue(attr, value);
            writeIndex += DirectMemory.DOUBLE_SIZE;
        }
        double classValue = DirectMemory.read(m_class_buffer + pos * DirectMemory.DOUBLE_SIZE);
        flyweight.setClassValue(classValue);
    }

    private double readAttr(long index) {
        return DirectMemory.read(m_buffer + index);
    }

    public native long allocate(long size);

    public native void release(long handle);

    public native void begin();

    public native void commit();

    public long data() {
        return m_buffer;
    }

    public long classes() {
        return m_class_buffer;
    }

}

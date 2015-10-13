/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.moa.gpu;

import org.moa.gpu.bridge.DenseOffHeapBuffer;
import weka.core.Instance;

/**
 *
 * @author john
 */
class SlidingWindow {

    /**
     * class data
     */
    private DenseOffHeapBuffer m_model;
    /**
     * number of instances in the window
     */
    private int m_size;
    /**
     * current row in the window
     */
    private int m_row;
    /**
     * return true if enough instances present
     */
    private boolean m_ready;

    public SlidingWindow(int size, int num_attributes) {
        m_size = size;
        m_ready = false;
        m_row = 0;
        m_model = new DenseOffHeapBuffer(m_size, num_attributes);
    }

    public void update(Instance instance) {
        m_model.set(instance, m_row);
        m_row++;
        if (m_row == m_size) {
            m_ready = true;
            m_row = 0;
        }
    }

    DenseOffHeapBuffer model() {
        return m_model;
    }

    public boolean isReady() {
        return m_ready;
    }

    public void dispose() {
        m_ready = false;
        m_model.release();
        m_model = null;
    }

}

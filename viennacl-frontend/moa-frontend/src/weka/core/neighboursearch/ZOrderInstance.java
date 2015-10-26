/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.core.neighboursearch;

import java.io.Serializable;
import java.math.BigInteger;

import weka.core.Instance;

/**
 *
 * @author john
 */
public class ZOrderInstance implements Comparable<ZOrderInstance>,  Serializable {
    /**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public Instance m_instance;
    public BigInteger m_order;

    ZOrderInstance(BigInteger value, Instance inst) {
       m_order = value;
       m_instance = inst;
    }

    public int compareTo(ZOrderInstance o2) {
        return m_order.compareTo(o2.m_order);
    }
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.java;

import static org.junit.Assert.assertEquals;

import java.io.FileInputStream;
import java.math.BigInteger;
import java.util.ArrayList;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.neighboursearch.ZOrder;
import weka.core.neighboursearch.ZOrderInstance;
import weka.core.neighboursearch.ZOrderSearch;

/**
 *
 * @author john
 */
public class TestZOrder {
    
    public TestZOrder() {
    }
    
    @Before
    public void setUp() {
    }
    
    @After
    public void tearDown() {
    }

    @Test
    public void newTest() {
    	
    }
    // TODO add test methods here.
    // The methods must be annotated with annotation @Test. For example:
    //
   @Test
    public void buildZOrder() throws Throwable
    {
        ArffLoader loader = new ArffLoader();
        loader.setSource(new FileInputStream("F:\\weka_377\\data\\cpu.arff"));
        loader.getStructure();
        Instances dataset = loader.getDataSet();
        ZOrder order = new ZOrder();
        ArrayList<ZOrderInstance> result = order.createZOrder(dataset, dataset);
        for (ZOrderInstance inst : result)
        {
            System.out.println(inst.m_instance_index);
        }
        System.out.println("*");
    }
   
   
   @Test
   public void dumpDistanceList() throws Throwable
   {
       ArffLoader loader = new ArffLoader();
       loader.setSource(new FileInputStream("F:\\weka_377\\data\\cpu.arff"));
       loader.getStructure();
       Instances dataset = loader.getDataSet();
       ZOrderSearch search = new ZOrderSearch();
       search.setNumSearchCurves(1);
       search.setUseReduction(false);
       search.setUseRandomShift(false);
       search.setInstances(dataset);
       search.buildDistanceLists();
       System.out.println("*");
   }

    
    @Test
    public void testInterleaving()  throws Throwable
    {
        ZOrder order = new ZOrder() {
            protected long doubleAsLong(double input)
            {
                return Double.doubleToLongBits(input);
            }
        };
        double one = Double.longBitsToDouble(2);
        double two = Double.longBitsToDouble(2);
        Attribute oneAtt = new Attribute("one"); 
        Attribute twoAtt = new Attribute("two");
        Attribute threeAtt = new Attribute("three");
        ArrayList<Attribute> list = new ArrayList<Attribute>();
        list.add(oneAtt);list.add(twoAtt);list.add(threeAtt);
        
        Instances newDataset = new Instances("dataset", list, 0);
        
        DenseInstance sample = new DenseInstance(3);
        sample.setDataset(newDataset);
        sample.setValue(0, one);
        sample.setValue(1, one);
        sample.setValue(2, two);
        
        //BigInteger result = order.interleave(sample);
        //assertEquals(result, 12);
        
        
        
        
    }
    // public void hello() {}
}

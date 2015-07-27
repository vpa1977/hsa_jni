package org.moa.gpu.util;

/*
 *    EvaluatePeriodicHeldOutTest.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *    @author Ammar Shaker (shaker@mathematik.uni-marburg.de)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */


import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

import moa.classifiers.Classifier;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.StringUtils;
import moa.core.TimingUtils;
import moa.evaluation.ClassificationPerformanceEvaluator;
import moa.evaluation.LearningCurve;
import moa.evaluation.LearningEvaluation;
import moa.options.ClassOption;
import moa.options.FileOption;
import moa.options.FlagOption;
import moa.options.IntOption;
import moa.streams.CachedInstancesStream;
import moa.streams.InstanceStream;
import moa.tasks.MainTask;
import moa.tasks.TaskMonitor;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Task for evaluating a classifier on a stream by periodically testing on a heldout set.
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class EvaluateTrainSpeed extends MainTask {

    @Override
    public String getPurposeString() {
        return "Evaluates a classifier on a stream by periodically testing on a heldout set.";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption learnerOption = new ClassOption("learner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public ClassOption streamOption = new ClassOption("stream", 's',
            "Stream to learn from.", InstanceStream.class,
            "generators.RandomTreeGenerator");

    public ClassOption evaluatorOption = new ClassOption("evaluator", 'e',
            "Classification performance evaluation method.",
            ClassificationPerformanceEvaluator.class,
            "BasicClassificationPerformanceEvaluator");

    public IntOption testSizeOption = new IntOption("testSize", 'n',
            "Number of testing examples.", 1000000, 0, Integer.MAX_VALUE);

    public IntOption trainSizeOption = new IntOption("trainSize", 'i',
            "Number of training examples, <1 = unlimited.", 0, 0,
            Integer.MAX_VALUE);

    public IntOption trainTimeOption = new IntOption("trainTime", 't',
            "Number of training seconds.", 10 * 60 * 60, 0, Integer.MAX_VALUE);

    public IntOption sampleFrequencyOption = new IntOption(
            "sampleFrequency",
            'f',
            "Number of training examples between samples of learning performance.",
            100000, 0, Integer.MAX_VALUE);

    public FileOption dumpFileOption = new FileOption("dumpFile", 'd',
            "File to append intermediate csv results to.", null, "csv", true);

    public FlagOption cacheTestOption = new FlagOption("cacheTest", 'c',
            "Cache test instances in memory.");

    @Override
    protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
        Classifier learner = (Classifier) getPreparedClassOption(this.learnerOption);
        InstanceStream stream = (InstanceStream) getPreparedClassOption(this.streamOption);
        ClassificationPerformanceEvaluator evaluator = (ClassificationPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
        learner.setModelContext(stream.getHeader());
        long instancesProcessed = 0;
        LearningCurve learningCurve = new LearningCurve("evaluation instances");
        File dumpFile = this.dumpFileOption.getFile();
        PrintStream immediateResultStream = null;
        if (dumpFile != null) {
            try {
                if (dumpFile.exists()) {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile, true), true);
                } else {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open immediate result file: " + dumpFile, ex);
            }
        }
        boolean firstDump = true;
        InstanceStream testStream = null;
        int testSize = this.testSizeOption.getValue();
        if (this.cacheTestOption.isSet()) {
            monitor.setCurrentActivity("Caching test examples...", -1.0);
            Instances testInstances = new Instances(stream.getHeader(),
                    this.testSizeOption.getValue());
            while (testInstances.numInstances() < testSize) {
                testInstances.add(stream.nextInstance());
                if (testInstances.numInstances()
                        % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
                    if (monitor.taskShouldAbort()) {
                        return null;
                    }
                    monitor.setCurrentActivityFractionComplete((double) testInstances.numInstances()
                            / (double) (this.testSizeOption.getValue()));
                }
            }
            testStream = new CachedInstancesStream(testInstances);
        } else {
            //testStream = (InstanceStream) stream.copy();
            testStream = stream;
            /*monitor.setCurrentActivity("Skipping test examples...", -1.0);
            for (int i = 0; i < testSize; i++) {
            stream.nextInstance();
            }*/
        }
        instancesProcessed = 0;
        TimingUtils.enablePreciseTiming();
        double totalTrainTime = 0.0;
        while ((this.trainSizeOption.getValue() < 1
                || instancesProcessed < this.trainSizeOption.getValue())
                && stream.hasMoreInstances() == true) {
            monitor.setCurrentActivityDescription("Training...");
            
            learner.trainOnInstance(stream.nextInstance()); // force lazy init
            
            long instancesTarget = instancesProcessed
                    + this.sampleFrequencyOption.getValue();
            long trainStartTime = System.nanoTime();
            while (instancesProcessed < instancesTarget && stream.hasMoreInstances() == true) {
                learner.trainOnInstance(stream.nextInstance());
                instancesProcessed++;
                if (instancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
                    if (monitor.taskShouldAbort()) {
                        return null;
                    }
                    monitor.setCurrentActivityFractionComplete((double) (instancesProcessed)
                            / (double) (this.trainSizeOption.getValue()));
                }
            }
            try {learner.getVotesForInstance(null); } catch (Exception e) {}
            
            double lastTrainTime = ((double)(System.nanoTime() - trainStartTime)) / 1000000000;
            totalTrainTime += lastTrainTime;
            if (totalTrainTime > this.trainTimeOption.getValue() ) {
                break;
            }
            List<Measurement> measurements = new ArrayList<Measurement>();
            measurements.add(new Measurement("evaluation instances",            		
                    instancesProcessed));
            measurements.add(new Measurement("total train speed",
                    instancesProcessed / totalTrainTime));
            measurements.add(new Measurement("last train speed",
                    this.sampleFrequencyOption.getValue() / lastTrainTime));
            learningCurve.insertEntry(new LearningEvaluation(measurements.toArray(new Measurement[measurements.size()])));
            if (immediateResultStream != null) {
                if (firstDump) {
                    immediateResultStream.println(learningCurve.headerToString());
                    firstDump = false;
                }
                immediateResultStream.println(learningCurve.entryToString(learningCurve.numEntries() - 1));
                immediateResultStream.flush();
            }
            if (monitor.resultPreviewRequested()) {
                monitor.setLatestResultPreview(learningCurve.copy());
            }

        }
        if (immediateResultStream != null) {
            immediateResultStream.close();
        }
        return learningCurve;
    }

    @Override
    public Class<?> getTaskResultType() {
        return LearningCurve.class;
    }
}

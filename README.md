## Spark ML Data Segmentation

This project is [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning) implementation on data using [Spark MLLIB](https://spark.apache.org/docs/latest/mllib-guide.html). 

It provides three Machine Learning algorithms implementation [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering), [Random Forest](https://en.wikipedia.org/wiki/Random_forest) and Dimentionality Reduction mechanism [PCA] (https://en.wikipedia.org/wiki/Principal_component_analysis)


## Prerequisites

Please ensure the availability of the following software on the machine

[Git](https://git-scm.com/downloads)

[Apache Maven 3.x](https://maven.apache.org/index.html)

[Java 1.8.x ](https://docs.oracle.com/javase/8/docs/technotes/guides/install/install_overview.html)

[Scala 2.10.4](http://www.scala-lang.org/download/2.10.4.html)

Please also ensure your [SPARK_HOME](https://spark.apache.org/docs/latest/configuration.html#environment-variables) is set before you use the algorithms.

### Compilation Steps

 * Clone the repo

`$ git clone https://github.com/behzadaltaf/Spark-ML.git
`
 
* Run MAVEN to build the jar

`$ mvn package
`

to build the jar and skip tests please use the command below

`$ mvn package -Dmaven.test.skip=true
`

The above step would create an `uber` jar in the `target` folder which can be run with a `SPARK` installation

### Running on a Cluster 

Please refer documentation on how to run `Spark` application on a cluster [here](docs/Stand-alone-cluster.md)

The `jar` contains the following

## 1. Random Forest Classification

- [Spark Random Forest](https://spark.apache.org/docs/latest/mllib-ensembles.html#random-forests)

### 1 a. RandomForestClassification: 

A random forest classification run on Subject Segmentation data. This program can generate and save the generated model for future use.

The ***input file*** to generated the Random Forest model should be in the following **CSV** format


| SubjectID,        | ...feature vectors...,           | SegmentID  |
| ------------- |:-------------:| -----:|
| 123  ,   | ..., | 5 |

In data cleansing step the **header**, the **first** and the **last** columns are dropped, 
The feature vectors can be expanded and collapsed as desired but a minimum one is required.

    Usage: RandomForestClassification [options] inputFile outputFile  outputModelLocation

    --numClasses <value>
        number of classes, default: 6
    --numTrees <value>
        number Of trees , default: 3
    --maxDepth <value>
        maximum depth of trees , default: 3
    --maxBins <value>
        maximum number of bins, default: 4
    --training <value>
        training slice, default: 0.7
    --test <value>
        test slice, default: 0.3
    --featureSubsetStrategy <value>
        feature Subset Strategy, default: auto
    --impurity <value>
        impurity, default: gini
    --categoricalFeaturesInfo <value>
        categoricalFeaturesInfo k1=v1,k2=v2..., default: Map()
    --master <value>
        master url, default: local
    --appName <value>
        application name, default: Subject Segmentation using Random Forest Classification
    inputFile
        input file the Subjectdataset, required
    outputFile
        output file for the run, required
    outputModelLocation
        output location for the model to be saved, required


### Submit Job
```
 spark-submit --class com.behzad.cs.randomForest.RandomForestClassification target/SubjectSegmentation-1.0.0.jar \
  --numClasses 6 \
  --numTrees 30 \
  --maxDepth 30 \
  --maxBins 20 \
  --training 0.8 \
  --test 0.2 \
  --featureSubsetStrategy auto \
  --impurity gini \
  --categoricalFeaturesInfo 1=2,4=3 \
  --appName Subject_Segmentation_RF \
  --master spark://127.0.0.1/master \ 
  data/InputData.csv \
  data/.out \
  model/RFModel
```

----------

### 1 b. RandomForestAnalysis: 

The ***input file*** on which the trained model has to be run should be in the following ***CSV*** format


| SubjectID,        | ...feature vectors...           | 
| ------------- |:-------------:| 
| 123  ,   | ...|

In data cleansing step the **first** column and the **header** are dropped, the feature vectors can be expanded and collapsed as desired but a minimum one is required.
The feature vectors should be mapped to the **same length** when the Random Forest Model was generated and saved.

    Usage: RandomForestAnalysis [options] inputFile outputFile modelLocation

    --master <value>
        master url, default: local
    --appName <value>
        application name, default: Subject Segmentation Analysis using Random Forest Classification
    inputFile
        input file the Subjectdataset, required
    outputFile
        output file for the run, required
    modelLocation
        location for the model to be loaded from, required



### Submit Job
```
 spark-submit --class com.behzad.cs.randomForest.RandomForestAnalysis target/SubjectSegmentation-1.0.0.jar \
  --appName Subject_Segmentation_RF \
  --master spark://127.0.0.1/master \ 
  data/InputData.csv \
  data/.out \
  model/RFModel
```

----------


### 1 c. RandomForestPrediction: 

A random forest classification run on SubjectSegmentation data. This program can generate and save the generated model for future use.

The ***input files*** to generated the Random Forest model and to run predictions should be in the following **CSV** format

| SubjectID,        | ...feature vectors...,           | SegmentID  |
| ------------- |:-------------:| -----:|
| 123  ,   | ..., | 5 |

In data cleansing step the **header**, the **first** and the **last** columns are dropped, 
The feature vectors can be expanded and collapsed as desired but a minimum one is required.

    Usage: RandomForestPrediction [options] inputFileForModelGeneration inputFileForRFAnalysis  outputFile

    --numClasses <value>
        number of classes, default: 6
    --numTrees <value>
        number Of trees , default: 3
    --maxDepth <value>
        maximum depth of trees , default: 3
    --maxBins <value>
        maximum number of bins, default: 4
    --training <value>
        training slice, default: 0.7
    --test <value>
        test slice, default: 0.3
    --featureSubsetStrategy <value>
        feature Subset Strategy, default: auto
    --impurity <value>
        impurity, default: gini
    --categoricalFeaturesInfo <value>
        categoricalFeaturesInfo k1=v1,k2=v2..., default: Map()
    --master <value>
        master url, default: local
    --appName <value>
        application name, default: SubjectSegmentation using Random Forest Classification
    inputFileForModelGeneration
        input file the model generation, required
    inputFileForRFAnalysis
        input file the analysis, required
    outputFile
        output file for the run, required


### Submit Job
```
 spark-submit --class com.behzad.cs.randomForest.RandomForestPrediction target/SubjectSegmentation-1.0.0.jar \
  --numClasses 6 \
  --numTrees 30 \
  --maxDepth 30 \
  --maxBins 20 \
  --training 0.8 \
  --test 0.2 \
  --featureSubsetStrategy auto \
  --impurity gini \
  --categoricalFeaturesInfo 1=2,4=3 \
  --appName Subject_Segmentation_RF \
  --master spark://127.0.0.1/master \ 
  data/InputDataForModelGeneration.csv \
  data/InputDataForRFAnalysis.csv \
  data/RF.out
```

----------

## 2. k-means Clustering

[Spark k-means](http://spark.apache.org/docs/latest/mllib-clustering.html#k-means)

### 2 a. KMeansClustering

A k-means clustering run on SubjectSegmentation data. This program can generate and save the generated model for future use.

The ***input file*** to generated the Random Forest model should be in the following **CSV** format

| SubjectID,        | ...feature vectors...,           | SegmentID  |
| ------------- |:-------------:| -----:|
| 123  ,   | ..., | 5 |

In data cleansing step the **header**, the **first** and the **last** columns are dropped, 
The feature vectors can be expanded and collapsed as desired but a minimum one is required.

*KMeansClustering:* a k-means clustering run on SubjectSegmentation data.

    Usage: KMeansClustering [options] inputFile outputFile outputModelLocation

    --k <value>
        min number of k-means partitions, default: 6
    --runs <value>
        number Of Runs , default: 10
    --epsilon <value>
        error margin epsilon, default: 1.0E-6
    --initializationMode <value>
        initialization mode (Random,Parallel), default: Parallel
    --seed <value>
        seed, default: Randomly generated by scala.util.Random.nextLong()
    --master <value>
        master url, default: local
    --appName <value>
        application name, default: SubjectSegmentation using k-means
    inputFile
        input file the Subjectdataset, required
    outputFile
        output file for the run, required
    outputModelLocation
        output location for the model to be saved, required


### Submit Job
```
 spark-submit --class com.behzad.cs.kmeans.KMeansClustering target/SubjectSegmentation-1.0.0.jar \
  --k 5 \
  --runs 8 \
  --epsilon 1.0e-4 \
  --initializationMode Random \  
  --seed 1000 \
  --appName Subject_Segmentation_K-Means \
  --master spark://127.0.0.1/master \
  data/InputData.csv \
  data/Kmeans.out \
  data/KMeansModel
```

### 2 b. KMeansAnalysis: 

The ***input file*** on which the trained model has to be run should be in the following ***CSV*** format

| SubjectID,        | ...feature vectors...           |
| ------------- |:-------------:| 
| 123  ,   | ... | 

In data cleansing step the **first** column and the **header** are dropped, the feature vectors can be expanded and collapsed as desired but a minimum one is required.
The feature vectors should be mapped to the **same length** when the k-means Model was generated and saved.

    Usage: KMeansAnalysis [options] inputFile outputFile modelLocation

    --master <value>
        master url, default: local
    --appName <value>
        application name, default: SubjectSegmentation Analysis using k-Means
    inputFile
        input file the Subjectdataset, required
    outputFile
        output file for the run, required
    modelLocation
        location for the model to be loaded from, required


### Submit Job
```
 spark-submit --class com.behzad.cs.kmeans.KMeansAnalysis target/SubjectSegmentation-1.0.0.jar \
  --appName Subject_Segmentation_RF \
  --master spark://127.0.0.1/master \ 
  data/InputData.csv \
  data/.out \
  model/KMeansModel
```

### 2 c. KMeansPrediction: 

The ***input file*** on which the trained model has to be run should be in the following ***CSV*** format

| SubjectID,        | ...feature vectors...,           | SegmentID  |
| ------------- |:-------------:| -----:|
| 123  ,   | ..., | 5 |

In data cleansing step the **header**, the **first** and the **last** columns are dropped, the feature vectors can be expanded and collapsed as desired but a minimum one is required.

    Usage: KMeansPrediction [options] inputFile outputFile 

    --k <value>
        min number of k-means partitions, default: 6
    --runs <value>
        number Of Runs , default: 10
    --epsilon <value>
        error margin epsilon, default: 1.0E-6
    --initializationMode <value>
        initialization mode (Random,Parallel), default: Parallel
    --seed <value>
        seed, default: Randomly generated by scala.util.Random.nextLong()
    --master <value>
        master url, default: local
    --appName <value>
        application name, default: SubjectSegmentation using k-means
    inputFile
        input file the Subjectdataset, required
    outputFile
        output file for the run, required


### Submit Job
```
 spark-submit --class com.behzad.cs.kmeans.KMeansAnalysis target/SubjectSegmentation-1.0.0.jar \
  --k 5 \
  --runs 8 \
  --epsilon 1.0e-4 \
  --initializationMode Random \
  --seed 1000 \
  --appName Subject_Segmentation_K-Means \
  --master spark://127.0.0.1/master \
  data/InputData.csv \
  data/Kmeans.out
```


----------

## 3. Principal Component Analysis

[Spark PCA](https://spark.apache.org/docs/1.4.1/mllib-dimensionality-reduction.html)

A Principal Component Analysis on SubjectSegmentation data.

The ***input file*** to reduce the dimentionality should be in the following **CSV** format

| SubjectID,        | ...feature vectors...,           | SegmentID  |
| ------------- |:-------------:| -----:|
| 123  ,   | ..., | 5 |

In data cleansing step the **first** column is dropped, 
The feature vectors can be expanded and collapsed as desired but a minimum one is required.

The ***output file*** of PCA is in the following format and can be used as the ***input file*** for both 

+ [Random Forest - RandomForestClassification](#1-random-forest-classification) and 

+ [k-means -KMeansClustering](#2-k-means-clustering)

| NIL- SubjectID,        | ...condensed feature vectors...,           | SegmentID  |
| ------------- |:-------------:| -----:|
| NIL  ,   | ..., | 5 |

### PCA

A Principal Component Analysis on Subject Segmentation data.

    Usage: PCA [options] inputFile outputFile

    --k <value>
         min number of principal components, default: 10
    --master <value>
        master url, default: local
    --appName <value>
        application name, default: SubjectSegmentation using k-means
    inputFile
        input file the Subjectdataset, required
    outputFile
        output file for the run, required


### Submit Job
```
 spark-submit --class com.behzad.cs.pca.PCA target/SubjectSegmentation-1.0.0.jar \
  --k 8 \
  --appName Subject_Segmentation_PCA\
  --master spark://127.0.0.1/master \
  data/InputData.csv \
  data/PCA.out 
```

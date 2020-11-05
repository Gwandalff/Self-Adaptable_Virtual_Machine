# Self-Adaptable Virtual Machines

This project aim to create a new implementation pattern for interpreters.
The main objective is to define an architectural solution to resolve the problem of dynamic adaptation in language interpreters.
To do that, we plan to use a modular approach defining an interface in the language specification and multiple modules of dynamic adaptation.

## Requirements

To make this project work correctly you need to clone [the ALE Language](https://github.com/gwendal-jouneaux/ale-lang) and do a `mvn install` at the root of the project. With this you'll get the appropriate dependancies in your .m2 folder.

This project need to be compiled using Java 8.

# Sobel filter output quality

In order to validate our approach, we have tested the adaptive Sobel filter on two images, a small one with easily detectable edges, and a big image with less visible edges (the filter is less precise on the ground).

## Output images when applying the sobel filter on a small image (640x480)

![Input image](https://user-images.githubusercontent.com/68585341/98281157-a9db1a80-1f9c-11eb-90c1-cec3294fe5c9.png)

| Interpolation | Aproximated version          |  Reference version        |
| :------------ | :--------------------------: | :-----------------------: | 
| 1/2 pixels    | ![png](https://user-images.githubusercontent.com/68585341/98281139-a6479380-1f9c-11eb-8b28-fef58af37ed0.png) | ![png](https://user-images.githubusercontent.com/68585341/98281168-ad6ea180-1f9c-11eb-966c-3e9179f26fe8.png) |
| 3/4 pixels    | ![png](https://user-images.githubusercontent.com/68585341/98281151-a8115700-1f9c-11eb-8e31-3e4a21e45008.png) | ![png](https://user-images.githubusercontent.com/68585341/98281168-ad6ea180-1f9c-11eb-966c-3e9179f26fe8.png) |
| 7/8 pixels    | ![png](https://user-images.githubusercontent.com/68585341/98281154-a9428400-1f9c-11eb-9e0e-f4fd72c2d0ed.png) | ![png](https://user-images.githubusercontent.com/68585341/98281168-ad6ea180-1f9c-11eb-966c-3e9179f26fe8.png) |

## Output images when applying the sobel filter on a bigger image (1920x1080)

![Big input image](https://user-images.githubusercontent.com/68585341/98281159-aa73b100-1f9c-11eb-8272-559941f1f55b.png)

| Interpolation | Aproximated version              |  Reference version            |
| :------------ | :------------------------------: | :---------------------------: |
| 1/2 pixels    | ![png](https://user-images.githubusercontent.com/68585341/98281145-a6e02a00-1f9c-11eb-8e8a-e7de48818865.png) | ![png](https://user-images.githubusercontent.com/68585341/98281171-ae073800-1f9c-11eb-9f2a-9e0e5f98475c.png) |
| 3/4 pixels    | ![png](https://user-images.githubusercontent.com/68585341/98281152-a8a9ed80-1f9c-11eb-868f-94ce85ad8fea.png) | ![png](https://user-images.githubusercontent.com/68585341/98281171-ae073800-1f9c-11eb-9f2a-9e0e5f98475c.png) |
| 7/8 pixels    | ![png](https://user-images.githubusercontent.com/68585341/98281155-a9428400-1f9c-11eb-9e0e-2656372c5920.png) | ![png](https://user-images.githubusercontent.com/68585341/98281171-ae073800-1f9c-11eb-9f2a-9e0e5f98475c.png) |

## Discussion

One of the conclusion on the quality of the produced images, is that the number of interpolated pixels should also depend on the size of the images. In fact, while the high definition image seems to be lightly impacted by the maximum interpolation, the output of the small image seems to be at the limit of being usable in image processing algorithms. However, the result on smaller interpolation ratio show that the impact on the quality of the resulting output is totally acceptable. Moreover, the interpolation ratio of three-quarter of the pixels show the interest of our approach by showing speedup comparing to the reference imprementation of the Virtual Machine while giving as output a degraded but usable edge detection.

# Evaluation Notebook

The following content is an export of the statistic analysis done for the evaluation of the Self-Adaptable implementation of MiniJava for image processing. This snapshot of the results was made the 2020-10-17.

## Self-Adaptable MiniJava for image processing

This notebook aim at providing the analysis process of the Self-Adaptable implementation of MiniJava performances for image processing with an Approximate Loop Unrolling adaptation. This implementation is compared to a reference implementation based on the interpreter design pattern.

### Imports


```python
from os import listdir
from os.path import isfile, join
import math
import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
```

### Data

Build the datastructure containing the results of JMH benchmarks. All the benchmarks were executed on Debian 9 with the **Krun** kernel, 15Go of RAM, an Intel(R) Xeon(R) W-2104 CPU (Quad Core - 3.20GHz), and run on the GraalVM CE JVM version 20.2.0.


```python
root_dir = './results/'

results=[]
for f in listdir(root_dir):
    path = join(root_dir, f)
    if isfile(path):
        fo = open(path)
        jso = json.loads(fo.read())
        splt = f.split('_')
        vm = splt[0]
        stress = splt[1][6:]
        measureNb = splt[2]
        for (idx, time) in enumerate(jso[0]['primaryMetric']['rawData'][0]):
                results.append({
                        'vm': vm,
                        'stress': int(stress),
                        'try': measureNb,
                        'time': time,
                        'idx': idx
                })

df = pd.DataFrame(results)
df.describe(include='all')
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vm</th>
      <th>stress</th>
      <th>try</th>
      <th>time</th>
      <th>idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>900</td>
      <td>900.000000</td>
      <td>900</td>
      <td>900.000000</td>
      <td>900.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>NaN</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>reference</td>
      <td>NaN</td>
      <td>tryI</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>450</td>
      <td>NaN</td>
      <td>300</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>50.000000</td>
      <td>NaN</td>
      <td>13.373716</td>
      <td>14.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>35.374997</td>
      <td>NaN</td>
      <td>4.602772</td>
      <td>8.660254</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>8.198310</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>25.000000</td>
      <td>NaN</td>
      <td>10.880730</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>50.000000</td>
      <td>NaN</td>
      <td>12.630456</td>
      <td>14.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>75.000000</td>
      <td>NaN</td>
      <td>13.175090</td>
      <td>22.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>100.000000</td>
      <td>NaN</td>
      <td>25.830481</td>
      <td>29.000000</td>
    </tr>
  </tbody>
</table>



### Statistic analysis

#### Basic metrics

##### Variance


```python
variances = pd.DataFrame(df.groupby(["vm", "stress"]).time.var())
variances["vm"] = ""
variances["Stress"] = ""
variances.vm = variances.index.get_level_values(0)
variances.Stress = variances.index.get_level_values(1)
variances.index = range(10)
variances.rename(columns={"time":"Variances"}, inplace=True)
```

##### Standard deviation


```python
deviations = pd.DataFrame(df.groupby(["vm", "stress"]).time.std())
deviations["vm"] = ""
deviations["Stress"] = ""
deviations.vm = deviations.index.get_level_values(0)
deviations.Stress = deviations.index.get_level_values(1)
deviations.index = range(10)
deviations.rename(columns={"time":"Deviation"}, inplace=True)
```

##### Average time


```python
means = pd.DataFrame(df.groupby(["vm", "stress"]).time.mean())
means["vm"] = ""
means["Stress"] = ""
means.vm = means.index.get_level_values(0)
means.Stress = means.index.get_level_values(1)
means.index = range(10)
means.rename(columns={"time":"Mean"}, inplace=True)
```

##### Median value


```python
medians = pd.DataFrame(df.groupby(["vm", "stress"]).time.median())
medians["vm"] = ""
medians["Stress"] = ""
medians.vm = medians.index.get_level_values(0)
medians.Stress = medians.index.get_level_values(1)
medians.index = range(10)
medians.rename(columns={"time":"Median"}, inplace=True)
```

##### Concatenate all metrics


```python
stats = pd.concat([variances, means, medians, deviations], axis=1)
stats["Variance"] = 0
stats.Variance = stats.Variances
del stats["Variances"]
stats = stats.loc[:,~stats.columns.duplicated()]
stats = stats.sort_values(['Stress', 'vm'])
stats.index = range(10)
stats
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vm</th>
      <th>Stress</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Deviation</th>
      <th>Variance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>adaptive</td>
      <td>0</td>
      <td>25.574801</td>
      <td>25.538089</td>
      <td>0.163591</td>
      <td>0.026762</td>
    </tr>
    <tr>
      <th>1</th>
      <td>reference</td>
      <td>0</td>
      <td>12.605884</td>
      <td>12.599488</td>
      <td>0.060774</td>
      <td>0.003693</td>
    </tr>
    <tr>
      <th>2</th>
      <td>adaptive</td>
      <td>25</td>
      <td>16.297958</td>
      <td>16.300900</td>
      <td>0.040881</td>
      <td>0.001671</td>
    </tr>
    <tr>
      <th>3</th>
      <td>reference</td>
      <td>25</td>
      <td>12.598613</td>
      <td>12.594642</td>
      <td>0.033373</td>
      <td>0.001114</td>
    </tr>
    <tr>
      <th>4</th>
      <td>adaptive</td>
      <td>50</td>
      <td>10.883349</td>
      <td>10.880684</td>
      <td>0.035120</td>
      <td>0.001233</td>
    </tr>
    <tr>
      <th>5</th>
      <td>reference</td>
      <td>50</td>
      <td>12.667502</td>
      <td>12.667936</td>
      <td>0.032941</td>
      <td>0.001085</td>
    </tr>
    <tr>
      <th>6</th>
      <td>adaptive</td>
      <td>75</td>
      <td>8.454307</td>
      <td>8.292755</td>
      <td>0.343647</td>
      <td>0.118093</td>
    </tr>
    <tr>
      <th>7</th>
      <td>reference</td>
      <td>75</td>
      <td>12.674002</td>
      <td>12.673094</td>
      <td>0.053748</td>
      <td>0.002889</td>
    </tr>
    <tr>
      <th>8</th>
      <td>adaptive</td>
      <td>100</td>
      <td>8.758603</td>
      <td>8.687938</td>
      <td>0.318147</td>
      <td>0.101217</td>
    </tr>
    <tr>
      <th>9</th>
      <td>reference</td>
      <td>100</td>
      <td>13.222144</td>
      <td>13.176022</td>
      <td>0.210273</td>
      <td>0.044215</td>
    </tr>
  </tbody>
</table>

#### Functions

compute_stats : print estimation of the mean and confidence interval of the data


```python
def compute_stats(values):
    mean = np.mean(values)
    lo, hi = st.t.interval(0.95, df=(len(values) - 1), loc=mean, scale=st.sem(values))
    print("Estimated mean : " + str(mean))
    print("Confidence interval : [" + str(lo) + ", " + str(hi) + "] delta = " + str(mean - lo))


```

#### Group data by VM and Stress level


```python
adaptive = df[df.vm == "adaptive"]
reference = df[df.vm == "reference"]

adaptive_stress0   = adaptive[adaptive.stress == 0]
adaptive_stress25  = adaptive[adaptive.stress == 25]
adaptive_stress50  = adaptive[adaptive.stress == 50]
adaptive_stress75  = adaptive[adaptive.stress == 75]
adaptive_stress100 = adaptive[adaptive.stress == 100]

reference_stress0   = reference[reference.stress == 0]
reference_stress25  = reference[reference.stress == 25]
reference_stress50  = reference[reference.stress == 50]
reference_stress75  = reference[reference.stress == 75]
reference_stress100 = reference[reference.stress == 100]
```

#### Get p-values


```python
t_stress0,   p_stress0   = st.ttest_ind(a=reference_stress0.time.tolist(),  b=adaptive_stress0.time.tolist(),  equal_var=False)
t_stress25,  p_stress25  = st.ttest_ind(a=reference_stress25.time.tolist(), b=adaptive_stress25.time.tolist(), equal_var=False)
t_stress50,  p_stress50  = st.ttest_ind(a=reference_stress50.time.tolist(), b=adaptive_stress50.time.tolist(), equal_var=False)
t_stress75,  p_stress75  = st.ttest_ind(a=reference_stress75.time.tolist(), b=adaptive_stress75.time.tolist(), equal_var=False)
t_stress100, p_stress100 = st.ttest_ind(a=reference_stress100.time.tolist(),b=adaptive_stress100.time.tolist(),equal_var=False)
pvalues = [p_stress0, p_stress25, p_stress50, p_stress75, p_stress100]
```

#### Compute Effect Size (Cohen's d)


```python
effectSize = stats.copy()
effectSize["P Values"] = 0
effectSize["Effect Size"] = 0
effectSize["Absolute Effect Size"] = 0
for i in range(10):
  if effectSize.loc[i,"vm"] == "adaptive":
    SDpooled = math.sqrt((89*effectSize.loc[i+1,"Variance"]+89*effectSize.loc[i,"Variance"])/178)
    
    meanDiff = effectSize.loc[i+1,"Mean"]-effectSize.loc[i,"Mean"]
    effectSize.loc[i,"Effect Size"] = meanDiff / SDpooled
    effectSize.loc[i+1,"Effect Size"] = 0
    effectSize.loc[i,"Absolute Effect Size"] = abs(meanDiff / SDpooled)
    effectSize.loc[i+1,"Absolute Effect Size"] = 0

del effectSize["Mean"]
del effectSize["Median"]
del effectSize["Variance"]
del effectSize["Deviation"]
effectSize = effectSize.groupby(["Stress"]).sum()
effectSize["P Values"] = pvalues
effectSize["Stress"] = ""
effectSize.Stress = effectSize.index.get_level_values(0)
effectSize.index = range(5)
```

#### Summary of the metrics


```python
print("Stats for adaptive VM with stress level at 0%")  
compute_stats(adaptive_stress0.time.tolist())

print("\nStats for adaptive VM with stress level at 25%")  
compute_stats(adaptive_stress25.time.tolist())

print("\nStats for adaptive VM with stress level at 50%")  
compute_stats(adaptive_stress50.time.tolist())

print("\nStats for adaptive VM with stress level at 75%")  
compute_stats(adaptive_stress75.time.tolist())

print("\nStats for adaptive VM with stress level at 100%")  
compute_stats(adaptive_stress100.time.tolist())

print("\n\n\nStats for reference VM with stress level at 0%")  
compute_stats(reference_stress0.time.tolist())

print("\nStats for reference VM with stress level at 25%")  
compute_stats(reference_stress25.time.tolist())

print("\nStats for reference VM with stress level at 50%")  
compute_stats(reference_stress50.time.tolist())

print("\nStats for reference VM with stress level at 75%")  
compute_stats(reference_stress75.time.tolist())

print("\nStats for reference VM with stress level at 100%")  
compute_stats(reference_stress100.time.tolist())

effectSize
```
```text
Stats for adaptive VM with stress level at 0%
Estimated mean : 25.57480081015556
Confidence interval : [25.540537433595826, 25.60906418671529] delta = 0.034263376559731995

Stats for adaptive VM with stress level at 25%
Estimated mean : 16.297958178133335
Confidence interval : [16.28939578405871, 16.306520572207962] delta = 0.008562394074626667

Stats for adaptive VM with stress level at 50%
Estimated mean : 10.883349342033332
Confidence interval : [10.875993569659844, 10.89070511440682] delta = 0.0073557723734882785

Stats for adaptive VM with stress level at 75%
Estimated mean : 8.454307300866665
Confidence interval : [8.382331807749136, 8.526282793984194] delta = 0.07197549311752915

Stats for adaptive VM with stress level at 100%
Estimated mean : 8.758603195688888
Confidence interval : [8.691968679732373, 8.825237711645403] delta = 0.06663451595651537



Stats for reference VM with stress level at 0%
Estimated mean : 12.605883964122222
Confidence interval : [12.59315511208231, 12.618612816162134] delta = 0.012728852039911587

Stats for reference VM with stress level at 25%
Estimated mean : 12.598612847099998
Confidence interval : [12.591622980866909, 12.605602713333088] delta = 0.006989866233089614

Stats for reference VM with stress level at 50%
Estimated mean : 12.667501712433333
Confidence interval : [12.6606022526539, 12.674401172212766] delta = 0.006899459779432959

Stats for reference VM with stress level at 75%
Estimated mean : 12.67400237858889
Confidence interval : [12.662745132295244, 12.685259624882535] delta = 0.01125724629364555

Stats for reference VM with stress level at 100%
Estimated mean : 13.222143579022223
Confidence interval : [13.178102749755944, 13.266184408288503] delta = 0.0440408292662795
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>P Values</th>
      <th>Effect Size</th>
      <th>Absolute Effect Size</th>
      <th>Stress</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.080518e-208</td>
      <td>-105.096197</td>
      <td>105.096197</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.968148e-294</td>
      <td>-99.134433</td>
      <td>99.134433</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.065232e-254</td>
      <td>52.400715</td>
      <td>52.400715</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.139275e-102</td>
      <td>17.156765</td>
      <td>17.156765</td>
      <td>75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.095489e-149</td>
      <td>16.552534</td>
      <td>16.552534</td>
      <td>100</td>
    </tr>
  </tbody>
</table>



### Plots

#### Effect size depending on the stress

Here we try to evaluate the impact of the stress put on the system on the reliability of our benchmarks


```python
sns.catplot(x="Stress", y="Absolute Effect Size", kind="bar", data=effectSize, legend_out=True);
```

![png](https://user-images.githubusercontent.com/68585341/98281161-aba4de00-1f9c-11eb-9a6e-609be6d68ac2.png)


As we can see, the effect size decrease when the stress rise. However, the fact that there is no big difference between 75% and 100% can alarm us on the fact that our approximate loop unrolling adaptive module can be the root cause of the effect size decrease. Yet our data can not show the difference between both due to the strong correlation of those variables in our experiment.

#### Summary of the benchmarks


```python
sns.lineplot(data=df, x="idx", y="time", hue="vm", style="stress");
```


![png](https://user-images.githubusercontent.com/68585341/98281163-ac3d7480-1f9c-11eb-9785-0d7d2aee2a63.png)



```python
grid = sns.FacetGrid(df, col="stress", hue="vm", sharey=True)
grid.map(sns.lineplot, "idx", "time");
```


![png](https://user-images.githubusercontent.com/68585341/98281164-acd60b00-1f9c-11eb-84c1-32d78f972eb1.png)



```python
sns.catplot(x="Stress", y="Mean", hue="vm", kind="bar", data=means, sharex=False, legend_out=True);
```


![png](https://user-images.githubusercontent.com/68585341/98281166-acd60b00-1f9c-11eb-9ae7-ced8adaca3be.png)


### Discussion

As the different plot shows, our approach has an important overhead when no adaptations can be done. We assume that it is due to the internal decision model that is called at every MAPE-K loop iteration, hence, when no module can provide optimizations for this concern, the system suffer of performance issues. However, the plots also shows that when the adaptations kick-in, the adaptive version can perform better than the original implementation (which is designed using an interpreter design pattern which is the best in terms of performances). This highlight the challenge behind Self-Adaptable Virtual Machines which is the implementation of an efficient generic feedback loop.

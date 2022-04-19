# DepthImage_DBSCAN
The rrf file obtained TOF is converted into 3D point cloud coordinates and saved into csv file. The object distance is calculated by DBSCAN method.

Implementation of DBSCAN Algorithm in Python.

<h2>Input:</h2>

It takes two inputs. First one is the .csv file which contains the data (no headers). In 'main.py' change line 12 to:

<i>DATA = '/path/to/data/file.csv'</i>

And the second is the config file which contains few parameters necessary for the algorithm. More details inside 'config' file. You can change the 'config' file as per your requirement.

UPDATE: July 13, 2017 - The code has been updated to support 3D points. Although technically, it can be used to perform multi-demensional clustering (might need to tweak the code more) - it is the visualization part will not work as expected.

<h2>Few Snapshots</h2>

<h3>3-D Clustering in action</h3>
<img src='https://github.com/Standdrinkmilk/DepthImage_DBSCAN/blob/master/img/figure_3D.png'>
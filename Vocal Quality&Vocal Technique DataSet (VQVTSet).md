# Vocal Quality&Vocal Technique DataSet (VQVTSet)

**## Introduction**

The motivation of designing VQVTSet is to better describe the singing voice using the paralinguistic singing attributes that are popularized in the vocal pedagogy. It can be used for performing the paralinguistic singing attribute recognition and analysis.

There are nearly 4000 vocal segments, totaling 10.7 hours, in the VQVTSet. The amount of vocal segments corresponding to the aria is shown in below. 

![]{song_num.jpg}

For each vocal segment, there are seven paralinguistic singing attributes should be subjectively labeled with different classes. The seven paralinguistic singing attribtues selected are head resonance, chest resonance, open throat, roughness, vibrato, front placement singing, back placement singing. The below figure shows the amounts of vocal segments of each class of each paralinguistic singing attribute.

![]{segnum2class.jpg}

To help better understand, we select the example pair for each paralinguistic singing attribute. However, note that the example pair is just a specific case for corresponding paralinguistic singing attributes since these attributes can not be well-defined.

​    

​    Tables and links ![]{examples.wav}



**## Data preparation** 

Please contact the corresponding author for getting the csv that included labels and YouTube links. Firstly, you should download the data by YouTube links. Furthermore, you need to further process the data, removing accompaniment, using the 2 stems separation model in [spleeter]{htpps://github.com/deezer/spleeter}. Specifically, 



\```

\# install spleeter 

详细介绍 

\# performing music 

安装spleeter 

运行代码

\```





**### Feature set**



\- Deployment

  \- FSSVM

​    \- 安装什么

​    \- 代码在哪里

​    \- 怎么跑起来

​    \- 怎么看结果

  \- E2E

​    \- 代码在哪里

​    \- 怎么跑起来

​    \- 怎么看结果

  \- RPSVM

​    \- 代码在哪里

​    \- 怎么跑起来

​    \- 怎么看结果

\- Other tools

  \- fusion

​    \- 为了干什么

​    \- 代码

​    \- 例子

  \- attention visualization

​    \- 为了干什么

​    \- 代码 

​    \- 例子 
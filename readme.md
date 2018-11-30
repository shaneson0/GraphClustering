

# 复现Learning Deep Representations for Graph Clustering

使用深度学习求解图聚类问题，论文《Learning Deep Representations for Graph Clustering》

运行代码

```python
    
    python test.py

```


```python
    
    kmeans_sae is : 0.542860749905315
    kmeans_raw is : 0.720237343769726

```


## 网络结构

```python

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
first_input (InputLayer)     (None, 178)               0         
_________________________________________________________________
first (Dense)                (None, 178)               31862     
_________________________________________________________________
second (Dense)               (None, 128)               22912     
_________________________________________________________________
embed (Dense)                (None, 64)                8256      
=================================================================

```

## 运行截图

![image.png](https://upload-images.jianshu.io/upload_images/5786775-24f336e38a95feba.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




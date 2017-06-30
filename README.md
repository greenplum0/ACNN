# ACNN
论文基于注意力机制的卷积神经网络模型 源代码

# Requirement
- python2.7
- keras 2.0
- sklearn 
- numpy
- keras backend = theano 

# 论文中提出一种基于注意力机制的卷积神经网络模型
文件说明：  
mr：电影评论数据集，分为两个文件，一个正向评论，一个负向评论。每个评论文件中每一行对应一条评论句  
con_att.py：模型的主要文件  
data_loader.py:数据加载与预处理  
word_vectors.save:针对数据集生成的词向量文件  

# 运行模型
模型接收两个参数：
mode：max  atten
inner: inner outer

运行ACNN-INNER：python con_atten.py atten inner  
运行ACNN-OUTER：python con_atten.py atten outer  
运行CNN：python con_atten.py max inner  

# 自己运行需要修改的地方
将文件con_atten.py 中第38,65行左右的文件路径修改为自己的本地路径  
根据自己的方式修改卷积层的参数  



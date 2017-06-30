# ACNN
论文基于注意力机制的卷积神经网络模型 源代码

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



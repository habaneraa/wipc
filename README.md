# Weibo Interaction Prediction Challenge

[微博互动预测挑战](https://tianchi.aliyun.com/competition/entrance/231574/information) (<u>W</u>eibo <u>I</u>nteraction <u>P</u>rediction <u>C</u>allenge)

> 对于一条原创博文而言,转发、评论、赞等互动行为能够体现出用户对于博文内容的兴趣程度，也是对博文进行分发控制的重要参考指标。本届赛题的任务就是根据抽样用户的原创博文在发表一天后的转发、评论、赞总数，建立博文的互动模型，并预测用户后续博文在发表一天后的互动情况。

## 数据情况

文件路径: `data/raw/weibo_train_data.txt` & `data/raw/weibo_predict_data.txt`

训练数据
- 包含 2015-02-01 至 2015-07-31 的博文
- 总计 1,229,618 条
- 37,263 个用户
- 每条数据包含：
  1. 用户 id
  2. 发表时间
  3. 文本内容
  4. 互动 (点赞/评论/转发)

测试数据
- 2015-08-01 至 2015-08-31
- 总计 178,297 条数据
- 预测互动次数 (点赞/评论/转发)

评估指标
- 已在 `metric.py` 中实现

验证集划分
- 将训练集中 7 月份发表的博文留出，作为验证集，总计 184,937 条


## 数据可视化

1. [散点图+线图] 互动数量分布，对数坐标，累积图
2. [矩阵或热力图] 3 个数值列，线性相关性系数
3. 用户统计
   1. [散点图] 每个用户累加的互动数量
   2. [柱状图] 每个用户的发帖数量
   3. [热力图] 用户发帖的平均点赞、平均转发、平均评论、发帖量，之间的相关性系数
4. 时间分析
   1. [折线图或曲线图] time of day 发帖量
   2. [折线图或柱状图] day of week 发帖量
5. 文本内容
   1. [柱状图] 文本长度分布 按字符 按单词
   2. 高频词 ?
   3. 自动聚类 ?

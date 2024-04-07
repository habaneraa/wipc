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

## Data Analysis & Visualization

1 - Distribution of post engagements

- "Engagements" represent likes, forwards, or comments
- Engagement metrics refer to the number of likes, shares, and comments on a post.
- `xxx_value_counts` chart:
  - The data point (x, y) indicates that there are y posts **receiving** x likes/forwards/comments.
- `cumulative_xxx_value_percent` chart:
  - The data point (x, y) indicates that there are y percent of posts receiving **over** x likes/forwards/comments.

2 - Correlations among post engagement metrics

- We calculated the **linear correlation coefficients**.
- Any pair of engagement metrics shows a positive correlation, which is in line with intuition.
- The coefficients of around 0.6 suggests *moderately* positive linear relationships.
- Using one metric to predict others may not be accurate enough.

3 - User statistics and distributions

- `num_posts_of_users_counts` chart:
  - The data point (x, y) represents that there are y users who have published x blog posts in the past.
- `cumulative_num_posts` chart:
  - There are y percent of users who have published more than x blog posts in the past.
- `user_mean_xxx_value_counts` & `cumulative_user_mean_xxx_value_percent` charts:
  - The data point (x, y) indicates that there are y percent of users receiving **over** x likes/forwards/comments on average.
- `user_engagement_corr` matrix:
  - For each user, we describe the mean value of likes/forwards/comments, as well as the number of posts.
  - We calculated the **linear correlation coefficients** among the 4 metrics.
  - "Number of Posts" almost has no linear correlation with other engagement metrics.
  - The correlation between average likes and average comments is strongly positive.
  - These coefficients may be affected by the data sparsity.

4 - The time of posting

- We have counted the number of posts at different times of the day. Hour 1st to hour 24th.
- We have counted the number of posts on different days of the week. Monday to Sunday.

5 - Text contents of posts

- We have cleaned the text contents by removing all URL links.
- We count the length of text contents by *characters*.

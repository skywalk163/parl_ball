# parl_ball

使用parl强化学习框架，对paddle ball进行强化学习。
目前看学习的效果还有待提高。


运行python train.py即可。
第二个项目：python paddleball.py

每10个episode输出一次接球情况，输出信息中，点代表失球，叹号代表接到球！
每100个episode用图形界面验证一次。可以看看ai怎么操作的。

发现很多次学习后，AI学会了龟缩在左边角上，这样能连续接2-3个，但是如果球跑到右边，就有可能接不住。

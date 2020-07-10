# parl_ball

使用parl强化学习框架，对paddle ball进行强化学习。
目前看学习的效果还有待提高。


第一个项目：运行python train.py即可。
第二个项目：运行python paddleball.py
第三个项目：运行python pb.py
第四个项目：运行pyrhon pbdqn+.py
前三个项目都是使用PolicyGradient，目前第三个项目训练1000次效果最好。
第四个项目使用DQN，效果更好。

为了程序运行速度较快，修改了原游戏代码，添加了不画图的fun_frame_quick函数，在训练的时候使用这个函数以加快速度。同时增加了输出信息：
每10个episode输出一次接球情况，输出信息中，点代表失球，叹号代表接到球！
每100个episode用图形界面验证一次。可以看看ai怎么操作的。
比如使用DQN，第800个episode输出为：
[07-10 20:02:20 MainThread @pbdqn+.py:438] episode:800    e_greed:0.01   test_reward:5.540000000000068
! ! ! . ! ! ! . ! ! ! ! . ! ! ! ! ! . ! ! ! ! ! ! ! ! ! ! . ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! . ! ! ! ! ! . ! . ! ! ! ! ! ! ! . ! ! ! ! ! ! ! ! . 810
中间的叹号表示接到球，点号表示失球，一共失了10个球，episode计数到810 。

发现在使用PG策略的时候，很多次学习后，AI学会了龟缩在左边角上，这样能连续接2-3个，但是如果球跑到右边，就有可能接不住。log输出如下：
[07-10 21:00:52 MainThread @pb+.py:343] 
Episode 230, Reward Sum -1.6699999999999569.
! ! ! . ! ! ! . ! ! ! . ! ! ! . ! ! ! . ! ! ! . ! ! ! . ! ! ! . ! ! ! . ! ! ! . 
可以明显看到每次都是同样的接到3个球。如果使用图形方式，会看到球拍一直在最左边，在接到3个球后，第四个球的时候球拍不会向右移动，导致丢球。

另外就是AI因为步数负反馈，会导致产生懒惰的情况，也就是不动，硬吃-3的reward
第三个项目表现不错！第三个项目相比前两个PQ策略项目，修改的地方有：
1 超参数调整：
LEARNING_RATE = 4e-3
GAMMA=0.98

2 将obs输出数据由5项该写成4项
return [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]
return [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx/3*2+self.ball.dy/3]






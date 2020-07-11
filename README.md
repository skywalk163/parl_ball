# parl_ball

使用parl强化学习框架，对paddle ball进行强化学习。
目前看学习的效果还有待提高。


第一个项目：运行python train.py即可。
第二个项目：运行python paddleball.py
第三个项目：运行python pb.py
第四个项目：运行pyrhon pbdqn+.py
前三个项目都是使用PolicyGradient，目前PG策略中第三个项目训练1000次效果最好。

第四个项目使用DQN，效果更好。

为了程序运行速度较快，修改了原游戏代码，添加了不画图的fun_frame_quick函数，在训练的时候使用这个函数以加快速度。同时增加了输出信息：
每10个episode输出一次接球情况，输出信息中，点代表失球，叹号代表接到球！
每100个episode用图形界面验证一次。可以看看ai怎么操作的。

比如使用DQN，第800个episode输出为：
[07-10 20:02:20 MainThread @pbdqn+.py:438] episode:800    e_greed:0.01   test_reward:5.540000000000068
! ! ! . ! ! ! . ! ! ! ! . ! ! ! ! ! . ! ! ! ! ! ! ! ! ! ! . ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! . ! ! ! ! ! . ! . ! ! ! ! ! ! ! . ! ! ! ! ! ! ! ! . 810

中间的叹号表示接到球，点号表示失球，一共失了10个球，episode计数到810 。

发现在使用PG策略的时候，很多次学习后，AI学会了龟缩在左边角上，这样能连续接2-3个，但是如果球跑到右边，就有可能接不住。log输出如下：

[07-10 21:00:52 MainThread @pb+.py:343] Episode 230, Reward Sum -1.6699999999999569.! ! ! . ! ! ! . ! ! ! . ! ! ! . ! ! ! . ! ! ! . ! ! ! . ! ! ! . ! ! ! . ! ! ! . 

可以明显看到每次都是同样的接到3个球。如果使用图形方式，会看到球拍一直在最左边，在接到3个球后，第四个球的时候球拍不会向右移动，导致丢球。

PG项目三运行1000episode的loss变化：

[07-10 23:45:24 MainThread @pb.py:343] Episode 0, Reward Sum -3.0899999999999928.

[07-10 23:49:42 MainThread @pb.py:343] Episode 100, Reward Sum -0.7899999999999388.

[07-10 23:53:30 MainThread @pb.py:343] Episode 200, Reward Sum -2.179999999999981.

[07-10 23:57:48 MainThread @pb.py:343] Episode 300, Reward Sum -1.5299999999999745.

[07-11 00:02:35 MainThread @pb.py:343] Episode 400, Reward Sum -3.06.

[07-11 00:07:15 MainThread @pb.py:343] Episode 500, Reward Sum 1.0400000000000542.

[07-11 00:11:15 MainThread @pb.py:343] Episode 600, Reward Sum 1.7000000000000455.

[07-11 00:15:17 MainThread @pb.py:343] Episode 700, Reward Sum 2.950000000000035.

[07-11 00:19:16 MainThread @pb.py:343] Episode 800, Reward Sum 1.630000000000047.

[07-11 00:23:16 MainThread @pb.py:343] Episode 900, Reward Sum 3.860000000000028.

但是这个reward太少了，实际上在使用了图形界面的情况下，数值要比这好很多，比如有时候能拿到较高的分数：

[07-11 15:14:56 MainThread @pb.py:347] Episode 630, Reward Sum 185.74000000000302.
! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! . ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! . ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! . ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! . ! ! ! ! ! . ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! . ! ! ! ! ! ! ! ! ! ! ! . ! ! ! ! ! ! ! ! ! ! ! . ! ! ! ! ! ! . ! ! ! ! ! ! ! ! ! ! ! ! . 
根据叹号可以看到能连续保持21次接球成功，这对PG策略来说已经比较不错了。



另外就是AI因为步数负反馈，会导致产生懒惰的情况，也就是不动，硬吃-3的reward

第三个项目表现不错！第三个项目相比前两个PQ策略项目，修改的地方有：

1 超参数调整：
LEARNING_RATE = 4e-3
GAMMA=0.98


2 将obs输出数据由5项该写成4项
return [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]

改成：
return [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx/3*2+self.ball.dy/3]

也就是将dx和dy的数据由两项变成一项，而且这一项也只有4个数值[-3 -1 1 3]

并且后期将最后一项改写成：(self.ball.dx/3+1)+(self.ball.dy/3+1)/2 这样就将这项的取值范围限定为这4个数[0 1 2 3]


3 将移动一次的反馈由-0.1减少到-0.01，以降低PG策略懒惰到不动的情况发生。


另外PG策略验证的时候，程序也有点问题，导致复现分数很低，我进行了修改，在使用python pb.py 运行后，会生成paddleball.ckpt模型文件，然后运行python pbtest.py进行图形界面验证，它会自动调用前面生成的paddleball.ckpt模型文件，实验下来，可以接球几十次到上百次。log文件如下：

[07-11 21:40:31 MainThread @pbtest.py:373] Test reward: 237.87000000000006

[07-11 21:47:13 MainThread @pbtest.py:373] Test reward: 444.1800000000001

[07-11 21:49:01 MainThread @pbtest.py:373] Test reward: 152.84000000000003

[07-11 21:49:35 MainThread @pbtest.py:373] Test reward: 50.34000000000001

[07-11 21:57:04 MainThread @pbtest.py:373] Test reward: 603.6500000000001


PG策略运行图：

![](https://github.com/skywalk163/parl_ball/blob/master/img/pbpg.gif)



DQN策略运行图

![](https://github.com/skywalk163/parl_ball/blob/master/img/pbdqn%2B.gif)




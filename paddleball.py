
import numpy as np
import parl
import os.path
import paddle
import paddle.fluid as fluid
from parl.utils import logger
# Author: Shiva Verma


import turtle as t


class Paddle():

    def __init__(self):

        self.done = False
        self.reward = 0
        self.hit, self.miss = 0, 0

        # Setup Background

        self.win = t.Screen()
        self.win.title('Paddle')
        self.win.bgcolor('black')
        self.win.setup(width=600, height=600)
        self.win.tracer(0)

        # Paddle

        self.paddle = t.Turtle()
        self.paddle.speed(0)
        self.paddle.shape('square')
        self.paddle.shapesize(stretch_wid=1, stretch_len=5)
        self.paddle.color('white')
        self.paddle.penup()
        self.paddle.goto(0, -275)

        # Ball

        self.ball = t.Turtle()
        self.ball.speed(0)
        self.ball.shape('circle')
        self.ball.color('red')
        self.ball.penup()
        self.ball.goto(0, 100)
        self.ball.dx = 3
        self.ball.dy = -3

        # Score

        self.score = t.Turtle()
        self.score.speed(0)
        self.score.color('white')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, 250)
        self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))

        # -------------------- Keyboard control ----------------------

        self.win.listen()
        self.win.onkey(self.paddle_right, 'Right')
        self.win.onkey(self.paddle_left, 'Left')

    # Paddle movement

    def paddle_right(self):

        x = self.paddle.xcor()
        if x < 225:
            self.paddle.setx(x+20)

    def paddle_left(self):

        x = self.paddle.xcor()
        if x > -225:
            self.paddle.setx(x-20)

    # ------------------------ AI control ------------------------

    # 0 move left
    # 1 do nothing
    # 2 move right

    def reset(self):

        self.paddle.goto(0, -275)
        self.ball.goto(0, 100)
        return [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]

    def step(self, action, render=False):

        self.reward = 0
        self.done = 0

        if action == 0:
            self.paddle_left()
            self.reward -= .1 #.1

        if action == 2:
            self.paddle_right()
            self.reward -= .1 #.1

        if render:
            self.run_frame()
        else:
            self.run_frame_quick()
        state = [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]
        return self.reward, state, self.done

    def run_frame(self):

        self.win.update()

        # Ball moving

        self.ball.setx(self.ball.xcor() + self.ball.dx)
        self.ball.sety(self.ball.ycor() + self.ball.dy)

        # Ball and Wall collision

        if self.ball.xcor() > 290:
            self.ball.setx(290)
            self.ball.dx *= -1

        if self.ball.xcor() < -290:
            self.ball.setx(-290)
            self.ball.dx *= -1

        if self.ball.ycor() > 290:
            self.ball.sety(290)
            self.ball.dy *= -1

        # Ball Ground contact

        if self.ball.ycor() < -290:
            self.ball.goto(0, 100)
            self.miss += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            #self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))



            self.reward -= 3
            self.done = True

        # Ball Paddle collision

        if abs(self.ball.ycor() + 250) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 55:
            self.ball.dy *= -1
            self.hit += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            self.reward += 3
            #self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))



    def run_frame_quick(self):

        #self.win.update()

        # Ball moving

        self.ball.setx(self.ball.xcor() + self.ball.dx)
        self.ball.sety(self.ball.ycor() + self.ball.dy)

        # Ball and Wall collision

        if self.ball.xcor() > 290:
            self.ball.setx(290)
            self.ball.dx *= -1

        if self.ball.xcor() < -290:
            self.ball.setx(-290)
            self.ball.dx *= -1

        if self.ball.ycor() > 290:
            self.ball.sety(290)
            self.ball.dy *= -1

        # Ball Ground contact

        if self.ball.ycor() < -290:
            self.ball.goto(0, 100)
            self.miss += 1
#             self.score.clear()
#             self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            #logger,info(f"Hit: {self.hit}   Missed: {self.miss}")
#             logger.info(f"Game Over Hit:{self.hit} Missed:{self.miss}")
            print(".", end=" ")
            self.reward -= 3 #3
            self.done = True

        # Ball Paddle collision

        if abs(self.ball.ycor() + 250) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 55:
            self.ball.dy *= -1
            self.hit += 1
            self.score.clear()
#             self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
#             logger.info(f"^-^ Good job!Hit: {self.hit}   Missed: {self.miss}")
            print("!", end=" ")
            self.reward += 3 #3



# while True:
#
#      env.run_frame()
import parl
from parl import layers

class BallModel(parl.Model):
    def __init__(self, act_dim):
        act_dim = act_dim
        hid1_size = act_dim * 10

        self.fc1 = layers.fc(size=hid1_size, act='tanh')
        self.fc2 = layers.fc(size=act_dim, act='softmax')

    def forward(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        return out
    
class BallAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(BallAgent, self).__init__(algorithm)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.train_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.act_prob = self.alg.predict(obs)

        with fluid.program_guard(self.train_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(name='act', shape=[1], dtype='int64')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            self.cost = self.alg.learn(obs, act, reward)

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.random.choice(range(self.act_dim), p=act_prob)
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.argmax(act_prob)
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int64'),
            'reward': reward.astype('float32')
        }
        cost = self.fluid_executor.run(
            self.train_program, feed=feed, fetch_list=[self.cost])[0]
        return cost
OBS_DIM = 5
ACT_DIM = 3
LEARNING_RATE = 1e-3
model = BallModel(act_dim=3)
alg = parl.algorithms.PolicyGradient(model, lr=1e-3)
#alg=parl.algorithms.DQN(model,lr=0.001,gamma=0.99)
agent = BallAgent(alg, obs_dim=OBS_DIM, act_dim=3)
#成功执行
def run_episode(env, agent, train_or_test='train'):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        if train_or_test == 'train':
            action = agent.sample(obs)
            action_list.append(action)
            reward, obs, done = env.step(action)
        else:
            action = agent.predict(obs)
            action_list.append(action)

        #obs, reward, done, info = env.step(action)
            reward, obs, done = env.step(action,True)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list
def calc_reward_to_go(reward_list):
    for i in range(len(reward_list) - 2, -1, -1):
        reward_list[i] += reward_list[i + 1]
    return np.array(reward_list)


#env = gym.make("CartPole-v0")
env = Paddle()
for i in range(1000):
      obs_list, action_list, reward_list = run_episode(env, agent)
      if i % 10 == 0:
          logger.info("Episode {}, Reward Sum {}.".format(i, sum(reward_list)))

      batch_obs = np.array(obs_list)
      batch_action = np.array(action_list)
      #batch_reward = calc_discount_norm_reward(reward_list, GAMMA)
      batch_reward = calc_reward_to_go(reward_list)

      agent.learn(batch_obs, batch_action, batch_reward)
      if (i + 1) % 100 == 0:
          _, _, reward_list = run_episode(env, agent, train_or_test='test')
          total_reward = np.sum(reward_list)
          logger.info('Test reward: {}'.format(total_reward))

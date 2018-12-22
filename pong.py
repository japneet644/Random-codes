# #PONG pygame
#
# import random
# from tkinter import *
#
# # import pygame, sys
# # from pygame.locals import *
#
# # pygame.init()
# # fps = pygame.time.Clock()
# root = Tk()
# canvas = Canvas(root,width=600,height=400)
# canvas.pack()
#
# #colors
# WHITE = (255,255,255)
# RED = (255,0,0)
# GREEN = (0,255,0)
# BLACK = (0,0,0)
#
# #globals
# WIDTH = 600
# HEIGHT = 400
# BALL_RADIUS = 20
# PAD_WIDTH = 8
# PAD_HEIGHT = 80
# HALF_PAD_WIDTH = PAD_WIDTH / 2
# HALF_PAD_HEIGHT = PAD_HEIGHT / 2
# ball_pos = [0,0]
# ball_vel = [0,0]
# paddle1_vel = 0
# paddle2_vel = 0
# l_score = 0
# r_score = 0
#
# #canvas declaration
# # window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
# # pygame.display.set_caption('Hello World')
#
# # helper function that spawns a ball, returns a position vector and a velocity vector
# # if right is True, spawn to the right, else spawn to the left
# def ball_init(right):
#     global ball_pos, ball_vel # these are vectors stored as lists
#     ball_pos = [WIDTH/2,HEIGHT/2]
#     horz = random.randrange(2,4)
#     vert = random.randrange(1,3)
#
#     if right == False:
#         horz = - horz
#
#     ball_vel = [horz,-vert]
#
# # define event handlers
# def init():
#     global paddle1_pos, paddle2_pos, paddle1_vel, paddle2_vel,l_score,r_score  # these are floats
#     global score1, score2  # these are ints
#     paddle1_pos = [HALF_PAD_WIDTH - 1,HEIGHT/2]
#     paddle2_pos = [WIDTH +1 - HALF_PAD_WIDTH,HEIGHT/2]
#     l_score = 0
#     r_score = 0
#     if random.randrange(0,2) == 0:
#         ball_init(True)
#     else:
#         ball_init(False)
#
#
# #draw function of canvas
# def draw(canvas):
#     global paddle1_pos, paddle2_pos, ball_pos, ball_vel, l_score, r_score
#
#     # canvas.fill(BLACK)
#     canvas.create_line( WIDTH/2,0, WIDTH/2,HEIGHT)
#     canvas.create_line(  PAD_WIDTH, 0,PAD_WIDTH, HEIGHT)
#     canvas.create_line( WIDTH - PAD_WIDTH, 0,WIDTH - PAD_WIDTH, HEIGHT)
#     canvas.create_oval( WIDTH//2, HEIGHT//2, 70, 1)
#
#     # update paddle's vertical position, keep paddle on the screen
#     if paddle1_pos[1] > HALF_PAD_HEIGHT and paddle1_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
#         paddle1_pos[1] += paddle1_vel
#     elif paddle1_pos[1] == HALF_PAD_HEIGHT and paddle1_vel > 0:
#         paddle1_pos[1] += paddle1_vel
#     elif paddle1_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle1_vel < 0:
#         paddle1_pos[1] += paddle1_vel
#
#     if paddle2_pos[1] > HALF_PAD_HEIGHT and paddle2_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
#         paddle2_pos[1] += paddle2_vel
#     elif paddle2_pos[1] == HALF_PAD_HEIGHT and paddle2_vel > 0:
#         paddle2_pos[1] += paddle2_vel
#     elif paddle2_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle2_vel < 0:
#         paddle2_pos[1] += paddle2_vel
#
#     #update ball
#     ball_pos[0] += int(ball_vel[0])
#     ball_pos[1] += int(ball_vel[1])
#
#     #draw paddles and ball
#     canvas.create_oval(  ball_pos[0]-BALL_RADIUS,ball_pos[1]-BALL_RADIUS,ball_pos[0]+BALL_RADIUS,ball_pos[1]-BALL_RADIUS)
#     canvas.create_polygon( [[paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT], [paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT], [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT], [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT]])
#     canvas.create_polygon(  [[paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT], [paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT], [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT], [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT]])
#
#     #ball collision check on top and bottom walls
#     if int(ball_pos[1]) <= BALL_RADIUS:
#         ball_vel[1] = - ball_vel[1]
#     if int(ball_pos[1]) >= HEIGHT + 1 - BALL_RADIUS:
#         ball_vel[1] = -ball_vel[1]
#
#     #ball collison check on gutters or paddles
#     if int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH and int(ball_pos[1]) in range(int(paddle1_pos[1] - HALF_PAD_HEIGHT),int(paddle1_pos[1] + HALF_PAD_HEIGHT),1):
#         ball_vel[0] = -ball_vel[0]
#         ball_vel[0] *= 1.1
#         ball_vel[1] *= 1.1
#     elif int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH:
#         r_score += 1
#         ball_init(True)
#
#     if int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH and int(ball_pos[1]) in range(int(paddle2_pos[1] - HALF_PAD_HEIGHT),int(paddle2_pos[1] + HALF_PAD_HEIGHT),1):
#         ball_vel[0] = -ball_vel[0]
#         ball_vel[0] *= 1.1
#         ball_vel[1] *= 1.1
#     elif int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH:
#         l_score += 1
#         ball_init(False)
#
#     #update scores
#     # myfont1 = pygame.font.SysFont("Comic Sans MS", 20)
#     # label1 = myfont1.render("Score "+str(l_score), 1, (255,255,0))
#     # canvas.blit(label1, (50,20))
#     #
#     # myfont2 = pygame.font.SysFont("Comic Sans MS", 20)
#     # label2 = myfont2.render("Score "+str(r_score), 1, (255,255,0))
#     # canvas.blit(label2, (470, 20))
#
#
# #keydown handler
# def keyPressed(event):
#     if event.type == KEYDOWN:
#         keydown(event)
#     elif event.type == KEYUP:
#         keyup(event)
#     elif event.type == QUIT:
#         root.destroy()
#
# def keydown(event):
#     global paddle1_vel, paddle2_vel
#
#     if event.key == K_UP:
#         paddle2_vel = -8
#     elif event.key == K_DOWN:
#         paddle2_vel = 8
#     elif event.key == K_w:
#         paddle1_vel = -8
#     elif event.key == K_s:
#         paddle1_vel = 8
#
# #keyup handler
# def keyup(event):
#     global paddle1_vel, paddle2_vel
#
#     if event.key in (K_w, K_s):
#         paddle1_vel = 0
#     elif event.key in (K_UP, K_DOWN):
#         paddle2_vel = 0
#
# init()
# ball_init(True)
# root.bind_all('<Key>', keyPressed)
# #game loop
# while True:
#
#     draw(canvas)
#     root.mainloop()
#     # for event in canvas.event.get():
#     #
#     #     if event.type == KEYDOWN:
#     #         keydown(event)
#     #     elif event.type == KEYUP:
#     #         keyup(event)
#     #     elif event.type == QUIT:
#     #         pygame.quit()
#     #         sys.exit()
#
#     # pygame.display.update()
#     fps.tick(60)
from tkinter import *
import random
import time
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.brea = 0
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs=1, mini_batch_size=1, eta=1,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        # for j in xrange(epochs):
        #     # random.shuffle(training_data)
        #     mini_batches = [
        #         training_data[k:k+mini_batch_size]
        #         for k in xrange(0, n, mini_batch_size)]
        #     for mini_batch in mini_batches:
        self.update_mini_batch(training_data, eta)
            # if test_data:
            #     print "Epoch {0}: {1} / {2}".format(
            #         j, self.evaluate(test_data), n_test)
            # else:
            #     print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta=2):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        x, y = mini_batch
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta)*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta)*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = [np.array(u) for u in x ]
        activations = [np.array(x)] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        # print(delta)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    def Policy(self):
        time = 0    
        while self.count < 1:
            self.count += 1
            print(self.pos, self.theta, self.reward,self.Qs)
            self.pos = [10.0,7.0]
            time = 0
            self.brea = 0
            while time<6000  and self.brea==0:
                time += 1
                self.epsilon = self.epsilon/(1+self.count//100)
                for a in range(Nofactions):
                    self.Qs[a] = self.feedforward([[self.pos[0]],[self.pos[1]],[3.14*self.theta/180],[a]])

                if random.uniform(0, 1)<(1-self.epsilon):
                    action = random.randint(0,Nofactions-1)
                else:
                    action = np.argmax(self.Qs)
                Qs2 = np.zeros(Nofactions)
                prew_pos,prev_theta = self.pos,self.theta
                for a in range(Nofactions):
                    newpos,newtheta = self.next_state(a)
                    Qs2[a] = self.feedforward([[newpos[0]],[newpos[1]],[3.14*newtheta/180],[a]])
                self.reward = 0
                if self.Is_collision()==1:
                    self.reward = -2
                    self.brea = 1
                elif newpos[1] > 15:
                    self.reward = 1
                    self.brea=1
                elif time==6000:
                    self.reward=-1
                # else:
                #     self.reward = 0.5 #-15+ self.pos[1]
                action2 = np.argmax(Qs2)
                self.Qs[action]=self.Qs[action] + 0.5*(self.reward + Qs2[action2]- self.Qs[action])
                # mini_batch = [[[prew_pos[0]],[prew_pos[1]],[prev_theta],[action]],self.Qs[action] + 0.5*(self.reward + Qs2[action2]- self.Qs[action])]
                self.update_mini_batch([[[prew_pos[0]],[prew_pos[1]],[prev_theta],[action]],self.Qs[action]])
                self.pos,self.theta = self.next_state(action2)
                # print(action2)
        return

#### Miscellaneous functions
# def sigmoid(z):
#     """The sigmoid function."""
#     return 1.0/(1.0+np.exp(-z))
#
# def sigmoid_prime(z):
#     """Derivative of the sigmoid function."""
#     return sigmoid(z)*(1-sigmoid(z))
#

# Define ball properties and functions
class Ball:
    def __init__(self, canvas, color, size, paddle):
        self.canvas = canvas
        self.paddle = paddle
        self.pos = [0,0,0,0]
        # self.color = color
        # self.posy = 10
        self.id = canvas.create_oval(10, 10, size, size, fill=color)
        # self.canvas.move(self.id, 245, 100)
        self.xspeed = random.randrange(-3,3)
        self.yspeed = -1
        self.hit_bottom = False
        self.score = 0

    def draw(self):
        self.canvas.move(self.id, self.xspeed, self.yspeed)
        # self.posx+=self.xspeed
        # self.posy+=self
        # canvas.create_oval(self.posx, self.posy, size, size, fill=self.color)
        self.pos = self.canvas.coords(self.id)
        if self.pos[1] <= 0:
            self.yspeed = 3
        if self.pos[3] >= 400:
            self.hit_bottom = True
        if self.pos[0] <= 0:
            self.xspeed = 3
        if self.pos[2] >= 500:
            self.xspeed = -3
        if self.hit_paddle() == True:
            self.yspeed = -3
            self.xspeed = random.randrange(-3,3)
            self.score += 1

    def hit_paddle(self):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if self.pos[2] >= paddle_pos[0] and self.pos[0] <= paddle_pos[2]:
            if self.pos[3] >= paddle_pos[1] and self.pos[3] <= paddle_pos[3]:
                return True
        return False

# Define paddle properties and functions
class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0,0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 350)
        self.xspeed = 0
        self.canvas.bind_all('<KeyPress-Left>', self.move_left)
        self.canvas.bind_all('<KeyPress-Right>', self.move_right)
        self.pos = [0,0,0,0]

    def draw(self):
        self.canvas.move(self.id, self.xspeed, 0)
        self.pos = self.canvas.coords(self.id)
        if self.pos[0] <= 0:
            self.xspeed = 0
        if self.pos[2] >= 500:
            self.xspeed = 0

    def move_left(self):
        self.xspeed = -3
    def move_right(self):
        self.xspeed = 3


N = Network([5,16,1])
# Create window and canvas to draw on

tk = Tk()
tk.title("Ball Game")
canvas = Canvas(tk, width=500, height=400, bd=0, bg='papaya whip')
canvas.pack()
# label = canvas.create_text(5, 5, anchor=NW, text="Score: 0")
tk.update()
paddle = Paddle(canvas, 'blue')
ball = Ball(canvas, 'red', 25, paddle)

# Animation loop
while ball.hit_bottom == False:
    ball.draw()
    paddle.draw()

    # canvas.itemconfig(label, text="Score: "+str(ball.score))
    # tk.update_idletasks()
    # tk.update()
    time.sleep(0.01)
# Game Over
# go_label = canvas.create_text(250,200,text="GAME OVER",font=("Helvetica",30))
tk.update()

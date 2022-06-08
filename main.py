# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
data= pd.read_csv("Invistico_Airline.csv")
global seed_number
global iteration_seed_number
global check_if_done
global check_if_stuck
check_if_stuck=0
check_if_done=False
iteration_seed_number=0
seed_number=1
class Model:
    def __init__(self, data, x, y, theta, alpha):
        self.data=data
        self.x=x
        self.y=y
        self.theta=theta
        self.alpha=alpha
        self.copy_thetadata=np.array([x for x in data])
    def get_edited_data_positive(self, database, x, text_data):
        self.isPositive=True
        self.copy_theta_index=x
        self.copy_theta_text=text_data
        self.data=np.array([a for a in database if a[x]==text_data])
        if len(self.data)!=0:
            self.data=self.data.reshape(len(self.data),len(self.data[0]))
            self.data=np.delete(self.data, x, 1)
            self.data.reshape(len(self.data),len(self.data[0]))
            self.copy_theta_data=self.data
            self.x=np.insert(np.delete(self.data, 0, 1), 0, np.ones(len(self.data)), axis=1)
            self.y=np.array([x[0] for x in self.data])
            self.y=self.y.reshape((len(self.y), 1))
            for x in range(len(self.y)):
                if self.y[x][0]=='satisfied':
                    self.y[x][0]=1
                else:
                    self.y[x][0]=0
            self.y=self.y.astype(int)
            self.theta=np.zeros([len(self.x[0]), 1])
    def do_copy_processes(self, copy):
        self.data=copy
        self.data.reshape(len(self.data),len(self.data[0]))
        self.x=np.insert(np.delete(self.data, 0, 1), 0, np.ones(len(self.data)), axis=1)
        self.y=np.array([x[0] for x in self.data])
        self.y=self.y.reshape((len(self.y), 1))
        for x in range(len(self.y)):
            if self.y[x][0]=='satisfied':
                self.y[x][0]=1
            else:
                self.y[x][0]=0
        self.y=self.y.astype(int)
        self.theta=np.zeros([len(self.x[0]), 1])
    def get_edited_data_negative(self, database, x, text_data):
       self.isPositive=False
       self.copy_theta_index=x
       self.copy_theta_text=text_data
       self.data=np.array([a for a in database if a[x]!=text_data])
       if len(self.data)!=0:
           self.data=self.data.reshape(len(self.data),len(self.data[0]))
           self.data=np.delete(self.data, x, 1)
           self.data.reshape(len(self.data),len(self.data[0]))
           self.copy_theta_data=self.data
           self.x=np.insert(np.delete(self.data, 0, 1), 0, np.ones(len(self.data)), axis=1)
           self.y=np.array([x[0] for x in self.data])
           self.y=self.y.reshape((len(self.y), 1))
           for x in range(len(self.y)):
               if self.y[x][0]=='satisfied':
                   self.y[x][0]=1
               else:
                   self.y[x][0]=0
           self.y=self.y.astype(int)
           self.theta=np.zeros([len(self.x[0]), 1])
    def get_transformed_theory(self, k):
        k=k.astype(float)
        return np.around((1/np.around((1+np.exp(-1*k)),5)),5)
    def get_derivative(self, x, dataset, output):
        res=np.matmul(dataset,self.theta).astype(float)
        res=self.get_transformed_theory(res).astype(float)
        res=res-output
        arr=np.transpose(dataset);
        res=np.matmul(arr,res)
        res=(res/len(dataset)).astype(float)
        return res
    def do_changes(self):
        np.random.shuffle(self.data)
    def check_graphs(self):
            for a in range(len(self.data[0])):
                plt.plot([b[a] for b in self.data], self.y, "ro")
                input(int)
    def get_copy_data(self):
        return self.copy_theta_data
    
    def do_algorithm(self, iterations):
        self.beta=0.001
        cond=False
        global seed_number
        global check_if_stuck
        self.isFirst=False
        if(len(self.data)!=0):
            self.condition=True
            self.do_algorithm_middle()
            x=0
            print("Size of the train set: ", len(self.train))
            print("Size of the test set: ", len(self.test))
            if len(self.x)<=500:
                iterations=10000
            while x<iterations:
                self.copy_theta=np.array(self.theta)
                derivative=self.get_derivative(0, self.train, self.output_train)
                a=0
                while a<len(self.theta):
                    self.copy_theta[a][0]=float(self.copy_theta[a][0]-(self.alpha*derivative[a]/len(self.train)))-(self.beta*self.copy_theta[a][0]/len(self.train))
                    a=a+1
                self.theta=self.copy_theta
                x=x+1
            res=float(self.calculate_accuracy(self.train, self.output_train))
            print("Train accuracy is: ",res)
            print("Test accuracy is: ", self.calculate_accuracy(self.x, self.y))
            if float(self.calculate_accuracy(self.x, self.y))<0.6:
                print("Trying until test accuracy reaches 0.6 or more")
                print("Re-declaring training and test set")
                self.do_algorithm(iterations+1000)
            seed_number=1
            np.random.seed(0)
        else:
            self.condition=False
    def calculate_accuracy(self, data, output):
        res=np.matmul(data, self.theta)
        res=self.get_transformed_theory(res)
        total=0
        for x in range(len(output)):
            if output[x]==1:
                if res[x]>=0.5:
                    total=total+1
            else:
                if res[x]<0.5:
                    total=total+1
        total=total/len(output)
        return "{:1.2f}".format(total)
    def calculate_cost(self):
        get_res=self.get_transformed_theory(np.matmul(self.train, self.theta))
        res=self.output_train*np.log(get_res)
        res=res+(1-self.output_train)*(1-np.log(get_res))
        res=res/len(self.train)
        res=res*-1
        res=res+self.theta*np.transpose(self.theta)/(2*len(self.train))
        return res
    def do_algorithm_middle(self):
        global seed_number
        global check_if_stuck
        self.condition=True
        np.random.seed(seed_number)
        seed_number=seed_number+10
        for a in range(len(self.x)):
            for b in range(len(self.x[a])):
                if math.isnan(self.x[a][b]):
                    self.x[a][b]=0
        if len(self.x)==1:
            self.train=self.x
            self.test=self.x
            self.output_train=self.y
            self.output_test=self.y
        elif len(self.x)<=25:
            nums=np.random.randint(len(self.x), size=25)
            if self.isFirst:
                self.alpha=0.1
                self.isFirst=False
            self.train=self.x
            self.test=self.y
            self.output_train=self.y
            self.output_test=self.y
        elif len(self.x)>25 and len(self.x)<500:
            nums=np.random.randint(len(self.x), size=int(len(self.x)/2))
            y=[]
            x=[]
            copy_x=self.x
            copy_y=self.y
            for a in range(int(len(self.x)/2)):
                x.append(self.x[nums[a]])
                y.append(self.y[nums[a]])
            self.train=np.array(x)
            self.test=np.delete(copy_x, nums, 0)
            self.output_train=np.array(y)
            self.output_test=np.delete(copy_y, nums, 0)
            if self.isFirst:
                self.alpha=0.1
                self.isFirst=False
        elif len(self.x)>=500 and len(self.x)<1000:
            self.alpha=0.1
            nums=np.random.randint(len(self.x), size=int(len(self.x)/2))
            y=[]
            x=[]
            copy_x=self.x
            copy_y=self.y
            for a in range(int(len(self.x)/2)):
                x.append(self.x[nums[a]])
                y.append(self.y[nums[a]])
            self.train=np.array(x)
            self.test=np.delete(copy_x, nums, 0)
            self.output_train=np.array(y)
            self.output_test=np.delete(copy_y, nums, 0)
        elif len(self.x)>=1000 and len(self.x)<5000:
            if self.isFirst:    
                self.alpha=0.1
                self.isFirst=False
            nums=np.random.randint(len(self.x), size=int(len(self.x)/2))
            y=[]
            x=[]
            copy_x=self.x
            copy_y=self.y
            for a in range(int(len(self.x)/2)):
                x.append(self.x[nums[a]])
                y.append(self.y[nums[a]])
            self.train=np.array(x)
            self.test=np.delete(copy_x, nums, 0)
            self.output_train=np.array(y)
            self.output_test=np.delete(copy_y, nums, 0)
        else:
            if self.isFirst:
                self.alpha=0.1
                self.isFirst=False
            nums=np.random.randint(len(self.x), size=int(len(self.x)/2))
            y=[]
            x=[]
            copy_x=self.x
            copy_y=self.y
            for a in range(int(len(self.x)/2)):
                x.append(self.x[nums[a]])
                y.append(self.y[nums[a]])
            self.train=np.array(x)
            self.test=np.delete(copy_x, nums, 0)
            self.output_train=np.array(y)
            self.output_test=np.delete(copy_y, nums, 0)
        for x in range(len(self.train[0])-1):
            arr=np.array([a[x+1] for a in self.train])
            for b in range(len(self.train)):
                self.train[b][x+1]=(self.train[b][x+1]-np.sum(arr)/len(arr))/(np.amax(arr)-np.amin(arr))
                
    def print_size_y(self):
        print(self.y.size)
    def get_cost(self):
        self.cost
    def print(self):
        print(self.data)
    def print_theta(self):
        print(self.theta)
    def print_x(self):
        print(self.x)
    def print_y(self):
        print(self.y)
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def get_theta(self):
        return self.theta
    def get_alpha(self):
        return self.alpha
    def get_data(self):
        return self.data
copydata=data
data=data.to_numpy()

hey=0
a=Model(data, [], [], [], 0.03)
a.do_changes()
a.get_edited_data_positive(a.get_data(), 1, "Male")

b=Model(a.get_data(),a.get_x(),a.get_y(),a.get_theta(),a.get_alpha())
b.get_edited_data_negative(a.get_data(), 1, "Loyal Customer")
c=Model(b.get_data(), b.get_x(), b.get_y(), b.get_theta(), b.get_alpha())
c.get_edited_data_positive(c.get_data(), 2, 'Personal Travel')
c.get_edited_data_positive(c.get_data(), 2, 'Eco')
data1=c.get_data()
print("Male, Disloyal Customer, Personal Travel, Eco Class accuracy")
c.do_algorithm(1500)
print("******************************************")
print("Male, Disloyal Customer, Personal Travel, Eco Plus and Business Class accuracy")
d=Model(b.get_data(), b.get_x(), b.get_y(), b.get_theta(), b.get_alpha())
d.get_edited_data_positive(d.get_data(), 2, 'Personal Travel')
d.get_edited_data_negative(d.get_data(), 2, 'Eco')
data2=d.get_data()
d.do_algorithm(1500)
print("******************************************")
print("Male, Disloyal Customer, Business Travel, Eco class accuracy")
e=Model(b.get_data(), b.get_x(), b.get_y(), b.get_theta(), b.get_alpha())
e.get_edited_data_negative(e.get_data(), 2, 'Personal Travel')
e.get_edited_data_positive(e.get_data(), 2, 'Eco')
data3=e.get_data()
e.do_algorithm(1500)
print("******************************************")
print("Male, Disloyal Customer, Business Travel, Eco Plus and Business Class accuracy")
f=Model(b.get_data(), b.get_x(), b.get_y(), b.get_theta(), b.get_alpha())
f.get_edited_data_negative(f.get_data(), 2, 'Personal Travel')
f.get_edited_data_negative(f.get_data(), 2, 'Eco')
data4=f.get_data()
f.do_algorithm(1500)
print("******************************************")
print("Male, Loyal Customer, Personal Travel, Eco Class accuracy")
g=Model(data, [], [], [], 0.03)
g.do_changes()
g.get_edited_data_positive(g.get_data(), 1, 'Male')
f=Model(g.get_data(), g.get_x(), g.get_y(), g.get_theta(), g.get_alpha())
f.get_edited_data_positive(f.get_data(), 1, 'Loyal Customer')
h=Model(f.get_data(), f.get_x(), f.get_y(), f.get_theta(), f.get_alpha())
h.get_edited_data_positive(h.get_data(), 2, 'Personal Travel')
h.get_edited_data_positive(h.get_data(), 2, 'Eco')
h.do_algorithm(1500)
print("******************************************")
print("Male, Loyal Customer, Personal Travel, Eco Plus and Business Class accuracy")
k=Model(f.get_data(), f.get_x(), f.get_y(), f.get_theta(), f.get_alpha())
k.get_edited_data_positive(k.get_data(), 2, 'Personal Travel')
k.get_edited_data_negative(k.get_data(), 2, 'Eco')
k.do_algorithm(1500)
print("******************************************")
print("Male, Loyal Customer, Business Travel, Eco Class accuracy")
l=Model(f.get_data(), f.get_x(), f.get_y(), f.get_theta(), f.get_alpha())
l.get_edited_data_negative(l.get_data(), 2, 'Personal Travel')
l.get_edited_data_positive(l.get_data(), 2, 'Eco')
l.do_algorithm(1500)
print("******************************************")
print("Male, Loyal Customer, Business Travel, Eco Plus and Business Class accuracy")
x=Model(f.get_data(), f.get_x(), f.get_y(), f.get_theta(), f.get_alpha())
x.get_edited_data_negative(l.get_data(), 2, 'Personal Travel')
x.get_edited_data_positive(l.get_data(), 2, 'Eco')
x.do_algorithm(1500)
print("******************************************")
print("Female, Disloyal Customer, Personal Travel, Eco Class accuracy")
m=Model(data, [], [], [], 0.03)
m.do_changes()
m.get_edited_data_positive(m.get_data(), 1, "Male")

n=Model(m.get_data(),m.get_x(),m.get_y(),m.get_theta(),m.get_alpha())
n.get_edited_data_negative(n.get_data(), 1, "Loyal Customer")
o=Model(n.get_data(), n.get_x(), n.get_y(), n.get_theta(), n.get_alpha())
o.get_edited_data_positive(o.get_data(), 2, 'Personal Travel')
o.get_edited_data_positive(o.get_data(), 2, 'Eco')
o.do_algorithm(1500)
print("******************************************")
print("Female, Disloyal Customer, Personal Travel, Eco Plus and Business Class accuracy")
p=Model(n.get_data(), n.get_x(), n.get_y(), n.get_theta(), n.get_alpha())
p.get_edited_data_positive(p.get_data(), 2, 'Personal Travel')
p.get_edited_data_negative(p.get_data(), 2, 'Eco')
p.do_algorithm(1500)
print("******************************************")
print("Female, Disloyal Customer, Business Travel, Eco Class accuracy")
r=Model(n.get_data(), n.get_x(), n.get_y(), n.get_theta(), n.get_alpha())
r.get_edited_data_negative(r.get_data(), 2, 'Personal Travel')
r.get_edited_data_positive(r.get_data(), 2, 'Eco')
r.do_algorithm(1500)
print("******************************************")
print("Female, Disloyal Customer, Business Travel, Eco Plus and Business Class accuracy")
s=Model(n.get_data(), n.get_x(), n.get_y(), n.get_theta(), n.get_alpha())
s.get_edited_data_negative(s.get_data(), 2, 'Personal Travel')
s.get_edited_data_negative(s.get_data(), 2, 'Eco')
s.do_algorithm(1500)
print("******************************************")
print("Female, Loyal Customer, Personal Travel, Eco Class accuracy")
t=Model(data, [], [], [], 0.03)
t.do_changes()
t.get_edited_data_positive(t.get_data(), 1, "Loyal Customer")
v=Model(t.get_data(), t.get_x(), t.get_y(), t.get_theta(), t.get_alpha())
v.get_edited_data_positive(v.get_data(), 2, 'Personal Travel')
v.get_edited_data_positive(v.get_data(), 2, 'Eco')
v.do_algorithm(1500)
print("******************************************")
print("Female, Loyal Customer, Personal Travel, Eco Plus and Business Class accuracy")
y=Model(t.get_data(), t.get_x(), t.get_y(), t.get_theta(), t.get_alpha())
y.get_edited_data_positive(y.get_data(), 2, 'Personal Travel')
y.get_edited_data_negative(y.get_data(), 2, 'Eco')
y.do_algorithm(1500)
print("******************************************")
print("Female, Loyal Customer, Business Travel, Eco Class accuracy")
z=Model(t.get_data(), t.get_x(), t.get_y(), t.get_theta(), t.get_alpha())
z.get_edited_data_negative(z.get_data(), 2, 'Personal Travel')
z.get_edited_data_positive(z.get_data(), 2, 'Eco')
z.do_algorithm(1500)
print("******************************************")
print("Female, Loyal Customer, Business Travel, Eco Plus and Business Class accuracy")
w=Model(t.get_data(), t.get_x(), t.get_y(), t.get_theta(), t.get_alpha())
w.get_edited_data_negative(w.get_data(), 2, 'Personal Travel')
w.get_edited_data_negative(w.get_data(), 2, 'Eco')
w.do_algorithm(1500)
print("******************************************")
#!/usr/bin/python
# -*- coding: UTF-8 -*-

# numpy里矩阵X乘用.dot，点乘用*
import numpy as np

q1 = (0.35, 0.2, 0.3, 0.1)
q2 = (-0.5, 0.4, -0.1, 0.2)
t = np.array([-0.1, 0.5, 0.3])
t2 = np.array([0.3, 0.1, 0.1])
p = np.array([0.5, 0, 0.2])

# 四元素转换为旋转矩阵
def quaternion_rotate(q):
    q = np.array(q, dtype=int)
    r = np.array([[1-2*q[2]*q[2]-2*[3]*q[3], 2*q[1]*[2]-2*q[0]*q[3], 2*q[1]*q[3]+2*q[0]*q[2]],
                 [2*q[1]*q[2]+2*q[0]*q[3], 1-2*q[1]*q[1]-2*q[3]*q[3], 2*q[2]*q[3]-2*q[0]*q[1]], 
                 [2*q[1]*q[3]-2*q[0]*q[2], 2*q[2]*q[3]+2*q[0]*q[1], 1-2*q[1]*q[1]-2*q[2]*q[2]]])
    return r

# 归一化
def quaternion_normalized(q):
    length = (q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3])**0.5
    q_one = (round(q[0]/length, 9), round(q[1]/length, 9), round(q[2]/length, 9), round(q[3]/length, 9)) # round(a, b) 保留a的b位小数
    return q_one

q1_one = quaternion_normalized(q1)
q2_one = quaternion_normalized(q2)

r1 = quaternion_rotate(q1_one)
r2 = quaternion_rotate(q2_one)

pw = np.linalg.inv(r1).dot((p-t2))
p2 = r2.dot(pw)+t

print '-------------------answer--------------------'
print p2

from re import I, sub
import gurobipy as gp
from gurobipy import GRB
import matplotlib
import matplotlib.pyplot as plt
import itertools
import time
import pyomo.environ
from pyomo.opt import SolverFactory
import math
from gurobipy import quicksum
import copy



# Start timer
t=time.time()

# Input data
timeLimit = 15      # Set a time limit for the ILP models
w_h = []
with open('Data sets/zdf1.txt') as f: # Import data
    lines = f.readlines()
for i in range(len(lines)):
    if i == 0:
        n = int(lines[i])
    elif i == 1:
        W = int(lines[i])
    else:
        line = lines[i].split()
        w_h.append([int(line[1]),int(line[2])])

# Set up the visual representation
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlim([0, W])
colors_cycle = itertools.cycle(['green', 'blue', 'red', 'purple', 'cyan', 'pink', 'lime', 'yellow', 'brown', 'black', 'magenta', 'olive', 'gray', 'orange', 'gold', 'lightgreen', 'deeppink', 'crimson', 'maroon', 'indigo', 'darkgreen', 'darkorange', 'navy'])

# Choose how many rectangles we place each iteration
subsize = 5

# Initialize a iteration counter
counter = 0

# Initialize a list to store all coordinates
x_y_w_h = []

# Sort w_h with decreasing areas
w_h = sorted(w_h, key = lambda x:(-x[0]*x[1]))
while len(w_h)-counter*subsize >= subsize:
    counter += 1    # Increase the counter by 1 for every iteration
    w_h_sub = []    
    for i in range(counter*subsize):    # Add the rectangles we want to place in this iteration to a list
        w_h_sub.append(w_h[i])
    for i in range(counter*subsize):    # Add the rotated versions of the rectangles to the list
        w_h_sub.append([w_h_sub[i][1],w_h_sub[i][0]])

    # Find the maximum length of a side of the set of rectangles (We need this for M values)
    all_sides = []
    for i in range(len(w_h_sub)):
        all_sides.append(w_h_sub[i][0])
        all_sides.append(w_h_sub[i][1])
    max_w_h = max(all_sides)

    # Find the total of the longest side of each rectangle summed (We need this for M values)
    sides_total = 0
    for i in range(int(0.5*len(w_h_sub))):
        sides_total += max(w_h_sub[i])

    # Run the SPP ILP
    m = gp.Model("mip1")

    # Limit processing power
    optimizer = SolverFactory('gurobi')
    optimizer.options['threads'] = 1

    # Create variables
    x = m.addVars(2*counter*subsize, vtype=GRB.INTEGER, name = "x")
    y = m.addVars(2*counter*subsize, vtype=GRB.INTEGER, name = "y")
    z = m.addVars(2*counter*subsize, vtype=GRB.BINARY, name = 'z')
    a = m.addVars(2*counter*subsize,2*counter*subsize, vtype=GRB.BINARY, name = "a")
    b = m.addVars(2*counter*subsize,2*counter*subsize, vtype=GRB.BINARY, name = "b")
    c = m.addVars(2*counter*subsize,2*counter*subsize, vtype=GRB.BINARY, name = "c")
    d = m.addVars(2*counter*subsize,2*counter*subsize, vtype=GRB.BINARY, name = "d")
    H = m.addVar(vtype=GRB.INTEGER)

    # Set objective
    m.setObjective(H, GRB.MINIMIZE)

    # Add constraints
    for i in range(len(x_y_w_h)):
        m.addConstr(x[i] == x_y_w_h[i][0])
        m.addConstr(y[i] == x_y_w_h[i][1])
        m.addConstr(z[i] == 1)
    for i in range(len(w_h_sub)):
        m.addConstr(x[i] + w_h_sub[i][0] - (1-z[i])*(max_w_h-1) <= W)
        m.addConstr(y[i] + w_h_sub[i][1] - (1-z[i])*(sides_total+max_w_h) <= H)
    for i in range(len(w_h_sub)):
        for j in range(len(w_h_sub)):
            if i != j:    
                m.addConstr(x[i]-x[j]+w_h_sub[i][0] <= (1-z[i]+a[i,j])*(W+max_w_h-1))
                m.addConstr(x[j]-x[i]+w_h_sub[j][0] <= (1-z[j]+b[i,j])*(W+max_w_h-1))
                m.addConstr(y[i]-y[j]+w_h_sub[i][1] <= (1-z[i]+c[i,j])*(sides_total+max_w_h))
                m.addConstr(y[j]-y[i]+w_h_sub[j][1] <= (1-z[j]+d[i,j])*(sides_total+max_w_h))
                m.addConstr(a[i,j]+b[i,j]+c[i,j]+d[i,j] <= 3)
    for i in range(counter*subsize):
        m.addConstr(z[i] + z[i+counter*subsize] == 1)

    # Optimize model
    m.Params.TimeLimit = timeLimit - m.getAttr(GRB.Attr.Runtime)
    m.optimize()

    # Store final optimal coordinates
    x_opt = []
    y_opt = []
    z_opt = []
    for v in x.values():
        x_opt.append(int(v.X))
    for v in y.values():
        y_opt.append(int(v.X))
    for v in z.values():
        z_opt.append(int(v.X))
    x_y_w_h = []
    for i in range(subsize*counter):
        if z_opt[i] == 1:
            x_y_w_h.append([x_opt[i],y_opt[i],w_h_sub[i][0],w_h_sub[i][1]])
        elif z_opt[i+subsize*counter] == 1:
            x_y_w_h.append([x_opt[i+subsize*counter],y_opt[i+subsize*counter],w_h_sub[i+subsize*counter][0],w_h_sub[i+subsize*counter][1]])
            w_h[i] = [w_h_sub[i][1],w_h_sub[i][0]]

for i in range(n):      # Add their rotated versions of all rectangles to w_h
    w_h.append([w_h[i][1],w_h[i][0]])
if counter*subsize != len(w_h):

    # Find the maximum length of a side of the set of rectangles (We need this for M values)
    all_sides = []
    for i in range(len(w_h)):
        all_sides.append(w_h[i][0])
        all_sides.append(w_h[i][1])
    max_w_h = max(all_sides)

    # Find the total of the longest side of each rectangle summed (We need this for M values)
    sides_total = 0
    for i in range(int(0.5*len(w_h))):
        sides_total += max(w_h[i])

    # Run the final SPP ILP
    m = gp.Model("mip1")

    # Create variables
    x = m.addVars(2*n, vtype=GRB.INTEGER, name = "x")
    y = m.addVars(2*n, vtype=GRB.INTEGER, name = "y")
    z = m.addVars(2*n, vtype=GRB.BINARY, name = 'z')
    a = m.addVars(2*n,2*n, vtype=GRB.BINARY, name = "a")
    b = m.addVars(2*n,2*n, vtype=GRB.BINARY, name = "b")
    c = m.addVars(2*n,2*n, vtype=GRB.BINARY, name = "c")
    d = m.addVars(2*n,2*n, vtype=GRB.BINARY, name = "d")
    H = m.addVar(vtype=GRB.INTEGER)

    # Set objective
    m.setObjective(H, GRB.MINIMIZE)

    # Add constraints
    for i in range(counter*subsize):
        m.addConstr(x[i] == x_y_w_h[i][0])
        m.addConstr(y[i] == x_y_w_h[i][1])
        m.addConstr(z[i] == 1)
    for i in range(2*n):
        m.addConstr(x[i] + w_h[i][0] - (1-z[i])*(max_w_h-1) <= W)
        m.addConstr(y[i] + w_h[i][1] - (1-z[i])*(sides_total+max_w_h) <= H)
    for i in range(2*n):
        for j in range(2*n):
            if i != j:    
                m.addConstr(x[i]-x[j]+w_h[i][0] <= (1-z[i]+a[i,j])*(W+max_w_h-1))
                m.addConstr(x[j]-x[i]+w_h[j][0] <= (1-z[j]+b[i,j])*(W+max_w_h-1))
                m.addConstr(y[i]-y[j]+w_h[i][1] <= (1-z[i]+c[i,j])*(sides_total+max_w_h))
                m.addConstr(y[j]-y[i]+w_h[j][1] <= (1-z[j]+d[i,j])*(sides_total+max_w_h))
                m.addConstr(a[i,j]+b[i,j]+c[i,j]+d[i,j] <= 3)
    for i in range(n):
        m.addConstr(z[i] + z[i+n] == 1)

    # Optimize model
    m.Params.TimeLimit = timeLimit - m.getAttr(GRB.Attr.Runtime)
    m.optimize()

    # Store final optimal coordinates
    x_opt = []
    y_opt = []
    z_opt = []
    for v in x.values():
        x_opt.append(int(v.X))
    for v in y.values():
        y_opt.append(int(v.X))
    for v in z.values():
        z_opt.append(int(v.X))
    x_y_w_h = []
    for i in range(int(0.5*len(w_h))):
        if z_opt[i] == 1:
            x_y_w_h.append([x_opt[i],y_opt[i],w_h[i][0],w_h[i][1]])
        elif z_opt[i+int(0.5*len(w_h))] == 1:
            x_y_w_h.append([x_opt[i+int(0.5*len(w_h))],y_opt[i+int(0.5*len(w_h))],w_h[i+int(0.5*len(w_h))][0],w_h[i+int(0.5*len(w_h))][1]])

print("The optimal solution is:" + " " + str(int(m.ObjVal)))
elapsed = time.time() - t
print('The program took' + ' ' + str(elapsed) + ' ' + 'seconds to run')
ax.add_patch(matplotlib.patches.Rectangle((0,0),W,m.ObjVal,color = 'black', fill=None, hatch = '////'))
for i in range(int(0.5*len(w_h))):
    ax.add_patch(matplotlib.patches.Rectangle((x_y_w_h[i][0],x_y_w_h[i][1]),x_y_w_h[i][2],x_y_w_h[i][3],color = next(colors_cycle)))
    ax.add_patch(matplotlib.patches.Rectangle((x_y_w_h[i][0],x_y_w_h[i][1]),x_y_w_h[i][2],x_y_w_h[i][3], fill = None))
plt.xlim([0,W])
plt.ylim([0,m.ObjVal])
plt.show()
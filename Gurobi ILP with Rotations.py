from re import I
import gurobipy as gp
from gurobipy import GRB
import matplotlib
import matplotlib.pyplot as plt
import itertools
import time
import pyomo.environ
from pyomo.opt import SolverFactory

# Limit processing power
optimizer = SolverFactory('gurobi')
optimizer.options['threads'] = 1

# Start timer
t=time.time()

# Input data
timeLimit = 60
w_h = []
with open('NGCUT07.txt') as f:
    lines = f.readlines()
for i in range(len(lines)):
    if i == 0:
        n = int(lines[i])
    elif i == 1:
        W = int(lines[i])
    else:
        line = lines[i].split()
        w_h.append([int(line[1]),int(line[2])])

for i in range(len(w_h)):
    sub = []
    sub.append(w_h[i][1])
    sub.append(w_h[i][0])
    w_h.append(sub)    
sum_h = 0
for i in range(len(w_h)):
    sum_h += w_h[i][1]

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

#Set up the visual representation
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlim([0, W])
colors_cycle = itertools.cycle(['green', 'blue', 'red', 'purple', 'cyan', 'pink', 'lime', 'yellow', 'brown', 'black', 'magenta', 'olive', 'gray', 'orange', 'gold', 'lightgreen', 'deeppink', 'crimson', 'maroon', 'indigo', 'darkgreen', 'darkorange', 'navy'])
try:

    # Create a new model
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
    for i in range(2*n):
        m.addConstr(x[i] + w_h[i][0] - (1-z[i])*(max_w_h-1) <= W)
        m.addConstr(y[i] + w_h[i][1] - (1-z[i])*(sides_total+max_w_h) <= H)
    for i in range(2*n):
        for j in range(2*n):
            if i != j:    
                m.addConstr(x[i]-x[j]+w_h[i][0] <= (W+max_w_h-1)*(1-z[i]+a[i,j]))
                m.addConstr(x[j]-x[i]+w_h[j][0] <= (W+max_w_h-1)*(1-z[j]+b[i,j]))
                m.addConstr(y[i]-y[j]+w_h[i][1] <= (sides_total+max_w_h)*(1-z[i]+c[i,j]))
                m.addConstr(y[j]-y[i]+w_h[j][1] <= (sides_total+max_w_h)*(1-z[j]+d[i,j]))
                m.addConstr(a[i,j]+b[i,j]+c[i,j]+d[i,j] <= 3)
    for i in range(n):
        m.addConstr(z[i] + z[i+n] == 1)

    # Optimize model
    m.Params.TimeLimit = timeLimit - m.getAttr(GRB.Attr.Runtime)
    m.optimize()

    # Create visual represenation 
    x_opt = []
    y_opt = []
    z_opt = []
    for v in x.values():
        x_opt.append(v.X)
    for v in y.values():
        y_opt.append(v.X)
    for v in z.values():
        z_opt.append(v.X)
    ax.add_patch(matplotlib.patches.Rectangle((0,0),W,m.ObjVal,color = 'black', fill=None, hatch = '////'))
    for i in range(2*n):
        ax.add_patch(matplotlib.patches.Rectangle((x_opt[i],y_opt[i]),z_opt[i]*w_h[i][0],z_opt[i]*w_h[i][1],color = next(colors_cycle)))
        ax.add_patch(matplotlib.patches.Rectangle((x_opt[i],y_opt[i]),z_opt[i]*w_h[i][0],z_opt[i]*w_h[i][1], fill = None))

    print('Obj: %g' % m.ObjVal)
    elapsed = time.time() - t
    print('The program took' + ' ' + str(elapsed) + ' ' + 'seconds to run')
    plt.ylim([0,m.ObjVal])
    plt.show()
except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')
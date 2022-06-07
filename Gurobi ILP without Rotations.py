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

#Start timer
t=time.time()

#Input data
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
sum_h = 0
for i in range(len(w_h)):
    sum_h += w_h[i][1]
n = len(w_h)

#Set up the visual representation
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlim([0, W])
plt.ylim([0, sum_h])
colors_cycle = itertools.cycle(['green', 'blue', 'red', 'purple', 'cyan', 'pink', 'lime', 'yellow', 'brown', 'black', 'magenta', 'olive', 'gray', 'orange', 'gold', 'lightgreen', 'deeppink', 'crimson', 'maroon', 'indigo', 'darkgreen', 'darkorange', 'navy'])
try:

    # Create a new model
    m = gp.Model("mip1")

    # Create variables
    x = m.addVars(n,vtype=GRB.INTEGER, name="x")
    y = m.addVars(n,vtype=GRB.INTEGER, name="y")
    a = m.addVars(n,n, vtype=GRB.BINARY, name = "a")
    b = m.addVars(n,n, vtype=GRB.BINARY, name = "b")
    c = m.addVars(n,n, vtype=GRB.BINARY, name = "c")
    d = m.addVars(n,n, vtype=GRB.BINARY, name = "d")
    H = m.addVar(vtype=GRB.INTEGER)
 
    # Set objective
    m.setObjective(H, GRB.MINIMIZE)

    # Add constraints
    for i in range(0,n):
        m.addConstr(x[i] + w_h[i][0] <= W)
        m.addConstr(y[i] + w_h[i][1] <= H)
    for i in range(0,n):
        for j in range(0,n):
            if i != j:    
                m.addConstr(x[i]-x[j]+w_h[i][0] <= 5000*a[i,j])
                m.addConstr(x[j]-x[i]+w_h[j][0] <= 5000*b[i,j])
                m.addConstr(y[i]-y[j]+w_h[i][1] <= 5000*c[i,j])
                m.addConstr(y[j]-y[i]+w_h[j][1] <= 5000*d[i,j])
                m.addConstr(a[i,j]+b[i,j]+c[i,j]+d[i,j] <= 3)
    # Optimize model
    m.Params.TimeLimit = timeLimit - m.getAttr(GRB.Attr.Runtime)
    m.optimize()

    # Create visual represenation 
    x_opt = []
    y_opt = []
    for v in x.values():
        x_opt.append(v.X)
    for v in y.values():
        y_opt.append(v.X)
    ax.add_patch(matplotlib.patches.Rectangle((0,0),W,m.ObjVal,color = 'black', fill=None, hatch = '////'))
    for i in range(0,n):
        ax.add_patch(matplotlib.patches.Rectangle((x_opt[i],y_opt[i]),w_h[i][0],w_h[i][1],color = next(colors_cycle)))
        ax.add_patch(matplotlib.patches.Rectangle((x_opt[i],y_opt[i]),w_h[i][0],w_h[i][1], fill = None))

    print('Obj: %g' % m.ObjVal)
    elapsed = time.time() - t
    print('The program took' + ' ' + str(elapsed) + ' ' + 'seconds to run')
    plt.ylim([0,m.ObjVal])
    plt.show()
except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')
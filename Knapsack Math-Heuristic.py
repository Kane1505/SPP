from re import I
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

# Limit processing power
optimizer = SolverFactory('gurobi')
optimizer.options['threads'] = 1

# Start timer
t=time.time()

# A function that rounds numbers to the nearest 5
def myround(x, base=5):
    return base * round(x/base)

# Input data
timeLimit = 30      # Set a time limit for the knapsacks
timeLimit_fin = 60  # Set a time limit for the final ILP
w_h = []
with open('Data sets/BENG09.txt') as f: # Import data
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
w_h_orig = copy.deepcopy(w_h)

# Set up the visual representation
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlim([0, W])
colors_cycle = itertools.cycle(['green', 'blue', 'red', 'purple', 'cyan', 'pink', 'lime', 'yellow', 'brown', 'black', 'magenta', 'olive', 'gray', 'orange', 'gold', 'lightgreen', 'deeppink', 'crimson', 'maroon', 'indigo', 'darkgreen', 'darkorange', 'navy'])

# Choose the size of the sub rectangles
W_sub = myround(0.5*W) 
H_sub = myround(0.5*W)
W_sub_big = myround(W)
H_sub_big = myround(0.5*W)

# Remove all rectangles from the list that are too large for the knapsack and add them to the final list
w_h_big = []  # We store the big rectangles in this list
delete = []
for i in range(len(w_h)):
    if w_h[i][0] > W_sub or w_h[i][1] > H_sub:
        if w_h[i][1] > W_sub or w_h[i][0] > H_sub:
            w_h_big.append(w_h[i])
            delete.append(i)
for i in range(len(delete)-1,-1,-1):
    w_h.pop(delete[i])

# Add profits to the rectangles
w_h_p = []
for i in range(len(w_h)):
    w_h_p.append([w_h[i][0],w_h[i][1],w_h[i][0]*w_h[i][1]])
w_h_p_big = []
for i in range(len(w_h_big)):
    w_h_p_big.append([w_h_big[i][0],w_h_big[i][1],w_h_big[i][0]*w_h_big[i][1]])

# Choose how many rectangles we consider for the knapsacks
areas = []
for i in range(len(w_h_p)):
    areas.append(w_h_p[i][2])
min_area = min(areas)
items = math.ceil((W_sub*H_sub)/min_area)

coordinates = [] # Start a list to save all coordinates of the subrectangles
counter = 0 # Start a counter of how many knapsacks there are
while len(w_h_p) > items:   # Pack all small items in knapsacks
    w_h_p_sub = []  # Start an empty list for each iteration
    counter += 1    # Increase the counter by one every iteration
    for i in range(items):  # Add the first "items" rectangles to the sublist
        w_h_p_sub.append(w_h_p[i])
    for i in range(items):  # Also add the rotated version
        w_h_p_sub.append([w_h_p[i][1],w_h_p_sub[i][0],w_h_p_sub[i][2]])

    # Find the maximum length of a side of the set of rectangles (We need this for M values)
    all_sides = []
    for i in range(len(w_h_p_sub)):
        all_sides.append(w_h_p_sub[i][0])
        all_sides.append(w_h_p_sub[i][1])
    max_w_h = max(all_sides)

    # Create a new model
    m = gp.Model("mip1")
    # Create variables
    x = m.addVars(2*items, vtype=GRB.INTEGER, name = "x")    # The optimal x-coordinates
    y = m.addVars(2*items, vtype=GRB.INTEGER, name = "y")    # The optimal y-coordinates
    z = m.addVars(2*items, vtype=GRB.BINARY, name = 'z')     # A variable that decides if the rotated or non-rotated version of the rectangle will be placed
    a = m.addVars(2*items,2*items, vtype=GRB.BINARY, name = "a")  # Variables a, b, c and d are used to make sure that at least one of four constraints is satisfied
    b = m.addVars(2*items,2*items, vtype=GRB.BINARY, name = "b")
    c = m.addVars(2*items,2*items, vtype=GRB.BINARY, name = "c")
    d = m.addVars(2*items,2*items, vtype=GRB.BINARY, name = "d")                  

    # Set objective
    m.setObjective(quicksum(w_h_p_sub[i][2]*z[i] for i in range(2*items)), GRB.MAXIMIZE)     # Our objective is to maximize the profit in each knapsack

    # Set constraints
    for i in range(2*items):
        m.addConstr(x[i] + w_h_p_sub[i][0] <= W_sub + (max_w_h-1)*(1-z[i]))    # The width of the KS cannot be exceeded
        m.addConstr(y[i] + w_h_p_sub[i][1]<= H_sub + (max_w_h-1)*(1-z[i])) # The height of the KS cannot be exceeded

    for i in range(2*items):
        for j in range(2*items):
            if i != j:
                m.addConstr(x[i]-x[j]+w_h_p_sub[i][0] <= (1-z[i]+a[i,j])*(W_sub+max_w_h-1))
                m.addConstr(x[j]-x[i]+w_h_p_sub[j][0] <= (1-z[j]+b[i,j])*(W_sub+max_w_h-1))
                m.addConstr(y[i]-y[j]+w_h_p_sub[i][1] <= (1-z[i]+c[i,j])*(H_sub+max_w_h-1))
                m.addConstr(y[j]-y[i]+w_h_p_sub[j][1] <= (1-z[j]+d[i,j])*(H_sub+max_w_h-1))
                # This constraint makes sure at least one of the four constraints above holds
                m.addConstr(a[i,j]+b[i,j]+c[i,j]+d[i,j] <= 3)
    for i in range(items):
        m.addConstr(z[i]+z[i+items] <= 1)

    m.Params.TimeLimit = timeLimit - m.getAttr(GRB.Attr.Runtime)
    m.optimize()

    x_opt = []
    y_opt = []
    z_opt = []
    for v in x.values():
        x_opt.append(int(v.X))
    for v in y.values():
        y_opt.append(int(v.X))
    for v in z.values():
        z_opt.append(int(v.X))

    coordinates_sub = []
    delete = []
    for i in range(items):
        if z_opt[i] == 1:
            coordinates_sub.append([x_opt[i],y_opt[i],w_h_p_sub[i][0],w_h_p_sub[i][1]])
            delete.append(i)
        if z_opt[i+items]:
            coordinates_sub.append([x_opt[i+items],y_opt[i+items],w_h_p_sub[i+items][0],w_h_p_sub[i+items][1]])
            delete.append(i)
    for i in range(len(delete)-1,-1,-1):
        w_h_p.pop(delete[i])
    coordinates.append(coordinates_sub)
if counter!=0:
    small_knapsacks = [[W_sub,H_sub]]*counter # Construct rectangles to represent the knapsacks
leftovers = []  # Start a list to save all leftover rectangles
for i in range(len(w_h_p)):
    leftovers.append([w_h_p[i][0],w_h_p[i][1]])


counter_big = 0
while len(w_h_p_big) > items:   # Pack all small items in knapsacks
    w_h_p_sub = []  # Start an empty list for each iteration
    counter_big += 1    # Increase the counter by one every iteration
    for i in range(items):  # Add the first "items" rectangles to the sublist
        w_h_p_sub.append(w_h_p_big[i])
    for i in range(items):  # Also add the rotated version
        w_h_p_sub.append([w_h_p_big[i][1],w_h_p_sub[i][0],w_h_p_sub[i][2]])
    
        # Find the maximum length of a side of the set of rectangles (We need this for M values)
    all_sides = []
    for i in range(len(w_h_p_sub)):
        all_sides.append(w_h_p_sub[i][0])
        all_sides.append(w_h_p_sub[i][1])
    max_w_h = max(all_sides)

    # Create a new model
    m = gp.Model("mip1")
    # Create variables
    x = m.addVars(2*items, vtype=GRB.INTEGER, name = "x")    # The optimal x-coordinates
    y = m.addVars(2*items, vtype=GRB.INTEGER, name = "y")    # The optimal y-coordinates
    z = m.addVars(2*items, vtype=GRB.BINARY, name = 'z')     # A variable that decides if the rotated or non-rotated version of the rectangle will be placed
    a = m.addVars(2*items,2*items, vtype=GRB.BINARY, name = "a")  # Variables a, b, c and d are used to make sure that at least one of four constraints is satisfied
    b = m.addVars(2*items,2*items, vtype=GRB.BINARY, name = "b")
    c = m.addVars(2*items,2*items, vtype=GRB.BINARY, name = "c")
    d = m.addVars(2*items,2*items, vtype=GRB.BINARY, name = "d")                  

    # Set objective
    m.setObjective(quicksum(w_h_p_sub[i][2]*z[i] for i in range(2*items)), GRB.MAXIMIZE)     # Our objective is to maximize the profit in each knapsack

    # Set constraints
    for i in range(2*items):
        m.addConstr(x[i] + w_h_p_sub[i][0] <= W_sub_big + (max_w_h-1)*(1-z[i]))    # The width of the KS cannot be exceeded
        m.addConstr(y[i] + w_h_p_sub[i][1]<= H_sub_big + (max_w_h-1)*(1-z[i])) # The height of the KS cannot be exceeded

    for i in range(2*items):
        for j in range(2*items):
            if i != j:
                m.addConstr(x[i]-x[j]+w_h_p_sub[i][0] <= (1-z[i]+a[i,j])*(W_sub_big+max_w_h-1))
                m.addConstr(x[j]-x[i]+w_h_p_sub[j][0] <= (1-z[j]+b[i,j])*(W_sub_big+max_w_h-1))
                m.addConstr(y[i]-y[j]+w_h_p_sub[i][1] <= (1-z[i]+c[i,j])*(H_sub_big+max_w_h-1))
                m.addConstr(y[j]-y[i]+w_h_p_sub[j][1] <= (1-z[j]+d[i,j])*(H_sub_big+max_w_h-1))
                # This constraint makes sure at least one of the four constraints above holds
                m.addConstr(a[i,j]+b[i,j]+c[i,j]+d[i,j] <= 3)
    for i in range(items):
        m.addConstr(z[i]+z[i+items] <= 1)

    m.Params.TimeLimit = timeLimit - m.getAttr(GRB.Attr.Runtime)
    m.optimize()

    x_opt = []
    y_opt = []
    z_opt = []
    for v in x.values():
        x_opt.append(int(v.X))
    for v in y.values():
        y_opt.append(int(v.X))
    for v in z.values():
        z_opt.append(int(v.X))

    coordinates_sub = []
    delete = []
    for i in range(items):
        if z_opt[i] == 1:
            coordinates_sub.append([x_opt[i],y_opt[i],w_h_p_sub[i][0],w_h_p_sub[i][1]])
            delete.append(i)
        if z_opt[i+items]:
            coordinates_sub.append([x_opt[i+items],y_opt[i+items],w_h_p_sub[i+items][0],w_h_p_sub[i+items][1]])
            delete.append(i)
    for i in range(len(delete)-1,-1,-1):
        w_h_p_big.pop(delete[i])    
    coordinates.append(coordinates_sub)
if counter_big!=0:
    big_knapsacks = [[W_sub_big,H_sub_big]]*counter_big # Construct rectangles to represent the knapsacks
for i in range(len(w_h_p_big)):
    leftovers.append([w_h_p_big[i][0],w_h_p_big[i][1]])
if counter_big != 0 and counter != 0:
    all_knapsacks = small_knapsacks + big_knapsacks
elif counter_big == 0 and counter != 0:
    all_knapsacks = small_knapsacks
elif counter_big != 0 and counter == 0:
    all_knapsacks = big_knapsacks
else:
    all_knapsacks = []

num = len(all_knapsacks)    # Store the length of this list
for i in range(num):        # Add the rotated versions of the knapsacks to the list
    all_knapsacks.append([all_knapsacks[i][1],all_knapsacks[i][0]])

# Find the maximum length of a side of the set of rectangles (We need this for M values)
all_sides = []
for i in range(len(all_knapsacks)):
    all_sides.append(all_knapsacks[i][0])
    all_sides.append(all_knapsacks[i][1])
max_w_h = max(all_sides)

# Find the total of the longest side of each rectangle summed (We need this for M values)
sides_total = 0
for i in range(int(0.5*len(all_knapsacks))):
    sides_total += max(all_knapsacks[i])

# Run the SPP ILP
m = gp.Model("mip1")

# Create variables
x = m.addVars(2*num, vtype=GRB.INTEGER, name = "x")
y = m.addVars(2*num, vtype=GRB.INTEGER, name = "y")
z = m.addVars(2*num, vtype=GRB.BINARY, name = 'z')
a = m.addVars(2*num,2*num, vtype=GRB.BINARY, name = "a")
b = m.addVars(2*num,2*num, vtype=GRB.BINARY, name = "b")
c = m.addVars(2*num,2*num, vtype=GRB.BINARY, name = "c")
d = m.addVars(2*num,2*num, vtype=GRB.BINARY, name = "d")
H = m.addVar(vtype=GRB.INTEGER)

# Set objective
m.setObjective(H, GRB.MINIMIZE)

# Add constraints
for i in range(2*num):
    m.addConstr(x[i] + all_knapsacks[i][0] - (1-z[i])*(max_w_h-1) <= W)
    m.addConstr(y[i] + all_knapsacks[i][1] - (1-z[i])*(sides_total+max_w_h) <= H)
for i in range(2*num):
    for j in range(2*num):
        if i != j:    
            m.addConstr(x[i]-x[j]+all_knapsacks[i][0] <= (1-z[i]+a[i,j])*(W+max_w_h-1))
            m.addConstr(x[j]-x[i]+all_knapsacks[j][0] <= (1-z[j]+b[i,j])*(W+max_w_h-1))
            m.addConstr(y[i]-y[j]+all_knapsacks[i][1] <= (1-z[i]+c[i,j])*(sides_total+max_w_h))
            m.addConstr(y[j]-y[i]+all_knapsacks[j][1] <= (1-z[j]+d[i,j])*(sides_total+max_w_h))
            m.addConstr(a[i,j]+b[i,j]+c[i,j]+d[i,j] <= 3)
for i in range(num):
    m.addConstr(z[i] + z[i+num] == 1)

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
x_y_w_h = [] # Start a list to store all coordinates
for i in range(num):
    for j in range(len(coordinates[i])):
        if z_opt[i] == 1:
            x_y_w_h.append([x_opt[i]+coordinates[i][j][0],y_opt[i]+coordinates[i][j][1],coordinates[i][j][2],coordinates[i][j][3]])
        if z_opt[i+num] == 1:
            x_y_w_h.append([x_opt[i+num]+coordinates[i][j][1],y_opt[i+num]+coordinates[i][j][0],coordinates[i][j][3],coordinates[i][j][2]])

w_h_final = []
for i in range(len(x_y_w_h)):
    w_h_final.append([x_y_w_h[i][2],x_y_w_h[i][3]])
w_h_final = w_h_final + leftovers
num = len(w_h_final) # Store the number of leftover rectangles
for i in range(num):
    w_h_final.append([w_h_final[i][1],w_h_final[i][0]])

# Find the maximum length of a side of the set of rectangles (We need this for M values)
all_sides = []
for i in range(len(w_h_final)):
    all_sides.append(w_h_final[i][0])
    all_sides.append(w_h_final[i][1])
max_w_h = max(all_sides)

# Find the total of the longest side of each rectangle summed (We need this for M values)
sides_total = 0
for i in range(int(0.5*len(w_h_final))):
    sides_total += max(w_h_final[i])

# Add the leftover rectangles to the strip with the knapsacks
m = gp.Model("mip1")

# Create variables
x = m.addVars(2*num, vtype=GRB.INTEGER, name = "x")
y = m.addVars(2*num, vtype=GRB.INTEGER, name = "y")
z = m.addVars(2*num, vtype=GRB.BINARY, name = 'z')
a = m.addVars(2*num,2*num, vtype=GRB.BINARY, name = "a")
b = m.addVars(2*num,2*num, vtype=GRB.BINARY, name = "b")
c = m.addVars(2*num,2*num, vtype=GRB.BINARY, name = "c")
d = m.addVars(2*num,2*num, vtype=GRB.BINARY, name = "d")
H = m.addVar(vtype=GRB.INTEGER)

# Set objective
m.setObjective(H, GRB.MINIMIZE)

# Add constraints
for i in range(len(x_y_w_h)):
    m.addConstr(x[i] == x_y_w_h[i][0])
    m.addConstr(y[i] == x_y_w_h[i][1])
    m.addConstr(z[i] == 1)
for i in range(2*num):
    m.addConstr(x[i] + w_h_final[i][0] - (1-z[i])*(max_w_h-1) <= W)
    m.addConstr(y[i] + w_h_final[i][1] - (1-z[i])*(sides_total+max_w_h) <= H)
for i in range(2*num):
    for j in range(2*num):
        if i != j:   
            m.addConstr(x[i]-x[j]+w_h_final[i][0] <= (1-z[i]+a[i,j])*(W+max_w_h-1))
            m.addConstr(x[j]-x[i]+w_h_final[j][0] <= (1-z[j]+b[i,j])*(W+max_w_h-1))
            m.addConstr(y[i]-y[j]+w_h_final[i][1] <= (1-z[i]+c[i,j])*(sides_total+max_w_h))
            m.addConstr(y[j]-y[i]+w_h_final[j][1] <= (1-z[j]+d[i,j])*(sides_total+max_w_h))
            m.addConstr(a[i,j]+b[i,j]+c[i,j]+d[i,j] <= 3)
for i in range(num):
    m.addConstr(z[i] + z[i+num] == 1)

# Optimize model
m.Params.TimeLimit = timeLimit_fin - m.getAttr(GRB.Attr.Runtime)
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

x_y_w_h_fin = []    # Create a list to store the final coordinates
for i in range(num):
    if z_opt[i] == 1:
        x_y_w_h_fin.append([x_opt[i],y_opt[i],w_h_final[i][0],w_h_final[i][1]])
    if z_opt[i+num] == 1:
        x_y_w_h_fin.append([x_opt[i+num],y_opt[i+num],w_h_final[i+num][0],w_h_final[i+num][1]])



print("The optimal solution is:" + " " + str(int(m.ObjVal)))
elapsed = time.time() - t
print('The program took' + ' ' + str(elapsed) + ' ' + 'seconds to run')
ax.add_patch(matplotlib.patches.Rectangle((0,0),W,sum_h,color = 'black', fill=None, hatch = '////'))
for i in range(num):
    ax.add_patch(matplotlib.patches.Rectangle((x_y_w_h_fin[i][0],x_y_w_h_fin[i][1]),x_y_w_h_fin[i][2],x_y_w_h_fin[i][3],color = next(colors_cycle)))
    ax.add_patch(matplotlib.patches.Rectangle((x_y_w_h_fin[i][0],x_y_w_h_fin[i][1]),x_y_w_h_fin[i][2],x_y_w_h_fin[i][3], fill = None))
plt.xlim([0,W])
plt.ylim([0,m.ObjVal])
plt.show()
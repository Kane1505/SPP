from re import A
import matplotlib
import matplotlib.pyplot as plt
import itertools
from numpy import delete
import time
def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

# Start timer
t=time.time()

# Input data
w_h = []
with open('Data sets/zdf6.TXT') as f:
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

w_h = sorted(w_h, key = lambda x:(-x[1]))

# Set up visual representation
colors_cycle = itertools.cycle(['green', 'blue', 'red', 'purple', 'cyan', 'pink', 'lime', 'yellow', 'brown', 'black', 'magenta', 'olive', 'gray', 'orange', 'gold', 'lightgreen', 'deeppink', 'crimson', 'maroon', 'indigo', 'darkgreen', 'darkorange', 'navy'])
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlim([0, W])
plt.ylim([0,sum_h])
ax.add_patch(matplotlib.patches.Rectangle((0,0),W,sum_h,color = 'black', fill=None, hatch = '////'))

x_y_w_h = []    # Start a list to store the x and y coordinates with corresponding width and height of all rectangles
cornerpoints = [[0,0]]  # Start a list to store the available cornerpoints

for i in range(n):
    cornerpoints = sorted(cornerpoints, key = lambda x:(x[1],x[0]))     #Sort the cornerpoints, lowest first, then most left
    for j in range(len(cornerpoints)):
        x_coordinates = [W]         
        y_coordinates = [sum_h]
        if i != 0:
            for k in range(i):
                height_k = [x_y_w_h[k][1],x_y_w_h[k][1] + w_h[k][1]]
                # If an already placed rectangle covers the same height as the new one will and is placed to the right of the new rectangle, limit the allowed width of the new rectangle
                if getOverlap(height_k, [cornerpoints[j][1], cornerpoints[j][1] + w_h[i][1]]) != 0:
                    if x_y_w_h[k][0] >= cornerpoints[j][0]:      
                        x_coordinates.append(x_y_w_h[k][0])
                # If an already placed rectangle covers the same width as the new one will and is placed on top of the new one rectangle, limit the allowed height of the new rectangle
                if getOverlap([cornerpoints[j][0],cornerpoints[j][0] + w_h[i][0]], [x_y_w_h[k][0], x_y_w_h[k][0]+w_h[k][0]]) != 0:    # Check how much vertical space there is  
                    if x_y_w_h[k][1] >= cornerpoints[j][1]:
                        y_coordinates.append(x_y_w_h[k][1])
            x_lim = min(x_coordinates)  # The amount of horizontal space
            y_lim = min(y_coordinates)  # The amount of vertical space  
            if cornerpoints[j][0] + w_h[i][0] <= x_lim and cornerpoints[j][1] + w_h[i][1] <= y_lim: # Check if the rectangle fits
                x_y_w_h.append([cornerpoints[j][0],cornerpoints[j][1],w_h[i][0],w_h[i][1]]) # Add the coordinates and widths and heights to the list
                ax.add_patch(matplotlib.patches.Rectangle((x_y_w_h[i][0],x_y_w_h[i][1]),x_y_w_h[i][2],x_y_w_h[i][3],color = next(colors_cycle)))
                ax.add_patch(matplotlib.patches.Rectangle((x_y_w_h[i][0],x_y_w_h[i][1]),x_y_w_h[i][2],x_y_w_h[i][3], fill = None))
                cornerpoints.pop(j) # Remove the used cornerpoint
                covered = [0]*w_h[i][0]             # Create a list to see how much of the new rectangle is covered
                for k in range(i):                  # Check for every rectangle that has already been placed...
                    if x_y_w_h[k][1] == x_y_w_h[i][1] + w_h[i][1]:    # ... That has the potential to cover the new rectangle ...
                        for l in range(x_y_w_h[i][0],x_y_w_h[i][0]+w_h[i][0]):
                            for m in range(x_y_w_h[k][0],x_y_w_h[k][0]+w_h[k][0]):
                                if l == m:          # ... If the new rectangle is covered             
                                    covered[l-x_y_w_h[i][0]] = 1
                # If the top of the rectangle if fully covered, we omit the top left cornerpoint  
                if sum(covered) != w_h[i][0] and sum(covered) != 0: # If the top of the rectangle is partially covered, we use the first open space to place the new cornerpoint
                    first_zero_index = covered.index(0)
                    cornerpoints.append([first_zero_index+x_y_w_h[i][0],x_y_w_h[i][1]+w_h[i][1]])
                elif sum(covered) == 0: # If the top of the rectangle is not covered we add a cornerpoint on the height of the new rectangle, as far left as possible
                    left_neighbour_x = [0]
                    for k in range(i):
                        if x_y_w_h[k][1] <= x_y_w_h[i][1] + w_h[i][1] and x_y_w_h[k][1] + w_h[k][1] > x_y_w_h[i][1] + w_h[i][1]:
                            if x_y_w_h[k][0] + w_h[k][0] <= x_y_w_h[i][0]:
                                left_neighbour_x.append(x_y_w_h[k][0]+w_h[k][0])
                    closest_left_neighbour = max(left_neighbour_x)
                    cornerpoints.append([closest_left_neighbour,x_y_w_h[i][1]+w_h[i][1]])
                if covered[0] == 0: # If there is space on the top left corner of the new rectangle, also add that point as a cornerpoint
                    cornerpoints.append([x_y_w_h[i][0],x_y_w_h[i][1]+w_h[i][1]])
                
                covered = [0]*w_h[i][1]     # We now check if the right side of the rectangle is covered
                for k in range(i):                  # Check for every rectangle that has already been placed...
                    if x_y_w_h[k][0] == x_y_w_h[i][0] + w_h[i][0]:    # ... That has the potential to cover the new rectangle ...
                        for l in range(x_y_w_h[i][1],x_y_w_h[i][1]+w_h[i][1]):
                            for m in range(x_y_w_h[k][1],x_y_w_h[k][1]+w_h[k][1]):
                                if l == m:          # ... If the new rectangle is covered             
                                    covered[l-x_y_w_h[i][1]] = 1
                # If the right side of the rectangle is fully covered, we omit the bottom right cornerpoint
                if sum(covered) != w_h[i][1] and sum(covered) != 0:    # If the bottom right corner of the rectangle is partially covered, we use the first open space to place the new cornerpoint
                    first_zero_index = covered.index(0)
                    cornerpoints.append([x_y_w_h[i][0]+w_h[i][0],first_zero_index+x_y_w_h[i][1]])
                elif sum(covered) == 0: # If the right side of the rectangle is not covered we add a cornerpoint on the width of the new rectangle, as far down as possible
                    bottom_neighbour_y = [0]
                    for k in range(i):
                        if x_y_w_h[k][0] <= x_y_w_h[i][0] + w_h[i][0] and x_y_w_h[k][0] + w_h[k][0] > x_y_w_h[i][0] + w_h[i][0]:
                            if x_y_w_h[k][1] + w_h[k][1] <= x_y_w_h[i][1]:
                                bottom_neighbour_y.append(x_y_w_h[k][1]+w_h[k][1])
                    closest_bottom_neighbour = max(bottom_neighbour_y)
                    cornerpoints.append([x_y_w_h[i][0] + w_h[i][0],closest_bottom_neighbour])
                if covered[0] == 0: # If there is space on the bottom right corner of the new rectangle, also add that point as a cornerpoint
                    cornerpoints.append([x_y_w_h[i][0]+w_h[i][0],x_y_w_h[i][1]])
                # Remove the cornerpoints that are covered by the new rectangle
                delete_index = []   # Store the index of all cornerpoints that have to be removed
                for k in range(len(cornerpoints)):  # First find the cornerpoints that have been blocked by the vertical side of the new rectangle
                    if x_y_w_h[i][0] == cornerpoints[k][0]:
                        if x_y_w_h[i][1] <= cornerpoints[k][1] and x_y_w_h[i][1] + w_h[i][1] > cornerpoints[k][1]:
                            delete_index.append(k)
                if len(delete_index) != 0:  # Remove the cornerpoints that have been vertically covered
                    for k in range(len(delete_index)):
                        cornerpoints.pop(delete_index[k]-k)
                delete_index = []   # Clear the list, to do the same procedure for the horizontally blocked cornerpoints
                for k in range(len(cornerpoints)):  # First find the cornerpoints that have been blocked by the horizontal side of the new rectangle
                    if x_y_w_h[i][1] == cornerpoints[k][1]:
                        if x_y_w_h[i][0] <= cornerpoints[k][0] and x_y_w_h[i][0] + w_h[i][0] > cornerpoints[k][0]:
                            delete_index.append(k)
                if len(delete_index) != 0:  # Remove the cornerpoints that have been vertically covered
                    for k in range(len(delete_index)):
                        cornerpoints.pop(delete_index[k]-k)
                cornerpoints_new = []
                for elem in cornerpoints:               # Remove duplicates cornerpoints
                    if elem not in cornerpoints_new:
                        cornerpoints_new.append(elem)
                cornerpoints = cornerpoints_new.copy()
                break                                   # Move onto the next rectangle
        else:
            # Add the first rectangle to the bottom left corner
            x_y_w_h.append([0,0,w_h[i][0],w_h[i][1]])   # Store the coordinates and width and length of the rectangle
            ax.add_patch(matplotlib.patches.Rectangle((x_y_w_h[i][0],x_y_w_h[i][1]),x_y_w_h[i][2],x_y_w_h[i][3],color = next(colors_cycle)))
            ax.add_patch(matplotlib.patches.Rectangle((x_y_w_h[i][0],x_y_w_h[i][1]),x_y_w_h[i][2],x_y_w_h[i][3], fill = None))
            cornerpoints.pop(j)     # Remove the used cornerpoint
            cornerpoints.append([x_y_w_h[i][0]+w_h[i][0],x_y_w_h[i][1]])  # Add the new cornerpoints
            cornerpoints.append([x_y_w_h[i][0],x_y_w_h[i][1]+w_h[i][1]])

# Stop timer
elapsed = time.time() - t
print('The program took' + ' ' + str(elapsed) + ' ' + 'seconds to run')

# Show the height of the strip
heights = []
for i in range(n):
    heights.append(x_y_w_h[i][1]+w_h[i][1])
height = max(heights)

print('The height of the strip is' + ' ' + str(height))
# Add the rectangles to a figure and plot this figure
for i in range(len(x_y_w_h)):
    ax.add_patch(matplotlib.patches.Rectangle((x_y_w_h[i][0],x_y_w_h[i][1]),x_y_w_h[i][2],x_y_w_h[i][3],color = next(colors_cycle)))
    ax.add_patch(matplotlib.patches.Rectangle((x_y_w_h[i][0],x_y_w_h[i][1]),x_y_w_h[i][2],x_y_w_h[i][3], fill = None))
plt.ylim([0, height])
plt.show()
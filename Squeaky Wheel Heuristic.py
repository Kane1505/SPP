from calendar import c
from operator import index
from re import A, I
import matplotlib
import matplotlib.pyplot as plt
import itertools
from numpy import delete
import time
import copy
import math

# Start timer
t=time.time()

# Import data
w_h = []
with open('Data sets/zdf10.TXT') as f:
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

# Rotate the rectangles such that width >= height
for i in range(n):
    if w_h[i][0] < w_h[i][1]:
        old_w = w_h[i][0]
        old_h = w_h[i][1]
        w_h[i][0] = old_h
        w_h[i][1] = old_w

# Add a penalty point counter to the rectangles
for i in range(n):
    w_h[i].append(0)
# Set up visual representation
colors_cycle = itertools.cycle(['green', 'blue', 'red', 'purple', 'cyan', 'pink', 'lime', 'yellow', 'brown', 'black', 'magenta', 'olive', 'gray', 'orange', 'gold', 'lightgreen', 'deeppink', 'crimson', 'maroon', 'indigo', 'darkgreen', 'darkorange', 'navy'])
fig = plt.figure()
plt.xlim([0, W])
ax = fig.add_subplot(111)

# Set up parameters
time_limit = 300        # Time after which the program terminates
iteration_limit = 10000000  # Number of iterations after which the program terminates

# Choose a target height
sum = 0
for i in range(n):
    sum += w_h[i][0] * w_h[i][1]
target_height = round(sum/W)

# Initialize a max_height_tot for later
max_height_tot = sum_h

# Initialize iteration counter
iteration = 0

penalty_points = n*[0]


# Place the rectangles according to Best Fit (Tallest Neighbour variant)
while time.time() - t <= time_limit and iteration <= iteration_limit:
    sheet = [0] * W     # Intialize a sheet to store heights
    x_y_w_h_p = []        # Intialize a list to store coordinates
    iteration += 1
    print("Iteration:" + " " + str(iteration))
    print("Best solution:" + " " + str(max_height_tot))
    w_h = sorted(w_h, key = lambda x:(-x[2],-x[0],-x[1]))
    while len(w_h) != 0:    # While items are not packed
        index_min = min(range(len(sheet)), key=sheet.__getitem__) # Find the index of the lowest gap
        min_height = min(sheet) # Find the height of the lowest gap
        len_gap = 0
        current_best = -1
        for i in range(len(sheet) - index_min):     # Find the width of the lowest gap
            if sheet[index_min + i] == min_height:
                len_gap += 1
            else:
                break
        if index_min == 0:  # If the gap is against the left wall
            for i in range(len(w_h)):
                if w_h[i][0] <= len_gap and w_h[i][2] > current_best: # Check if rectangle i fits and is has more penalty points than our current best 
                    current_best = w_h[i][2]
                    current_best_index = i
                    current_best_or = 0
                    continue
                elif w_h[i][1] <= len_gap and w_h[i][2] > current_best: # Check if the rotated version of rectangle i fits and if it has more penalty points than our current best
                    current_best = w_h[i][2]
                    current_best_index = i
                    current_best_or = 1
                    continue
                elif w_h[i][2] <= current_best:   # If the current rectangle has less penalty points than our current best, place the current best 
                    break
            if current_best == -1:   # If no rectangle fits, raise the gap to the height of the rectangle on the right
                for j in range(len_gap):
                    sheet[index_min + j] = sheet[index_min + len_gap]
            elif current_best_or == 1:
                rec_height = sheet[index_min] + w_h[current_best_index][0]
                p_incr = max(0,rec_height - target_height)
                p = w_h[current_best_index][2] + p_incr
                x_y_w_h_p.append([len_gap - w_h[current_best_index][1],sheet[index_min],w_h[current_best_index][1],w_h[current_best_index][0],p])   # Store coordinates of the rectangle that has been placed
                for j in range(w_h[current_best_index][1]): # Update the sheet
                    sheet[index_min + len_gap - 1 - j] += w_h[current_best_index][0]                
                w_h.pop(current_best_index) # Remove placed rectangle from the list
            elif current_best_or == 0:
                rec_height = sheet[index_min] + w_h[current_best_index][1]
                p_incr = max(0,rec_height - target_height)
                p = w_h[current_best_index][2] + p_incr
                x_y_w_h_p.append([len_gap - w_h[current_best_index][0],sheet[index_min],w_h[current_best_index][0],w_h[current_best_index][1],p]) # Store coordinates of the rectangle that has been placed
                for j in range(w_h[current_best_index][0]): # Update the sheet
                    sheet[index_min + len_gap - 1 - j] += w_h[current_best_index][1]                
                w_h.pop(current_best_index) # Remove placed rectangle from the list
        elif index_min + len_gap == W:  # If the lowest gap reaches till the right side of the strip:
            for i in range(len(w_h)):
                if w_h[i][0] <= len_gap and w_h[i][2] > current_best:   # Check if rectangle i fits and is has more penalty points than our current best 
                    current_best = w_h[i][2]
                    current_best_index = i
                    current_best_or = 0
                    continue
                elif w_h[i][1] <= len_gap and w_h[i][2] > current_best: # Check if rectangle i fits and is has more penalty points than our current best 
                    current_best = w_h[i][2]
                    current_best_index = i
                    current_best_or = 1
                    continue
                elif w_h[i][2] <= current_best: # If the current rectangle has less penalty points than our current best, place the current best
                    break
            if current_best == -1:   # If no rectangle fits, raise the gap to the left neighbour
                for j in range(len_gap):
                    sheet[index_min + j] = sheet[index_min - 1]
            elif current_best_or == 1:
                rec_height = sheet[index_min] + w_h[current_best_index][0]
                p_incr = max(0,rec_height - target_height)
                p = w_h[current_best_index][2] + p_incr
                x_y_w_h_p.append([index_min,sheet[index_min],w_h[current_best_index][1],w_h[current_best_index][0],p])  # Store the coordinates of the rectangle that has been placed
                for j in range(w_h[current_best_index][1]): # Update sheet height
                    sheet[index_min + j] += w_h[current_best_index][0]                
                w_h.pop(current_best_index) # Remove placed rectangle from the list of rectangles
            elif current_best_or == 0:
                rec_height = sheet[index_min] + w_h[current_best_index][1]
                p_incr = max(0,rec_height - target_height)
                p = w_h[current_best_index][2] + p_incr
                x_y_w_h_p.append([index_min,sheet[index_min],w_h[current_best_index][0],w_h[current_best_index][1],p])  # Store the coordinates of the rectangle that has been placed
                for j in range(w_h[current_best_index][0]): # Update sheet height
                    sheet[index_min + j] += w_h[current_best_index][1]                
                w_h.pop(current_best_index) # Remove placed rectangle from the list of rectangles
        else:   # If the gap is not against one of the edges of the strip
            neighbour_1 = sheet[index_min - 1]  # Find the height of the left neighbour
            neighbour_2 = sheet[index_min + len_gap]    # Find the height of the right neighbour
            if neighbour_1 >= neighbour_2:  # Store the indices of the shortest and tallest neighbour
                tallest_neighbour = 1
                tallest_neighbour_index = index_min - 1
                shortest_neighbour_index = index_min + len_gap
            else:
                tallest_neighbour = 2
                tallest_neighbour_index = index_min + len_gap
                shortest_neighbour_index = index_min - 1
            if tallest_neighbour == 1:  # If the tallest neighbour is on the left:
                for i in range(len(w_h)):
                    if w_h[i][0] <= len_gap and w_h[i][2] > current_best: # Check if rectangle i fits and is has more penalty points than our current best 
                        current_best = w_h[i][2]
                        current_best_index = i
                        current_best_or = 0
                        continue
                    elif w_h[i][1] <= len_gap and w_h[i][2] > current_best: # Check if rectangle i fits and is has more penalty points than our current best 
                        current_best = w_h[i][2]
                        current_best_index = i
                        current_best_or = 1
                        continue
                    elif w_h[i][2] <= current_best: # If the current rectangle has less penalty points than our current best, place the current best
                        break
                if current_best == -1:   # If no rectangle fits, raise the gap to the shortest neighbour
                    for j in range(len_gap):
                        sheet[index_min + j] = sheet[index_min + len_gap]
                elif current_best_or == 1:
                    rec_height = sheet[index_min] + w_h[current_best_index][0]
                    p_incr = max(0,rec_height - target_height)
                    p = w_h[current_best_index][2] + p_incr
                    x_y_w_h_p.append([index_min,sheet[index_min],w_h[current_best_index][1],w_h[current_best_index][0],p]) # Store coordinates of placed rectangle
                    for j in range(w_h[current_best_index][1]): # Update sheet height
                        sheet[index_min + j] += w_h[current_best_index][0]                
                    w_h.pop(current_best_index) # Remove rectangle from list of rectangles
                elif current_best_or == 0:
                    rec_height = sheet[index_min] + w_h[current_best_index][1]
                    p_incr = max(0,rec_height - target_height)
                    p = w_h[current_best_index][2] + p_incr
                    x_y_w_h_p.append([index_min,sheet[index_min],w_h[current_best_index][0],w_h[current_best_index][1],p]) # Store coordinates of placed rectangle
                    for j in range(w_h[current_best_index][0]): # Update sheet height
                        sheet[index_min + j] += w_h[current_best_index][1]                
                    w_h.pop(current_best_index) # Remove rectangle from list of rectangles
            elif tallest_neighbour == 2: # If the tallest neighbour is on the right:
                for i in range(len(w_h)):
                    if w_h[i][0] <= len_gap and w_h[i][2] > current_best: # Check if rectangle i fits and is has more penalty points than our current best 
                        current_best = w_h[i][2]
                        current_best_index = i
                        current_best_or = 0
                        continue
                    elif w_h[i][1] <= len_gap and w_h[i][2] > current_best: # Check if rectangle i fits and is has more penalty points than our current best 
                        current_best = w_h[i][2]
                        current_best_index = i
                        current_best_or = 1
                        continue
                    elif w_h[i][2] <= current_best: # If the current rectangle has less penalty points than our current best, place the current best
                        break
                if current_best == -1:   # If no rectangle fits, raise the gap to the level of the shortest neighbour
                    for j in range(len_gap):
                        sheet[index_min + j] = sheet[index_min - 1]
                elif current_best_or == 1:
                    rec_height = sheet[index_min] + w_h[current_best_index][0]
                    p_incr = max(0,rec_height - target_height)
                    p = w_h[current_best_index][2] + p_incr
                    x_y_w_h_p.append([index_min + len_gap - w_h[current_best_index][1],sheet[index_min],w_h[current_best_index][1],w_h[current_best_index][0],p]) # Store coordinates of placed rectangle
                    for j in range(w_h[current_best_index][1]): # Update sheet height
                        sheet[index_min + len_gap - 1 - j] += w_h[current_best_index][0]                
                    w_h.pop(current_best_index) # Remove the placed rectangle from the list of rectangles
                elif current_best_or == 0:
                    rec_height = sheet[index_min] + w_h[current_best_index][1]
                    p_incr = max(0,rec_height - target_height)
                    p = w_h[current_best_index][2] + p_incr
                    x_y_w_h_p.append([index_min + len_gap - w_h[current_best_index][0],sheet[index_min],w_h[current_best_index][0],w_h[current_best_index][1],p]) # Store coordinates of placed rectangle
                    for j in range(w_h[current_best_index][0]): # Update sheet height
                        sheet[index_min + len_gap - 1 - j] += w_h[current_best_index][1]                
                    w_h.pop(current_best_index) # Remove the placed rectangle from the list of rectangles
    opt_finish = 0  # Let opt_finish be zero as long as there are still improvements possible
    while opt_finish == 0:
        sheet_new = sheet.copy()
        x_y_w_h_p_new = copy.deepcopy(x_y_w_h_p)
        max_height = max(sheet)
        heights = []
        for i in range(len(x_y_w_h_p)):
            heights.append(x_y_w_h_p[i][1] + x_y_w_h_p[i][3])       
        index_max = max(range(len(heights)), key=heights.__getitem__)   # Find the index of the highest rectangle
        for i in range(x_y_w_h_p[index_max][2]):  # Remove the rectangle and update the sheet accordingly
            sheet_new[x_y_w_h_p[index_max][0] + i] -= x_y_w_h_p[index_max][3]
        fit = 0
        if x_y_w_h_p[index_max][2] > x_y_w_h_p[index_max][3]: # If the width is larger than the height of the removed rectangle, optimization is finished
            opt_finish = 1
        else:
            width = x_y_w_h_p_new[index_max][3]   # Rotate the rectangle
            height = x_y_w_h_p_new[index_max][2]
            x_y_w_h_p_new[index_max][2] = width
            x_y_w_h_p_new[index_max][3] = height
            if width > W:
                opt_finish = 1
                break
            while fit == 0:
                index_min = min(range(len(sheet_new)), key=sheet_new.__getitem__)   # Find the index of the lowest gap
                min_height = min(sheet_new) # Find the height of the lowest gap
                len_gap = 0
                for i in range(len(sheet_new) - index_min): # Find the width of the lowest gap
                    if sheet_new[index_min + i] == min_height:
                        len_gap += 1
                    else:
                        break
                if index_min == 0:  # If the lowest gap is against the left wall:
                    if width <= len_gap:    # Check if the rectangle fits
                        x_y_w_h_p_new[index_max][0] = index_min + len_gap - width     # Store coordinates of the rectangle
                        x_y_w_h_p_new[index_max][1] = sheet_new[index_min + len_gap - width]
                        for i in range(width):  # Update sheet height
                            sheet_new[index_min + len_gap - width + i] += height
                        fit = 1
                        max_height_new = max(sheet_new) # Find the height of the strip
                        if max_height_new >= max_height:    # Check if the height has improved
                            opt_finish = 1  # If it has not improved, we stop the algorithm
                        else:
                            sheet = sheet_new.copy()    # If it has improved we update the max height and store the solution
                            x_y_w_h_p = copy.deepcopy(x_y_w_h_p_new)
                            max_height = max_height_new
                    else:
                        for i in range(len_gap):    # If the rectangle does not fit in the gap, raise the gap to the level of the right neighbour
                            sheet_new[index_min + i] = sheet_new[index_min + len_gap]   
                elif index_min + len_gap == W:  # If the gap borders the right wall of the strip:
                    if width <= len_gap:    # Check if the rectangle fits
                        x_y_w_h_p_new[index_max][0] = index_min   # Store coordinates of the rectangle
                        x_y_w_h_p_new[index_max][1] = sheet_new[index_min]
                        for i in range(width):  # Update the strip height
                            sheet_new[index_min + i] += height
                        fit = 1
                        max_height_new = max(sheet_new) # Find the new height of the strip
                        if max_height_new >= max_height:    # Check if the height has improved
                            opt_finish = 1  # If the height has not improved, stop the algorithm
                        else:
                            sheet = sheet_new.copy()    # If the height has improved, store the solution and update the best height    
                            x_y_w_h_p = copy.deepcopy(x_y_w_h_p_new)
                            max_height = max_height_new
                    else:
                        for i in range(len_gap):    # If the rectangle does not fit, raise the gap to the level of the left neighbour
                            sheet_new[index_min + i] = sheet_new[index_min - 1]
                else: # If the gap does not border any of the strip walls
                    if sheet_new[index_min - 1] >= sheet_new[index_min + len_gap]:  # Find the tallest neighbour
                        tallest_neighbour = 0
                    else:
                        tallest_neighbour = 1
                    if tallest_neighbour == 1:  # If the tallest neighbour is to the right
                        if width <= len_gap:    # Check if the rectangle fits in the gap
                            x_y_w_h_p_new[index_max][0] = index_min + len_gap - width # Store the coordinates of the rectangle
                            x_y_w_h_p_new[index_max][1] = sheet_new[index_min + len_gap - width]
                            for i in range(width):  # Update the sheet height
                                sheet_new[index_min + len_gap - width + i] += height
                            fit = 1
                            max_height_new = max(sheet_new) # Find the new height of the sheet 
                            if max_height_new >= max_height:    # If the height has not improved, stop the algorithm
                                opt_finish = 1
                            else:
                                sheet = sheet_new.copy()    # If the solution has improved, store the solution and update the max height
                                x_y_w_h_p = copy.deepcopy(x_y_w_h_p_new)
                                max_height = max_height_new
                        else:
                            for i in range(len_gap):    # If the rectangle does not fit, update the height to the level of the left neighbour
                                sheet_new[index_min + i] = sheet_new[index_min - 1]
                    elif tallest_neighbour == 0:    # If the tallest neighbour is on the right of the gap:
                        if width <= len_gap:    # Check if the rectangle fits in the gap
                            x_y_w_h_p_new[index_max][0] = index_min   # Store the coordinates
                            x_y_w_h_p_new[index_max][1] = sheet_new[index_min]
                            for i in range(width):  # Update the sheet heights
                                sheet_new[index_min + i] += height
                            fit = 1
                            max_height_new = max(sheet_new) # Find new max height
                            if max_height_new >= max_height:    # Check if the solution has improved
                                opt_finish = 1  # If it has not improved, stop the algorithm
                            else:   # If it has improved, save the solution and update the max height
                                sheet = sheet_new.copy()
                                x_y_w_h_p = copy.deepcopy(x_y_w_h_p_new)
                                max_height = max_height_new
                        else:   # If the rectangle does not fit in the gap, fill the gap till the level of the right neighbour
                            for i in range(len_gap):    
                                sheet_new[index_min + i] = sheet_new[index_min + len_gap]
    for i in range(n):  # Refill w_h with updated penalty points for the next run
        w_h.append([x_y_w_h_p[i][2],x_y_w_h_p[i][3],x_y_w_h_p[i][4]])
    if max_height <= max_height_tot:    
        max_height_tot = max_height
        x_y_w_h_best = copy.deepcopy(x_y_w_h_p)

# Show height of strip
print('The height of the strip is' + ' ' + str(max_height_tot))

# End timer
elapsed = time.time() - t
print(elapsed)

# Show plot
ax.add_patch(matplotlib.patches.Rectangle((0,0),W,max_height,color = 'black', fill=None, hatch = '////'))
for i in range(len(x_y_w_h_p)):
    ax.add_patch(matplotlib.patches.Rectangle((x_y_w_h_best[i][0],x_y_w_h_best[i][1]),x_y_w_h_best[i][2],x_y_w_h_best[i][3],color = next(colors_cycle)))
    ax.add_patch(matplotlib.patches.Rectangle((x_y_w_h_best[i][0],x_y_w_h_best[i][1]),x_y_w_h_best[i][2],x_y_w_h_best[i][3], fill = None))
plt.xlim([0, W])
plt.ylim([0, max_height_tot])
plt.show()         

from calendar import c
from operator import index
from re import A, I
import matplotlib
import matplotlib.pyplot as plt
import itertools
from numpy import delete
import time
import copy

# Start timer
t=time.time()

# Input data
w_h = []
with open('Data sets\zdf11.TXT') as f:
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

# Sort data
for i in range(n):
    if w_h[i][0] < w_h[i][1]:
        old_w = w_h[i][0]
        old_h = w_h[i][1]
        w_h[i][0] = old_h
        w_h[i][1] = old_w
w_h = sorted(w_h, key = lambda x:(-x[0],-x[1]))

# Set up visual representation
colors_cycle = itertools.cycle(['green', 'blue', 'red', 'purple', 'cyan', 'pink', 'lime', 'yellow', 'brown', 'black', 'magenta', 'olive', 'gray', 'orange', 'gold', 'lightgreen', 'deeppink', 'crimson', 'maroon', 'indigo', 'darkgreen', 'darkorange', 'navy'])
fig = plt.figure()
# plt.xlim([0, W])
# plt.ylim([0, 22])
ax = fig.add_subplot(111)

# Place the rectangles according to Best Fit (Tallest Neighbour variant)

sheet = [0] * W
x_y_w_h = []
while len(w_h) != 0:
    index_min = min(range(len(sheet)), key=sheet.__getitem__)
    min_height = min(sheet)
    len_gap = 0
    current_best = 0
    for i in range(len(sheet) - index_min):
        if sheet[index_min + i] == min_height:
            len_gap += 1
        else:
            break
    if index_min == 0:
        for i in range(len(w_h)):
            if w_h[i][0] <= len_gap and w_h[i][0] > current_best:
                current_best = w_h[i][0]
                current_best_index = i
                current_best_or = 0
                continue
            elif w_h[i][1] <= len_gap and w_h[i][1] > current_best:
                current_best = w_h[i][1]
                current_best_index = i
                current_best_or = 1
                continue
            elif w_h[i][0] <= current_best and w_h[i][1] <= current_best:
                break
        if current_best == 0:
            for j in range(len_gap):
                sheet[index_min + j] = sheet[index_min + len_gap]
        elif current_best_or == 1:
            x_y_w_h.append([len_gap - w_h[current_best_index][1],sheet[index_min],w_h[current_best_index][1],w_h[current_best_index][0]])
            for j in range(w_h[current_best_index][1]):
                sheet[index_min + len_gap - 1 - j] += w_h[current_best_index][0]                
            w_h.pop(current_best_index)
        elif current_best_or == 0:
            x_y_w_h.append([len_gap - w_h[current_best_index][0],sheet[index_min],w_h[current_best_index][0],w_h[current_best_index][1]])
            for j in range(w_h[current_best_index][0]):
                sheet[index_min + len_gap - 1 - j] += w_h[current_best_index][1]                
            w_h.pop(current_best_index)
    elif index_min + len_gap == W:
        for i in range(len(w_h)):
            if w_h[i][0] <= len_gap and w_h[i][0] > current_best:
                current_best = w_h[i][0]
                current_best_index = i
                current_best_or = 0
                continue
            elif w_h[i][1] <= len_gap and w_h[i][1] > current_best:
                current_best = w_h[i][1]
                current_best_index = i
                current_best_or = 1
                continue
            elif w_h[i][0] <= current_best and w_h[i][1] <= current_best:
                break
        if current_best == 0:
            for j in range(len_gap):
                sheet[index_min + j] = sheet[index_min - 1]
        elif current_best_or == 1:
            x_y_w_h.append([index_min,sheet[index_min],w_h[current_best_index][1],w_h[current_best_index][0]])
            for j in range(w_h[current_best_index][1]):
                sheet[index_min + j] += w_h[current_best_index][0]                
            w_h.pop(current_best_index)
        elif current_best_or == 0:
            x_y_w_h.append([index_min,sheet[index_min],w_h[current_best_index][0],w_h[current_best_index][1]])
            for j in range(w_h[current_best_index][0]):
                sheet[index_min + j] += w_h[current_best_index][1]                
            w_h.pop(current_best_index)
    else:
        neighbour_1 = sheet[index_min - 1]
        neighbour_2 = sheet[index_min + len_gap]
        if neighbour_1 >= neighbour_2:
            tallest_neighbour = 1
            tallest_neighbour_index = index_min - 1
            shortest_neighbour_index = index_min + len_gap
        else:
            tallest_neighbour = 2
            tallest_neighbour_index = index_min + len_gap
            shortest_neighbour_index = index_min - 1
        if tallest_neighbour == 1:
            for i in range(len(w_h)):
                if w_h[i][0] <= len_gap and w_h[i][0] > current_best:
                    current_best = w_h[i][0]
                    current_best_index = i
                    current_best_or = 0
                    continue
                elif w_h[i][1] <= len_gap and w_h[i][1] > current_best:
                    current_best = w_h[i][1]
                    current_best_index = i
                    current_best_or = 1
                    continue
                elif w_h[i][0] <= current_best and w_h[i][1] <= current_best:
                    break
            if current_best == 0:
                for j in range(len_gap):
                    sheet[index_min + j] = sheet[index_min + len_gap]
            elif current_best_or == 1:
                x_y_w_h.append([index_min,sheet[index_min],w_h[current_best_index][1],w_h[current_best_index][0]])
                for j in range(w_h[current_best_index][1]):
                    sheet[index_min + j] += w_h[current_best_index][0]                
                w_h.pop(current_best_index)
            elif current_best_or == 0:
                x_y_w_h.append([index_min,sheet[index_min],w_h[current_best_index][0],w_h[current_best_index][1]])
                for j in range(w_h[current_best_index][0]):
                    sheet[index_min + j] += w_h[current_best_index][1]                
                w_h.pop(current_best_index)
        elif tallest_neighbour == 2:
            for i in range(len(w_h)):
                if w_h[i][0] <= len_gap and w_h[i][0] > current_best:
                    current_best = w_h[i][0]
                    current_best_index = i
                    current_best_or = 0
                    continue
                elif w_h[i][1] <= len_gap and w_h[i][1] > current_best:
                    current_best = w_h[i][1]
                    current_best_index = i
                    current_best_or = 1
                    continue
                elif w_h[i][0] <= current_best and w_h[i][1] <= current_best:
                    break
            if current_best == 0:
                for j in range(len_gap):
                    sheet[index_min + j] = sheet[index_min - 1]
            elif current_best_or == 1:
                x_y_w_h.append([index_min + len_gap - w_h[current_best_index][1],sheet[index_min],w_h[current_best_index][1],w_h[current_best_index][0]])
                for j in range(w_h[current_best_index][1]):
                    sheet[index_min + len_gap - 1 - j] += w_h[current_best_index][0]                
                w_h.pop(current_best_index)
            elif current_best_or == 0:
                x_y_w_h.append([index_min + len_gap - w_h[current_best_index][0],sheet[index_min],w_h[current_best_index][0],w_h[current_best_index][1]])
                for j in range(w_h[current_best_index][0]):
                    sheet[index_min + len_gap - 1 - j] += w_h[current_best_index][1]                
                w_h.pop(current_best_index)
opt_finish = 0
while opt_finish == 0:
    sheet_new = sheet.copy()
    x_y_w_h_new = copy.deepcopy(x_y_w_h)
    max_height = max(sheet)
    heights = []
    for i in range(len(x_y_w_h)):
        heights.append(x_y_w_h[i][1] + x_y_w_h[i][3])       
    index_max = max(range(len(heights)), key=heights.__getitem__)
    for i in range(x_y_w_h[index_max][2]):
        sheet_new[x_y_w_h[index_max][0] + i] -= x_y_w_h[index_max][3]
    fit = 0
    if x_y_w_h[index_max][2] > x_y_w_h[index_max][3]:
        opt_finish = 1
    else:
        width = x_y_w_h_new[index_max][3]
        height = x_y_w_h_new[index_max][2]
        x_y_w_h_new[index_max][2] = width
        x_y_w_h_new[index_max][3] = height
        if width > W:
            opt_finish = 1
            break
        while fit == 0:
            index_min = min(range(len(sheet_new)), key=sheet_new.__getitem__)
            min_height = min(sheet_new)
            len_gap = 0
            for i in range(len(sheet_new) - index_min):
                if sheet_new[index_min + i] == min_height:
                    len_gap += 1
                else:
                    break
            if index_min == 0:
                if width <= len_gap:
                    x_y_w_h_new[index_max][0] = index_min + len_gap - width
                    x_y_w_h_new[index_max][1] = sheet_new[index_min + len_gap - width]
                    for i in range(width):
                        sheet_new[index_min + len_gap - width + i] += height
                    fit = 1
                    max_height_new = max(sheet_new)
                    if max_height_new >= max_height:
                        opt_finish = 1
                    else:
                        sheet = sheet_new.copy()
                        x_y_w_h = copy.deepcopy(x_y_w_h_new)
                        max_height = max_height_new
                else:
                    for i in range(len_gap):
                        sheet_new[index_min + i] = sheet_new[index_min + len_gap]
            elif index_min + len_gap == W:
                if width <= len_gap:
                    x_y_w_h_new[index_max][0] = index_min
                    x_y_w_h_new[index_max][1] = sheet_new[index_min]
                    for i in range(width):
                        sheet_new[index_min + i] += height
                    fit = 1
                    max_height_new = max(sheet_new)
                    if max_height_new >= max_height:
                        opt_finish = 1
                    else:
                        sheet = sheet_new.copy()
                        x_y_w_h = copy.deepcopy(x_y_w_h_new)
                        max_height = max_height_new
                else:
                    for i in range(len_gap):
                        sheet_new[index_min + i] = sheet_new[index_min - 1]
            else: 
                if sheet_new[index_min - 1] >= sheet_new[index_min + len_gap]:
                    tallest_neighbour = 0
                else:
                    tallest_neighbour = 1
                if tallest_neighbour == 1:
                    if width <= len_gap:
                        x_y_w_h_new[index_max][0] = index_min + len_gap - width
                        x_y_w_h_new[index_max][1] = sheet_new[index_min + len_gap - width]
                        for i in range(width):
                            sheet_new[index_min + len_gap - width + i] += height
                        fit = 1
                        max_height_new = max(sheet_new)
                        if max_height_new >= max_height:
                            opt_finish = 1
                        else:
                            sheet = sheet_new.copy()
                            x_y_w_h = copy.deepcopy(x_y_w_h_new)
                            max_height = max_height_new
                    else:
                        for i in range(len_gap):
                            sheet_new[index_min + i] = sheet_new[index_min - 1]
                elif tallest_neighbour == 0:
                    if width <= len_gap:
                        x_y_w_h_new[index_max][0] = index_min
                        x_y_w_h_new[index_max][1] = sheet_new[index_min]
                        for i in range(width):
                            sheet_new[index_min + i] += height
                        fit = 1
                        max_height_new = max(sheet_new)
                        if max_height_new >= max_height:
                            opt_finish = 1
                        else:
                            sheet = sheet_new.copy()
                            x_y_w_h = copy.deepcopy(x_y_w_h_new)
                            max_height = max_height_new
                    else:
                        for i in range(len_gap):
                            sheet_new[index_min + i] = sheet_new[index_min + len_gap]

        



# Show height of strip
max_height = max(sheet)
print('The height of the strip is' + ' ' + str(max_height))

# End timer
elapsed = time.time() - t
print(elapsed)

# Show plot
ax.add_patch(matplotlib.patches.Rectangle((0,0),W,max_height,color = 'black', fill=None, hatch = '////'))
for i in range(len(x_y_w_h)):
    ax.add_patch(matplotlib.patches.Rectangle((x_y_w_h[i][0],x_y_w_h[i][1]),x_y_w_h[i][2],x_y_w_h[i][3],color = next(colors_cycle)))
    ax.add_patch(matplotlib.patches.Rectangle((x_y_w_h[i][0],x_y_w_h[i][1]),x_y_w_h[i][2],x_y_w_h[i][3], fill = None))
plt.xlim([0, W])
plt.ylim([0, max_height])
plt.show()         
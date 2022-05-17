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
# W=200
# w_h = [[18,90],[15,63],[23,89],[108,19],[17,45],[19,106],[12,30],[22,14],[74,26],[5,16],[17,41],[9,16],[82,12],[17,25],[3,4],[79,17],[12,13],[7,27],[8,32],[33,5],[4,17],[20,5],[48,13],[10,20],[10,32],[25,22],[8,12],[9,12],[3,20],[8,36],[14,9],[26,7],[8,7],[9,6],[14,73],[25,5],[12,10],[5,10],[4,8],[9,2],[17,11],[18,11],[35,11],[7,29],[23,9],[7,2],[49,14],[30,10],[28,40],[21,48],[11,44],[3,18],[13,13],[16,16],[87,30],[5,26],[8,3],[24,23],[10,38],[22,6],[23,9],[67,16],[32,24],[1,4],[10,12],[6,11],[16,3],[22,8],[39,8],[8,20],[4,26],[55,8],[63,18],[18,5],[20,4],[4,1],[6,13],[10,12],[22,22],[49,15],[35,11],[3,13],[1,1],[9,11],[7,10],[35,20],[28,9],[31,11],[4,2],[7,19],[1,6],[7,9],[47,16],[2,7],[5,13],[66,11],[40,9]]
# w_h = [[52,34],[31,14],[29,7],[16,79],[7,48],[7,24],[21,12],[22,37],[15,13],[11,7],[4,15],[14,10],[3,6],[3,10],[15,26],[5,24],[10,35],[42,22],[7,5],[7,19],[1,6],[2,4],[11,14],[5,2],[13,14],[50,11],[2,2],[62,9],[27,11],[13,10],[15,36],[33,11],[18,18],[42,32],[19,34],[20,26],[14,26],[23,98],[25,11],[8,7],[26,4],[21,26],[30,10],[24,52],[11,36],[14,28],[72,16],[35,14],[13,48],[76,13],[7,17],[7,18],[25,8],[12,22],[13,45],[2,8],[12,29],[37,20],[13,29],[26,94],[6,13],[1,1],[13,21],[8,12],[26,23],[28,15],[9,9],[21,20],[3,5],[28,15],[22,6],[6,14],[9,5],[23,9],[7,14],[18,17],[6,9],[3,16],[42,26],[9,26],[60,14],[5,12],[18,41],[12,7],[5,34],[2,3],[20,31],[37,12],[23,19],[20,29],[14,28],[28,7],[46,7],[97,21],[10,2],[15,11],[48,9]]
# w_h = [[28,7],[9,8],[8,6],[59,22],[23,10],[7,5],[27,8],[10,45],[29,32],[3,14],[4,3],[4,2],[4,14],[8,21],[20,13],[13,12],[31,11],[18,41],[5,9],[26,8],[13,18],[37,8],[20,6],[30,10],[9,31],[19,24],[7,10],[65,29],[24,25],[6,21],[4,13],[25,57],[20,14],[11,18],[3,19],[12,59],[23,70],[22,38],[15,4],[23,70],[19,57],[13,3],[29,8],[20,18],[6,12],[7,5],[7,26],[4,1],[7,25],[36,7],[23,59],[18,50],[1,3],[14,40],[7,46],[20,40],[37,7],[4,23],[32,27],[5,16],[5,30],[7,11],[9,11],[5,13],[14,86],[8,25],[6,6],[9,7],[4,14],[16,9],[14,5],[2,9],[16,18],[15,6],[26,17],[8,4],[6,14],[28,73],[7,12],[2,7],[35,11],[24,10],[25,9],[6,5],[62,22],[49,8],[47,19],[7,14],[24,20],[18,25],[6,21],[41,41],[69,18],[2,12],[22,5],[40,7],[117,20]]
# w_h = [[13,9],[22,7],[22,64],[11,12],[35,11],[36,14],[21,55],[18,116],[22,56],[10,2],[12,27],[23,25],[5,1],[30,42],[16,41],[16,68],[7,20],[13,6],[4,14],[9,35],[27,94],[8,46],[11,21],[7,11],[31,20],[8,2],[4,18],[4,25],[41,27],[10,60],[12,75],[29,16],[35,7],[12,7],[7,11],[4,24],[47,12],[6,20],[34,102],[23,49],[8,11],[4,4],[11,7],[14,10],[6,5],[27,8],[4,5],[2,10],[5,21],[14,6],[33,7],[18,5],[18,15],[5,9],[28,5],[20,21],[13,4],[15,7],[18,3],[28,15],[56,9],[14,4],[4,25],[56,19],[16,14],[13,56],[13,34],[50,24],[7,33],[7,21],[7,2],[2,4],[7,22],[5,3],[2,2],[4,1],[65,17],[11,12],[13,10],[37,29],[6,1],[55,15],[8,5],[3,15],[24,35],[14,11],[12,8],[6,22],[2,10],[8,3],[4,11],[22,8],[19,21],[36,12],[5,12],[110,16],[47,9]]
# w_h = [[18,90],[15,63],[23,89],[108,19],[17,45],[19,106],[12,30],[22,14],[74,26],[5,16],[17,41]]
# n = len(w_h)
w_h = []
with open('Data sets/Test04.TXT') as f:
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
            # ax.add_patch(matplotlib.patches.Rectangle((len_gap - w_h[current_best_index][1],sheet[index_min]),w_h[current_best_index][1],w_h[current_best_index][0],color = next(colors_cycle)))
            # ax.add_patch(matplotlib.patches.Rectangle((len_gap - w_h[current_best_index][1],sheet[index_min]),w_h[current_best_index][1],w_h[current_best_index][0], fill = None))                
            for j in range(w_h[current_best_index][1]):
                sheet[index_min + len_gap - 1 - j] += w_h[current_best_index][0]                
            w_h.pop(current_best_index)
        elif current_best_or == 0:
            x_y_w_h.append([len_gap - w_h[current_best_index][0],sheet[index_min],w_h[current_best_index][0],w_h[current_best_index][1]])
            # ax.add_patch(matplotlib.patches.Rectangle((len_gap - w_h[current_best_index][0],sheet[index_min]),w_h[current_best_index][0],w_h[current_best_index][1],color = next(colors_cycle)))
            # ax.add_patch(matplotlib.patches.Rectangle((len_gap - w_h[current_best_index][0],sheet[index_min]),w_h[current_best_index][0],w_h[current_best_index][1], fill = None))                
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
            # ax.add_patch(matplotlib.patches.Rectangle((index_min,sheet[index_min]),w_h[current_best_index][1],w_h[current_best_index][0],color = next(colors_cycle)))
            # ax.add_patch(matplotlib.patches.Rectangle((index_min,sheet[index_min]),w_h[current_best_index][1],w_h[current_best_index][0], fill = None))                
            for j in range(w_h[current_best_index][1]):
                sheet[index_min + j] += w_h[current_best_index][0]                
            w_h.pop(current_best_index)
        elif current_best_or == 0:
            x_y_w_h.append([index_min,sheet[index_min],w_h[current_best_index][0],w_h[current_best_index][1]])
            # ax.add_patch(matplotlib.patches.Rectangle((index_min,sheet[index_min]),w_h[current_best_index][0],w_h[current_best_index][1],color = next(colors_cycle)))
            # ax.add_patch(matplotlib.patches.Rectangle((index_min,sheet[index_min]),w_h[current_best_index][0],w_h[current_best_index][1], fill = None))                
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
                # ax.add_patch(matplotlib.patches.Rectangle((index_min,sheet[index_min]),w_h[current_best_index][1],w_h[current_best_index][0],color = next(colors_cycle)))
                # ax.add_patch(matplotlib.patches.Rectangle((index_min,sheet[index_min]),w_h[current_best_index][1],w_h[current_best_index][0], fill = None))                
                for j in range(w_h[current_best_index][1]):
                    sheet[index_min + j] += w_h[current_best_index][0]                
                w_h.pop(current_best_index)
            elif current_best_or == 0:
                x_y_w_h.append([index_min,sheet[index_min],w_h[current_best_index][0],w_h[current_best_index][1]])
                # ax.add_patch(matplotlib.patches.Rectangle((index_min,sheet[index_min]),w_h[current_best_index][0],w_h[current_best_index][1],color = next(colors_cycle)))
                # ax.add_patch(matplotlib.patches.Rectangle((index_min,sheet[index_min]),w_h[current_best_index][0],w_h[current_best_index][1], fill = None))                
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
                # ax.add_patch(matplotlib.patches.Rectangle((index_min + len_gap - w_h[current_best_index][1],sheet[index_min],sheet[index_min]),w_h[current_best_index][1],w_h[current_best_index][0],color = next(colors_cycle)))
                # ax.add_patch(matplotlib.patches.Rectangle((index_min + len_gap - w_h[current_best_index][1],sheet[index_min],sheet[index_min]),w_h[current_best_index][1],w_h[current_best_index][0], fill = None))                
                for j in range(w_h[current_best_index][1]):
                    sheet[index_min + len_gap - 1 - j] += w_h[current_best_index][0]                
                w_h.pop(current_best_index)
            elif current_best_or == 0:
                x_y_w_h.append([index_min + len_gap - w_h[current_best_index][0],sheet[index_min],w_h[current_best_index][0],w_h[current_best_index][1]])
                # ax.add_patch(matplotlib.patches.Rectangle((index_min + len_gap - w_h[current_best_index][0],sheet[index_min],w_h[current_best_index][0]),w_h[current_best_index][0],w_h[current_best_index][1],color = next(colors_cycle)))
                # ax.add_patch(matplotlib.patches.Rectangle((index_min + len_gap - w_h[current_best_index][0],sheet[index_min],w_h[current_best_index][0]),w_h[current_best_index][0],w_h[current_best_index][1], fill = None))
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
# ax.add_patch(matplotlib.patches.Rectangle((0,0),W,max_height,color = 'black', fill=None, hatch = '////'))
for i in range(len(x_y_w_h)):
    ax.add_patch(matplotlib.patches.Rectangle((x_y_w_h[i][0],x_y_w_h[i][1]),x_y_w_h[i][2],x_y_w_h[i][3],color = next(colors_cycle)))
    ax.add_patch(matplotlib.patches.Rectangle((x_y_w_h[i][0],x_y_w_h[i][1]),x_y_w_h[i][2],x_y_w_h[i][3], fill = None))
plt.xlim([0, W])
plt.ylim([0,22])
# plt.ylim([0, max_height])
plt.show()         
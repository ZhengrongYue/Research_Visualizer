# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# print(plt.style.available)

plt.style.use('seaborn-v0_8-muted')

plt.rcParams['font.serif']=['Times New Roman']


font = FontProperties(size=6)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# 使用matplotlib作图
fig = plt.figure()
fig.set_facecolor('#FFFFFF')
fig.subplots_adjust(wspace=0.5, hspace=0.20, top=0.85, bottom=0.05)

#图1

algs=['Qwen-VL','Gemini','GPT-4o','Qwen-VL (3-Shot)','Gemini (3-Shot)','GPT-4o (3-Shot)']

stats_ls = [
    [70.00  , 62.50  , 56.73  , 54.80  , 46.71  , 67.57  , 46.83  , 43.33  , 41.08  , 45.7  , 48.96  , 46.85],
    [65.00  , 65.91  , 58.65  , 51.41  , 46.05  , 56.76  , 39.02  , 39.33  , 34.85  , 35.40  , 35.52  , 29.72],
    [46.00  , 43.18  , 38.46  , 33.90  , 32.89  , 32.43  , 27.32  , 24.67  , 28.63  , 19.59  , 27.76  , 23.78],
    [60.00  , 76.14  , 72.12  , 69.49  , 57.24  , 71.17  , 64.39  , 58.67  , 52.70  , 66.67  , 68.06  , 67.83],
    [69.00  , 64.77  , 52.88  , 41.24  , 46.05  , 55.86  , 38.54  , 41.33  , 39.83  , 33.33  , 33.73  , 34.97],
    [80.00  , 84.09  , 75.00  , 81.92  , 69.74 , 72.07  , 68.78  , 61.33  , 56.85  , 52.23  , 66.57  , 59.44]]

labels=['Level 1','Level 2','Level 3','Level 4','Level 5','Level 6','Level 7','Level 8','Level 9','Level 10','Level 11','Level 12']

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

ax = fig.add_subplot(121, polar=True)

lines = []
for i,stats in enumerate(stats_ls):

    stats = np.concatenate((stats, [stats[0]]))
    
    line, = ax.plot(np.concatenate((angles, [angles[0]])), stats, linewidth=0.75,color=colors[i],label=algs[i])#'o-', 
    lines.append(line)
    ax.fill(np.concatenate((angles, [angles[0]])), stats, alpha=0.1, color=colors[i])
    ax.set_rgrids([20,40,60,80],font=FontProperties(size=6),color='grey')
    # ax.set_ylim(min-5, max+5)

# ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties=font,fontname='Times New Roman')
ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties=font)

ax.spines['polar'].set_visible(False)

# ax.set_title("(a) Closed-source LMMs",fontsize=8,loc="center",y=-0.25,fontname='Times New Roman')
ax.set_title("(a) Closed-source LMMs",fontsize=8,loc="center",y=-0.25)

ax.legend(handles=lines, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.4), fancybox=True, shadow=False, frameon=True, prop=FontProperties(size=6, family='Times New Roman'), framealpha=0.1)



#图2

algs=['CogVLM2','InternLM-VL','Qwen2-VL-Instruct','LLaVA-v1.6-mistral','Math-LMM (Ours 7B)','Math-LMM (Ours 72B)']

stats_ls = [
    [38.0  , 36.36  , 21.15  , 29.38  , 20.39  , 24.32  , 27.32  , 29.33  , 24.9  , 23.02  , 25.37  , 22.73],
    [18.0  , 15.91  , 20.19  , 12.99  , 6.58  , 28.83  , 23.90  , 26.0  , 26.97  , 13.06  , 22.69  , 20.63],
    [54.00 , 57.95 , 49.04  , 49.72, 33.55  , 48.65  , 47.80 , 46.00  ,41.08  , 37.46 ,38.81 , 38.46],
    [33.00 , 19.32 , 25.00 , 18.64 , 16.45 , 13.51 , 16.59 , 14.67 , 10.37, 15.12, 18.51, 14.34],
    [32.00 , 44.32 , 37.50 , 35.59 , 32.24 , 35.14 , 32.20 , 28.67 , 32.37 , 30.58 , 31.94 , 26.22],
    [47.00 , 62.50 , 53.85 , 59.32 , 48.03 , 58.56 , 51.71 , 43.33 , 38.59 , 46.05 , 44.78 , 48.60]
    ]


labels=['Level 1','Level 2','Level 3','Level 4','Level 5','Level 6','Level 7','Level 8','Level 9','Level 10','Level 11','Level 12']

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

ax = fig.add_subplot(122, polar=True)

lines = []
for i,stats in enumerate(stats_ls):

    stats = np.concatenate((stats, [stats[0]]))    
    line, = ax.plot(np.concatenate((angles, [angles[0]])), stats, linewidth=0.75,color=colors[i],label=algs[i])
    lines.append(line)
    ax.fill(np.concatenate((angles, [angles[0]])), stats, alpha=0.1, color=colors[i])
    ax.set_rgrids([10,25,40,55],font=FontProperties(size=6),color='grey')

# ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties=font,fontname='Times New Roman')
ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties=font)

ax.spines['polar'].set_visible(False)

# ax.set_title("(b) Open-source LMMs",fontsize=8,loc="center",y=-0.25,fontname='Times New Roman')
ax.set_title("(b) Open-source LMMs",fontsize=8,loc="center",y=-0.25)

ax.legend(handles=lines, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.4), fancybox=True, shadow=False, frameon=True, prop=FontProperties(size=6, family='Times New Roman'), framealpha=0.1)


plt.savefig('radar_all_level.pdf',format='pdf',dpi=500, bbox_inches='tight', pad_inches = +0.1)

plt.show()

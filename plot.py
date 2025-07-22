import matplotlib.pyplot as plt
import numpy as np


title_size = 25  
label_size = 20  
tick_size = 16
legend_size = 13

# # 数据
# methods = ['GA', 'GD', 'KL', 'PO', 'Ours']
# origin_concrete = [60.0, 69.2, 71.7, 85.9, 97.5]  # Concrete Concepts - Origin
# translation_concrete = [58.4, 65.4, 68.2, 63.1, 94.3]  # Concrete Concepts - Translation
# quantization_concrete = [56.5, 60.8, 65.3, 60.7, 90.4]  # Concrete Concepts - Quantization

# origin_abstract = [50.2, 66.0, 67.1, 74.6, 92.0]  # Abstract Concepts - Origin
# translation_abstract = [40.0, 64.2, 65.0, 68.3, 90.1]  # Abstract Concepts - Translation
# quantization_abstract = [50.3, 62.1, 65.7, 64.9, 89.4]  # Abstract Concepts - Quantization

# # 设置柱状图的位置和宽度
# x = np.arange(len(methods))  # 横坐标位置
# width = 0.2  # 柱状图宽度

# # 创建图形和子图
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# # 绘制第一张图：Concrete Concepts
# rects1 = ax1.bar(x - width, origin_concrete, width, label='Origin', color='skyblue')
# rects2 = ax1.bar(x, translation_concrete, width, label='Translation', color='lightgreen')
# rects3 = ax1.bar(x + width, quantization_concrete, width, label='Quantization', color='salmon')

# # 设置第一张图的属性
# ax1.set_xlabel('Methods', fontsize=label_size)
# ax1.set_ylabel('Unlearning Accuracy', fontsize=label_size)
# ax1.set_title('Concrete Concepts', fontsize=title_size)
# ax1.set_xticks(x)
# ax1.set_xticklabels(methods, fontsize=tick_size)
# ax1.legend(fontsize=legend_size)
# ax1.grid(True, linestyle='--', alpha=0.6)
# ax1.set_facecolor('#f7f7f7')  # 淡淡的灰色背景

# # 绘制第二张图：Abstract Concepts
# rects4 = ax2.bar(x - width, origin_abstract, width, label='Origin', color='skyblue')
# rects5 = ax2.bar(x, translation_abstract, width, label='Translation', color='lightgreen')
# rects6 = ax2.bar(x + width, quantization_abstract, width, label='Quantization', color='salmon')

# # 设置第二张图的属性
# ax2.set_xlabel('Methods', fontsize=label_size)
# ax2.set_ylabel('Unlearning Accuracy', fontsize=label_size)
# ax2.set_title('Abstract Concepts', fontsize=title_size)
# ax2.set_xticks(x)
# ax2.set_xticklabels(methods, fontsize=tick_size)
# ax2.legend(fontsize=legend_size)
# ax2.grid(True, linestyle='--', alpha=0.6)
# ax2.set_facecolor('#f7f7f7')  # 淡淡的灰色背景

# # 调整布局
# plt.tight_layout()

# # 显示图形
# # plt.show()
# plt.savefig("robustness.png", dpi=400, bbox_inches="tight")


# #### robustness 只画其中一个图形
# # 数据 SAUCE
# title_size = 25  
# label_size = 22  
# tick_size = 18
# legend_size = 18

# methods = ['GA', 'GD', 'KL', 'PO', 'SAUCE']
# origin_concrete = [60.0, 69.2, 79.7, 77.9, 91.5]  # Concrete Concepts - Origin
# translation_concrete = [58.4, 65.4, 68.2, 64.1, 86.0]  # Concrete Concepts - Translation
# quantization_concrete = [56.5, 61.8, 69.3, 60.7, 84.4]  # Concrete Concepts - Quantization


# x = np.arange(len(methods))  # 横坐标位置
# width = 0.2  # 柱状图宽度

# # 创建图形和子图
# fig, ax1 = plt.subplots(figsize=(12, 8))

# # 绘制第一张图：Concrete Concepts
# rects1 = ax1.bar(x - width, origin_concrete, width, label='Before Attacks', color='#718DBF') # 'color='skyblue'
# rects2 = ax1.bar(x, translation_concrete, width, label='Translation Attack', color='#E58E68')  # 'lightgreen'
# rects3 = ax1.bar(x + width, quantization_concrete, width, label='Quantization Attack', color='#5CB090') # salmon'

# # 设置第一张图的属性
# ax1.set_xlabel('Methods', fontsize=label_size)
# ax1.set_ylabel(r'$\text{UA}_{\text{d}}$ (%)', fontsize=label_size)
# # ax1.set_title('Concrete Concept Unlearning Task', fontsize=title_size)
# ax1.set_xticks(x)
# ax1.set_xticklabels(methods, fontsize=tick_size)
# ax1.legend(fontsize=legend_size)
# ax1.grid(True, linestyle='--', alpha=0.6)
# ax1.set_facecolor('#f7f7f7')  # 淡淡的灰色背景
# ax1.tick_params(axis='y', labelsize=tick_size) 

# # 调整布局
# plt.tight_layout()

# # 显示图形
# # plt.show()
# plt.savefig("methods_robustness.png", dpi=400, bbox_inches="tight")



# # # ##### composite accuracy
# # # import matplotlib.pyplot as plt
# # # import numpy as np

# title_size = 25  
# label_size = 22  
# tick_size = 18
# legend_size = 18

# # 数据
# num_concepts = [1, 2, 3, 4, 5, 6]  # 横坐标：Number of Concepts

# # 具体概念的数据
# concrete_unlearning = [88.9, 89.2, 90.4, 89.0, 90.1, 91.3]  # 具体概念的 Unlearning Accuracy
# concrete_utility = [91.5, 90.3, 91.0, 88.4, 89.2, 87.5]  # 具体概念的 Utility Accuracy

# # 抽象概念的数据
# abstract_unlearning = [85.3, 84.5, 87.2, 86.5, 88.1, 88.0]  # 抽象概念的 Unlearning Accuracy
# abstract_utility = [83.8, 82.0, 81.2, 79.3, 80.2, 79.4]  # 抽象概念的 Utility Accuracy

# # 创建图形
# plt.figure(figsize=(12, 8))

# # 绘制具体概念的折线
# plt.plot(num_concepts, concrete_unlearning, label='Concrete Task Unlearning Score', linestyle='-', marker='o', color='blue',linewidth=4, markersize=12)
# plt.plot(num_concepts, concrete_utility, label='Concrete Task Utility Score', linestyle='-', marker='s', color='green',linewidth=4, markersize=12)

# # 绘制抽象概念的折线
# plt.plot(num_concepts, abstract_unlearning, label='Abstract Task Unlearning Score', linestyle='--', marker='o', color='red',linewidth=4, markersize=12)
# plt.plot(num_concepts, abstract_utility, label='Abstract Task Utility Score', linestyle='--', marker='s', color='orange',linewidth=4, markersize=12)

# # 设置图形属性
# plt.xlabel('Number of Unlearning Concepts', fontsize=label_size)
# plt.ylabel('Scores (%)', fontsize=label_size)  # Unlearning or Utility 
# # plt.title('Accuracy vs. Number of Concepts', fontsize=14)
# plt.xticks(num_concepts, fontsize=tick_size)
# plt.yticks(np.arange(0, 101, 10), fontsize=tick_size)  # 纵坐标从 0 到 100，步长为 10
# plt.ylim(40, 100) 
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(fontsize=legend_size)

# # 设置背景颜色
# plt.gca().set_facecolor('#f7f7f7')  # 淡淡的灰色背景

# # 显示图形
# plt.tight_layout()
# # plt.show()
# plt.savefig("composite.png", dpi=400, bbox_inches="tight")


#### robustness 只画其中一个图形 classification
# 数据 SAUCE
title_size = 30 # 25  
label_size = 30  # 22  
tick_size = 25 # 18
legend_size = 25 # 18

methods = ['Forget Set', 'Test Set', 'Retain Set', 'Real Celebrity']
origin_concrete = [33.24, 32.79, 42.22, 48.48]  # Concrete Concepts - Origin
translation_concrete = [35.45, 34.93, 41.23, 47.25]  # Concrete Concepts - Translation
quantization_concrete = [36.27, 35.28, 40.55, 46.11]  # Concrete Concepts - Quantization


x = np.arange(len(methods))  # 横坐标位置
width = 0.2  # 柱状图宽度

# 创建图形和子图
fig, ax1 = plt.subplots(figsize=(12, 8))

# 绘制第一张图：Concrete Concepts
rects1 = ax1.bar(x - width, origin_concrete, width, label='Before Attacks', color='#718DBF') # 'color='skyblue'
rects2 = ax1.bar(x, translation_concrete, width, label='Translation Attack', color='#E58E68')  # 'lightgreen'
rects3 = ax1.bar(x + width, quantization_concrete, width, label='Quantization Attack', color='#5CB090') # salmon'

# 设置第一张图的属性
ax1.set_xlabel('Dataset', fontsize=label_size)
ax1.set_ylabel(r'ACC (%)', fontsize=label_size)
ax1.set_title('Classification Task', fontsize=title_size)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=tick_size)
# ax1.legend(fontsize=legend_size)
ax1.legend(fontsize=legend_size, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15))  # 调整图例位置
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_facecolor('#f7f7f7')  # 淡淡的灰色背景
ax1.tick_params(axis='y', labelsize=tick_size) 

# 调整布局
plt.tight_layout()

# 显示图形
# plt.show()
plt.savefig("methods_robustness_classification_update.png", dpi=400, bbox_inches="tight")



# #### robustness 只画其中一个图形 generation
# # 数据 SAUCE
# title_size = 25  
# label_size = 22  
# tick_size = 18
# legend_size = 18

# methods = ['Forget Set', 'Test Set', 'Retain Set', 'Real Celebrity']
# origin_concrete = [0.424, 0.182, 0.522, 0.422]  # Concrete Concepts - Origin
# translation_concrete = [0.456, 0.202, 0.503, 0.408]  # Concrete Concepts - Translation
# quantization_concrete = [0.471, 0.194, 0.495, 0.391]  # Concrete Concepts - Quantization


# x = np.arange(len(methods))  # 横坐标位置
# width = 0.2  # 柱状图宽度

# # 创建图形和子图
# fig, ax1 = plt.subplots(figsize=(12, 8))

# # 绘制第一张图：Concrete Concepts
# rects1 = ax1.bar(x - width, origin_concrete, width, label='Before Attacks', color='#718DBF') # 'color='skyblue'
# rects2 = ax1.bar(x, translation_concrete, width, label='Translation Attack', color='#E58E68')  # 'lightgreen'
# rects3 = ax1.bar(x + width, quantization_concrete, width, label='Quantization Attack', color='#5CB090') # salmon'

# # 设置第一张图的属性
# ax1.set_xlabel('Dataset', fontsize=label_size)
# # ax1.set_ylabel(r'ACC (%)', fontsize=label_size)
# ax1.set_ylabel(r'Rouge Score', fontsize=label_size)
# ax1.set_title('Generation Task', fontsize=title_size)
# ax1.set_xticks(x)
# ax1.set_xticklabels(methods, fontsize=tick_size)
# ax1.legend(fontsize=legend_size, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15))  # 调整图例位置
# ax1.grid(True, linestyle='--', alpha=0.6)
# ax1.set_facecolor('#f7f7f7')  # 淡淡的灰色背景
# ax1.tick_params(axis='y', labelsize=tick_size) 

# # 调整布局
# plt.tight_layout()

# # 显示图形
# # plt.show()
# plt.savefig("methods_robustness_generation.png", dpi=400, bbox_inches="tight")


# #### robustness 只画其中一个图形 cloze
# # 数据 SAUCE
# title_size = 25  
# label_size = 22  
# tick_size = 18
# legend_size = 18

# methods = ['Forget Set', 'Test Set', 'Retain Set', 'Real Celebrity']
# origin_concrete = [7.88, 16.23, 16.85, 12.94]  # Concrete Concepts - Origin
# translation_concrete = [8.03, 18.54, 15.78, 12.46]  # Concrete Concepts - Translation
# quantization_concrete = [8.12, 18.33, 15.23, 12.33]  # Concrete Concepts - Quantization


# x = np.arange(len(methods))  # 横坐标位置
# width = 0.2  # 柱状图宽度

# # 创建图形和子图
# fig, ax1 = plt.subplots(figsize=(12, 8))

# # 绘制第一张图：Concrete Concepts
# rects1 = ax1.bar(x - width, origin_concrete, width, label='Before Attacks', color='#718DBF') # 'color='skyblue'
# rects2 = ax1.bar(x, translation_concrete, width, label='Translation Attack', color='#E58E68')  # 'lightgreen'
# rects3 = ax1.bar(x + width, quantization_concrete, width, label='Quantization Attack', color='#5CB090') # salmon'

# # 设置第一张图的属性
# ax1.set_xlabel('Dataset', fontsize=label_size)
# ax1.set_ylabel(r'ACC (%)', fontsize=label_size)
# ax1.set_title('Cloze Task', fontsize=title_size)
# ax1.set_xticks(x)
# ax1.set_xticklabels(methods, fontsize=tick_size)
# # ax1.legend(fontsize=legend_size)
# ax1.legend(fontsize=legend_size, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15))  # 调整图例位置
# ax1.grid(True, linestyle='--', alpha=0.6)
# ax1.set_facecolor('#f7f7f7')  # 淡淡的灰色背景
# ax1.tick_params(axis='y', labelsize=tick_size) 

# # 调整布局
# plt.tight_layout()

# # 显示图形
# # plt.show()
# plt.savefig("methods_robustness_cloze.png", dpi=400, bbox_inches="tight")
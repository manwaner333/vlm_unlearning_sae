import matplotlib.pyplot as plt
import numpy as np


# title_size = 25  
# label_size = 20  
# tick_size = 16
# legend_size = 13

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



##### composite accuracy
import matplotlib.pyplot as plt
import numpy as np

# 数据
num_concepts = [1, 2, 3, 4, 5, 6]  # 横坐标：Number of Concepts

# 具体概念的数据
concrete_unlearning = [85, 80, 78, 82, 84, 88]  # 具体概念的 Unlearning Accuracy
concrete_utility = [90, 88, 85, 87, 89, 92]  # 具体概念的 Utility Accuracy

# 抽象概念的数据
abstract_unlearning = [75, 70, 68, 72, 74, 78]  # 抽象概念的 Unlearning Accuracy
abstract_utility = [80, 78, 75, 77, 79, 82]  # 抽象概念的 Utility Accuracy

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制具体概念的折线
plt.plot(num_concepts, concrete_unlearning, label='Concrete Unlearning', linestyle='-', marker='o', color='blue')
plt.plot(num_concepts, concrete_utility, label='Concrete Utility', linestyle='-', marker='s', color='green')

# 绘制抽象概念的折线
plt.plot(num_concepts, abstract_unlearning, label='Abstract Unlearning', linestyle='--', marker='o', color='red')
plt.plot(num_concepts, abstract_utility, label='Abstract Utility', linestyle='--', marker='s', color='orange')

# 设置图形属性
plt.xlabel('Number of Concepts', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Accuracy vs. Number of Concepts', fontsize=14)
plt.xticks(num_concepts)
plt.yticks(np.arange(0, 101, 10))  # 纵坐标从 0 到 100，步长为 10
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)

# 设置背景颜色
plt.gca().set_facecolor('#f7f7f7')  # 淡淡的灰色背景

# 显示图形
plt.tight_layout()
# plt.show()
plt.savefig("composite.png", dpi=400, bbox_inches="tight")
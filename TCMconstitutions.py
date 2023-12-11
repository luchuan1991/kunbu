import pandas as pd

# 读取 Excel 文件
data = pd.read_excel(r'C:\Users\guangzhou\Desktop\consti.xls')

# 创建空列表，用于存储每行的体质得分
result_list = []

# 遍历每一行数据
for index, row in data.iterrows():
    # 计算每种体质的得分
    qixu = sum(row[[1, 2, 3, 13]])
    yangxu = sum(row[[10, 11, 12, 28]])
    yinxu = sum(row[[9, 20, 25, 30]])
    tanshi = sum(row[[8, 15, 27, 31]])
    shire = sum(row[[22, 24, 26, 29]])
    xueyu = sum(row[[18, 21, 23, 32]])
    qiyu = sum(row[[4, 5, 6, 7]])
    tebing = sum(row[[14, 16, 17, 19]])
    pinghe = 24 + row[0] - row[1] - row[3] - row[4] - row[12]
    # pinghe = 24 + row[0] - sum(row[[1,3,4,12]]

    # 将结果存储到字典中
    result = {"气虚质": qixu, "阳虚质": yangxu, "阴虚质": yinxu, "痰湿质": tanshi, "湿热质": shire, "血瘀质": xueyu,
              "气郁质": qiyu, "特禀质": tebing, "平和质": pinghe}

    # 添加到结果列表中
    result_list.append(result)

# 将结果列表转换成DataFrame对象
result_df = pd.DataFrame(result_list)

#
# # 定义排序函数
# def sort_constitution(row):
#     return sorted(row.items(), key=lambda x: x[1], reverse=True)
#
# # 对完整结果输出进行排序
# result_sorted = result_df.apply(sort_constitution, axis=1)
#
# # 打印排序后的结果
# for index, row in result_sorted.items():
#     print(f"第{index + 1}行体质排序: {row}")
# # 将排序后的结果保存到csv文件
# result_sorted.to_csv(r'C:\Users\guangzhou\Desktop\consti1.csv', index=False)

# #输出的结果没有分数
# # 定义排序函数
# def sort_constitution(row):
#     # 只对非平和质大于10分数的结果进行排序
#     filtered_row = {k: v for k, v in row.items() if k != '平和质' and v > 10}
#     if filtered_row:
#         # 非平和质得分大于10的情况下，进行排序
#         return sorted(filtered_row.items(), key=lambda x: x[1], reverse=True)
#     else:
#         # 否则只输出平和质结果
#         return [('平和质', row['平和质'])]
#
# # 对每行数据进行排序
# result_sorted = result_df.apply(sort_constitution, axis=1)
#
# # 将排序后的结果保存到csv文件中
# result_sorted.apply(lambda x: [i[0] for i in x]).to_csv(r'C:\Users\guangzhou\Desktop\consti2.csv', index=False)

#输出的结果有分数

# 定义排序函数
def sort_constitution(row):
    # 只对非平和质得分大于10的结果进行排序
    filtered_row = {k: v for k, v in row.items() if k != '平和质' and v > 10}
    if filtered_row:
        # 非平和质得分大于10的情况下，进行排序
        return sorted(filtered_row.items(), key=lambda x: x[1], reverse=True)
    else:
        # 否则只输出平和质结果
        return [('平和质', row['平和质'])]

# 对每行数据进行排序
result_sorted = result_df.apply(sort_constitution, axis=1)

# 将排序后的结果保存到csv文件中
result_sorted.to_csv(r'C:\Users\guangzhou\Desktop\consti2.csv', index=False)

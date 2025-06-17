import json

# 文件路径
episode_data_path = 'episode_data.jsonl'

# 初始化变量用于存储最小值和最大值
min_value = float('inf')  # 初始化为正无穷
max_value = float('-inf')  # 初始化为负无穷

# 打开文件并逐行读取
with open(episode_data_path, 'r') as file:
    for line in file:
        # 解析每一行的 JSON 数据
        episode_data = json.loads(line)
        # 获取当前 episode 的 action 数据
        action_data = episode_data['stats']['action']
        # 获取最后一维的最小值和最大值
        current_min = action_data['min'][-1]
        current_max = action_data['max'][-1]
        # 更新全局最小值和最大值
        if current_min < min_value:
            min_value = current_min
        if current_max > max_value:
            max_value = current_max

# 输出结果
print(f"最后一维的最小值: {min_value}")
print(f"最后一维的最大值: {max_value}")

#task09
# 最后一维的最小值: -3990.0 
# 最后一维的最大值: 56770.0

#Newtask02
#  最后一维的最小值: -3570.0
    # 最后一维的最大值: 63210.0

##Newtask01
# 最后一维的最小值: -3710.0
# 最后一维的最大值: 61460.0
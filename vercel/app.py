from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 加载数据
data1 = pd.read_csv('https://raw.githubusercontent.com/zh19980811/ML_ITMO_TASK2/refs/heads/main/vercel/updated_gas_oil_tasks%20(2).csv')  # 替换为你的CSV文件路径
data2 = pd.read_csv('https://raw.githubusercontent.com/zh19980811/ML_ITMO_TASK2/refs/heads/main/OneDrive_1_2024-12-13/val_rus.csv')  # 如果需要，也可以加载第二个数据集

# 创建 Flask 应用
app = Flask(__name__)

# 搜索函数
def search_task(data1, search_text):
    # 筛选出包含 search_text 的 task_name
    matching_rows = data1[data1['task_name'].str.lower().str.startswith(search_text.lower(), na=False) & 
                          data1['task_name'].str.lower().str.contains(search_text.lower(), na=False, regex=False)]
    
    # 如果没有找到任何匹配项，返回最相似的 task_name
    if matching_rows.empty:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(data1['task_name'])
        
        # 计算输入文本与所有 task_name 的相似度
        search_vector = tfidf_vectorizer.transform([search_text])
        similarity_scores = cosine_similarity(search_vector, tfidf_matrix).flatten()
        
        # 找到最相似的文本
        best_match_index = np.argmax(similarity_scores)
        
        # 返回最相似的 task_id 或 general_name
        return data1.iloc[best_match_index][['task_id', 'task_name', 'measurement', 'general_name']]

    # 如果只有一个匹配项，直接返回该 task_id
    if len(matching_rows) == 1:
        return matching_rows[['task_id', 'task_name', 'measurement', 'general_name']].iloc[0]

    # 如果有多个匹配项，计算相似度
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(matching_rows['task_name'])
    
    # 计算输入文本与每个匹配文本的相似度
    search_vector = tfidf_vectorizer.transform([search_text])
    similarity_scores = cosine_similarity(search_vector, tfidf_matrix).flatten()
    
    # 找到最相似的文本对应的行
    best_match_index = np.argmax(similarity_scores)
    
    # 返回最相似的文本的 general_name
    return matching_rows.iloc[best_match_index][['task_id', 'task_name', 'measurement', 'general_name']]

# 创建主页路由
@app.route('/')
def index():
    return render_template('index.html')

# 创建搜索功能的路由
@app.route('/search', methods=['POST'])
@app.route('/search', methods=['POST'])
def search():
    # 获取用户输入
    search_text = request.form.get('search_text')
    measurement = request.form.get('measurement')  # 获取 measurement 输入值
    
    if search_text:
        result = search_task(data1, search_text)
        
        # 检查 result 是否为空，并返回适当的模板
        if result.empty:  # 如果结果为空
            return render_template('index.html', error="No matching tasks found!")
        
        return render_template('result.html', result=result.to_dict())
    else:
        return render_template('index.html', error="Please enter a valid search text!")


if __name__ == '__main__':
    app.run(debug=True)

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
import os

# 读取数据
def read_data(record_path):
    data = []
    with open(record_path, "r") as file:
        lines = file.readlines()[2:]  # 跳过前两行
        for line in lines:
            parts = line.strip().split('|')
            pp_micro = [int(x) for x in parts[0].split(',')]
            times = [float(x) for x in parts[1].split(',')]
            data.append(pp_micro + times)

    # 转换为 NumPy 数组
    data = np.array(data)
    X = data[:, :2]  # pp_size 和 microbatch_size
    y_vision = data[:, 2]  # vision forward time
    y_text = data[:, 3]  # text forward time
    return X, y_vision, y_text
# 选择最佳多项式阶数
def best_poly_degree(X, y):
    param_grid = {'polynomialfeatures__degree': np.arange(1, 6)}
    pipeline = make_pipeline(PolynomialFeatures(), LinearRegression())
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    return grid_search.best_estimator_

# 训练模型
def train_save_model(X, y_vision, y_text, save_dir_path):
    # check if joblib file exists
    if os.path.exists(save_dir_path+"/model_vision.joblib") and os.path.exists(save_dir_path+"/model_text.joblib"):
        print("Model files already exist.")
        return
    model_vision = best_poly_degree(X, y_vision)
    model_text = best_poly_degree(X, y_text)
    dump(model_vision, save_dir_path+"/model_vision.joblib")
    dump(model_text, save_dir_path+"/model_text.joblib")

# 定义预测函数
def predict_times(modal, pp_size, microbatch_size, vL, tL):
    load_dir_path = f"{os.environ['MEGATRON_HOME']}/planner/profile_logs/FlexProfile_vL{vL}_tL{tL}"
    if modal == "vision":
        if not os.path.exists(load_dir_path+"/model_vision.joblib"):
            print(f"Model file not found in {load_dir_path}/model_vision.joblib. Please train the performance model first.")
            return None
        model = load(load_dir_path+"/model_vision.joblib")
    elif modal == "text":
        if not os.path.exists(load_dir_path+"/model_text.joblib"):
            print(f"Model file not found in {load_dir_path}/model_text.joblib. Please train the performance model first.")
            return None
        model = load(load_dir_path+"/model_text.joblib")

    predict_time = model.predict(np.array([[pp_size, microbatch_size]]))
    return predict_time[0]

if __name__ == "__main__":
    vL = 168
    tL = 88
    record_dir_path = f"{os.environ['MEGATRON_HOME']}/planner/profile_logs/FlexProfile_vL{vL}_tL{tL}"
    record_path =record_dir_path+f"/profile_record_vL{vL}_tL{tL}.txt"
    # 读取数据
    X, y_vision, y_text = read_data(record_path)
    # 训练并保存模型
    train_save_model(X, y_vision, y_text, record_dir_path)
    # 使用函数进行预测示例
    # pp_size = 4
    # microbatch_size = 4
    # vision_time, text_time = predict_times(pp_size, microbatch_size, record_dir_path)
    # print(f"Predicted Vision Time: {vision_time}")
    # print(f"Predicted Text Time: {text_time}")
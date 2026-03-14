# Iris SVM Project (BIO3510 HW2)

支持向量机（SVM, `sklearn.svm.SVC`）在鸢尾花数据集上的训练与评估（含简单参数探索）。

## 项目结构

- `data/iris.csv`: 数据集（从根目录复制一份，便于项目化管理）
- `src/iris_svm/`: 训练与评估代码
- `svm_iris.py`: 兼容入口（不安装包也能直接运行）

## 环境准备

建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## 运行方式

```bash
python svm_iris.py              # 默认会展示“调参过程”（baseline -> GridSearchCV -> 对比）
python svm_iris.py --data data/iris.csv
python svm_iris.py --kernel linear
python svm_iris.py --gridsearch --topk 5
python svm_iris.py --no-tune    # 关闭调参：只训练并评估当前参数
```

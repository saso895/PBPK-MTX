import pymc3 as pm

# 创建一个示例模型
model = pm.Model()
pr = pm.LogNormal('Rest', mu=np.log(2), sigma=0.1)
print(f"LogNormal {pr.name} created successfully.")
model.check()

print(pm.__version__)  # 查看pymc3版本
import theano
print(theano.__version__)  # 查看theano版本
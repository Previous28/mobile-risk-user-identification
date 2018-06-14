import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics

#记录程序运行时间
import time 
start_time = time.time()

train = pd.read_csv('feature/train_featureV1.csv')
test = pd.read_csv('feature/test_featureV1.csv')

dtrain = lgb.Dataset(train.drop(['uid','label'],axis=1),label=train.label)
dtest = lgb.Dataset(test.drop(['uid'],axis=1))

lgb_params =  {
  'boosting_type': 'gbdt',
  'objective': 'binary',
  'is_training_metric': False,
  'min_data_in_leaf': 40,
  'num_leaves': 16,
  'learning_rate': 0.08,
  'feature_fraction': 0.8,
  'bagging_fraction': 0.8,
  'bagging_freq': 1,
  'verbosity':-1,
  'is_unbalance': True,
  'max_depth':6,
  'feature_fraction_seed': 2, #default: 2
  'bagging_seed': 3 #default: 3
}

# 评估函数
def evalMetric(preds,dtrain):
  label = dtrain.get_label()
  
  pre = pd.DataFrame({'preds':preds,'label':label})
  pre= pre.sort_values(by='preds',ascending=False)
  
  auc = metrics.roc_auc_score(pre.label,pre.preds)

  pre.preds=pre.preds.map(lambda x: 1 if x>=0.5 else 0)

  f1 = metrics.f1_score(pre.label,pre.preds)
  
  res = 0.6*auc +0.4*f1
  
  return 'res',res,True
    
#feval：– Customized evaluation function. Note: should return (eval_name, eval_result, is_higher_better) or list of such tuples.
# 本地cv,初步观察效果
cv = lgb.cv(lgb_params,dtrain,feval=evalMetric,early_stopping_rounds=100,verbose_eval=1,num_boost_round=10000,nfold=3,metrics=['evalMetric'])

model =lgb.train(lgb_params,dtrain,feval=evalMetric,verbose_eval=10,num_boost_round=230,valid_sets=[dtrain])

# 输出feature_importance
feature_importance = model.feature_importance()
col_name = model.feature_name()
df = pd.DataFrame()
df['feature_importance'] = feature_importance
df['col_name'] = col_name
df = df.sort_values(by='feature_importance', ascending = False)
df.to_csv('feature_importance.csv',index=False,header=False,sep=',')

pred=model.predict(test.drop(['uid'],axis=1))

res =pd.DataFrame({'uid':test.uid,'label':pred})
res=res.sort_values(by='label',ascending=False)
res.label=res.label.map(lambda x: 1 if x>=0.62 else 0)
res.label = res.label.map(lambda x: int(x))

res.to_csv('result/latest.csv',index=False,header=False,sep=',',columns=['uid','label'])

#输出运行时长
cost_time = time.time()-start_time
print ("cost time:",cost_time,"(s)......")

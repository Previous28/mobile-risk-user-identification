import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#记录程序运行时间
import time 
start_time = time.time()

# 获取通话时间
def get_talk_time(itime):
  day = int(itime[0:2])
  hour = int(itime[2:4])
  minute = int(itime[4:6])
  second = int(itime[6:8])
  talk_time = day*86400 + hour*3600 + minute*60 + second
  return talk_time

# 获取日期
def get_date(stime):
  return stime[0:2]

uid_train = pd.read_csv('train/uid_train.txt',sep='\t',header=None,names=('uid','label'))
voice_train = pd.read_csv('train/voice_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_train = pd.read_csv('train/sms_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_train = pd.read_csv('train/wa_train.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})

voice_test = pd.read_csv('Test-B/voice_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_test = pd.read_csv('Test-B/sms_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_test = pd.read_csv('Test-B/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})

uid_test = pd.DataFrame({'uid':pd.unique(wa_test['uid'])})
uid_test.to_csv('Test-B/uid_test_b.txt',index=None)

voice = pd.concat([voice_train,voice_test],axis=0)
sms = pd.concat([sms_train,sms_test],axis=0)
wa = pd.concat([wa_train,wa_test],axis=0)

# voice
voice_feature = pd.DataFrame()
# 统计每个用户的通话总数
x = voice.groupby('uid')['opp_num'].apply(lambda x: x.count())
voice_feature['uid'] = x.index
voice_feature['voice_opp_count_all'] = x.values
# 统计每个用户的不同通话对象数
x = voice.groupby('uid')['opp_num'].apply(lambda x: len(set(x)))
voice_feature['voice_opp_count_unique'] = x.values
# 分别统计每个用户接入打出通话总数
x = voice.groupby(['uid','in_out'])['opp_num'].apply(lambda x: x.count())
x = x.unstack(fill_value=0).reset_index(drop=True)
x.columns = ['0', '1']
voice_feature['voice_opp_count_out'] = x['0']
voice_feature['voice_opp_count_in'] = x['1']
# 分别统计每个用户不同通话类型的通话总数
x = voice.groupby(['uid','call_type'])['opp_num'].apply(lambda x: x.count())
x = x.unstack(fill_value=0).reset_index(drop=True)
x.columns = ['1', '2','3','4','5']
voice_feature['voice_count_type1'] = x['1']
voice_feature['voice_count_type2'] = x['2']
voice_feature['voice_count_type3'] = x['3']
voice_feature['voice_count_type4'] = x['4']
voice_feature['voice_count_type5'] = x['5']

# 以下分别统计每个用户接入,打出的电话号码长度的均值,最大值,和
x = voice.groupby(['uid','in_out'])['opp_len'].apply(lambda x:x.mean())
x = x.unstack(fill_value=0).reset_index(drop=True)
x.columns = ['0','1']
voice_feature['voice_opp_len_mean_out'] = x['0']
voice_feature['voice_opp_len_mean_in'] = x['1']

x = voice.groupby(['uid','in_out'])['opp_len'].apply(lambda x:x.max())
x = x.unstack(fill_value=0).reset_index(drop=True)
x.columns = ['0','1']
voice_feature['voice_opp_len_max_out'] = x['0']
voice_feature['voice_opp_len_max_in'] = x['1']

x = voice.groupby(['uid','in_out'])['opp_len'].apply(lambda x:x.sum())
x = x.unstack(fill_value=0).reset_index(drop=True)
x.columns = ['0','1']
voice_feature['voice_opp_len_sum_out'] = x['0']
voice_feature['voice_opp_len_sum_in'] = x['1']

# 以下分别统计每个用户不同通话类型的电话号码长度的均值,最大值,和
x = voice.groupby(['uid','call_type'])['opp_len'].apply(lambda x:x.mean())
x = x.unstack(fill_value=0).reset_index(drop=True)
x.columns = ['1','2','3','4','5']
voice_feature['voice_opp_len_mean_ct1'] = x['1']
voice_feature['voice_opp_len_mean_ct2'] = x['2']
voice_feature['voice_opp_len_mean_ct3'] = x['3']
voice_feature['voice_opp_len_mean_ct4'] = x['4']
voice_feature['voice_opp_len_mean_ct5'] = x['5']

x = voice.groupby(['uid','call_type'])['opp_len'].apply(lambda x:x.max())
x = x.unstack(fill_value=0).reset_index(drop=True)
x.columns = ['1','2','3','4','5']
voice_feature['voice_opp_len_max_ct1'] = x['1']
voice_feature['voice_opp_len_max_ct2'] = x['2']
voice_feature['voice_opp_len_max_ct3'] = x['3']
voice_feature['voice_opp_len_max_ct4'] = x['4']
voice_feature['voice_opp_len_max_ct5'] = x['5']

x = voice.groupby(['uid','call_type'])['opp_len'].apply(lambda x:x.sum())
x = x.unstack(fill_value=0).reset_index(drop=True)
x.columns = ['1','2','3','4','5']
voice_feature['voice_opp_len_sum_ct1'] = x['1']
voice_feature['voice_opp_len_sum_ct2'] = x['2']
voice_feature['voice_opp_len_sum_ct3'] = x['3']
voice_feature['voice_opp_len_sum_ct4'] = x['4']
voice_feature['voice_opp_len_sum_ct5'] = x['5']

# 每个用户通话时间的六个聚合值:标准差,最大,最小,中值,平均值,和
voice['talk_time'] = voice['end_time'].apply(get_talk_time) - voice['start_time'].apply(get_talk_time)
talk_time = voice.groupby(['uid'])['talk_time'].agg(['std','max','min','median','mean','sum']).add_prefix('voice_talk_time_').reset_index()
# 每个用户通话记录的有效天数
voice['date'] = voice['end_time'].apply(get_date)
voice_date = voice.groupby(['uid'])['date'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_date_').reset_index()
# 每个用户不同通话号码头部的计数
voice_opp_head=voice.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_').reset_index()

voice_opp_len=voice.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)
# 统计每个用户通话号码长度,通话类型,接入打出的聚合值
voice_opp_len_data = voice.groupby(['uid'])['opp_len'].agg(['std','max','min','median','mean','sum']).add_prefix('voice_opp_len_').reset_index()
voice_call_type_data = voice.groupby(['uid'])['call_type'].agg(['std','max','median','mean','sum']).add_prefix('voice_call_type_').reset_index()
voice_in_out_data = voice.groupby(['uid'])['in_out'].agg(['std','median','mean','sum']).add_prefix('voice_in_out_').reset_index()

# sms
sms_feature = pd.DataFrame()
# 统计每个用户的短信总数
x = sms.groupby('uid')['opp_num'].apply(lambda x: x.count())
sms_feature['uid'] = x.index
sms_feature['sms_opp_count_all'] = x.values
# 统计每个用户的不同短信对象数
x = sms.groupby('uid')['opp_num'].apply(lambda x: len(set(x)))
sms_feature['sms_opp_count_unique'] = x.values
# 分别统计每个用户收到,发出的短信总数
x = sms.groupby(['uid','in_out'])['opp_num'].apply(lambda x: x.count())
x = x.unstack(fill_value=0).reset_index(drop=True)
x.columns = ['0', '1']
sms_feature['sms_opp_count_out'] = x['0']
sms_feature['sms_opp_count_in'] = x['1']

x = sms.groupby(['uid','in_out'])['opp_len'].apply(lambda x:x.mean())
x = x.unstack(fill_value=0).reset_index(drop=True)
x.columns = ['0','1']
sms_feature['sms_opp_len_mean_out'] = x['0']
sms_feature['sms_opp_len_mean_in'] = x['1']

x = sms.groupby(['uid','in_out'])['opp_len'].apply(lambda x:x.max())
x = x.unstack(fill_value=0).reset_index(drop=True)
x.columns = ['0','1']
sms_feature['sms_opp_len_max_out'] = x['0']
sms_feature['sms_opp_len_max_in'] = x['1']

x = sms.groupby(['uid','in_out'])['opp_len'].apply(lambda x:x.sum())
x = x.unstack(fill_value=0).reset_index(drop=True)
x.columns = ['0','1']
sms_feature['sms_opp_len_sum_out'] = x['0']
sms_feature['sms_opp_len_sum_in'] = x['1']

x = sms.groupby(['uid','in_out'])['opp_head'].apply(lambda x:x.mean())
x = x.unstack(fill_value=0).reset_index(drop=True)
x.columns = ['0','1']
sms_feature['sms_opp_head_mean_out'] = x['0']
sms_feature['sms_opp_head_mean_in'] = x['1']

x = sms.groupby(['uid','in_out'])['opp_head'].apply(lambda x:x.max())
x = x.unstack(fill_value=0).reset_index(drop=True)
x.columns = ['0','1']
sms_feature['sms_opp_head_max_out'] = x['0']
sms_feature['sms_opp_head_max_in'] = x['1']

x = sms.groupby(['uid','in_out'])['opp_head'].apply(lambda x:x.sum())
x = x.unstack(fill_value=0).reset_index(drop=True)
x.columns = ['0','1']
sms_feature['sms_opp_head_sum_out'] = x['0']
sms_feature['sms_opp_head_sum_in'] = x['1']

sms_opp_head=sms.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_head_').reset_index()
sms_opp_len=sms.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)
sms_opp_len_data = sms.groupby(['uid'])['opp_len'].agg(['std','max','min','median','mean','sum']).add_prefix('sms_opp_len_').reset_index()
sms_in_out_data = sms.groupby(['uid'])['in_out'].agg(['std','max','min','median','mean','sum']).add_prefix('sms_in_out_').reset_index()

# 同一时刻发出,收到的短信数
temp = sms.groupby(['uid','start_time'])['in_out'].count().unstack().fillna(0).max(axis=1)
sms_send_same_time = pd.DataFrame()
sms_send_same_time['uid'] = temp.index
sms_send_same_time['sms_send_same_time'] = temp.values
# 每个用户短信记录的有效天数
sms['date'] = sms['start_time'].apply(get_date)
sms_date = sms.groupby(['uid'])['date'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_date_').reset_index()

# wa
wa_feature = pd.DataFrame()
x = wa.groupby('uid')['wa_name'].apply(lambda x: len(set(x)))
wa_feature['uid'] = x.index
wa_feature['wa_name_count_unique'] = x.values
# 根据网站和软件类型的不同计算每种类别的数量
x = wa.groupby(['uid','wa_type'])['wa_name'].apply(lambda x: x.count())
x = x.unstack(fill_value=0).reset_index(drop=True)
x.columns = ['0', '1']
wa_feature['wa_count_type0'] = x['0']
wa_feature['wa_count_type1'] = x['1']
# 软件,网站不同特征的聚合值
visit_cnt = wa.groupby(['uid'])['visit_cnt'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_cnt_').reset_index()
visit_dura = wa.groupby(['uid'])['visit_dura'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_dura_').reset_index()
up_flow = wa.groupby(['uid'])['up_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_up_flow_').reset_index()
down_flow = wa.groupby(['uid'])['down_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_down_flow_').reset_index()

feature = [voice_feature,sms_feature,wa_feature,talk_time,voice_date,voice_opp_head,voice_opp_len,voice_opp_len_data,voice_call_type_data,
          voice_in_out_data,sms_opp_head,sms_opp_len,sms_opp_len_data,sms_in_out_data,sms_send_same_time,sms_date,visit_cnt,visit_dura,
          up_flow,down_flow]

train_feature = uid_train
for feat in feature:
  train_feature=pd.merge(train_feature,feat,how='left',on='uid')

test_feature = uid_test
for feat in feature:
  test_feature=pd.merge(test_feature,feat,how='left',on='uid')

train_feature.to_csv('feature/train_feature.csv',index=None)
test_feature.to_csv('feature/test_feature.csv',index=None)

cost_time = time.time()-start_time
print ("cost time:",cost_time,"(s)......")

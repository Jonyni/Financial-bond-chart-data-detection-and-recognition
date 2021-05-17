#coding=utf-8
import os
#读取预测文件
predict_dict = {}
label_dict={}
with open("data.txt", "r",encoding='utf-8') as f:  # 打开文件
    data = f.readlines()  # 读取文件
    for da in data:
        # print(da)
        predict_dict[da.split('，')[0]]=da.split('，')[1].strip('\n')
    # print(predict_dict)
datas=[]
with open("image_info_A_2000.txt", "r", encoding='utf-8') as f:  # 打开文件
    data = f.readlines()  # 读取文件
    for da in data:
        # print(da)
        datas.append(da.split('	')[0])
        label_dict[da.split('	')[0]] = da.split('	')[1].strip('\n')
    # print(label_dict)
count=0
num=0
if len(predict_dict)==len(label_dict):
    num=len(predict_dict)
    print (datas)
for imgname in datas:
    if predict_dict[imgname]==label_dict[imgname]:
        count+=1
print("{}/{},acc:{}".format(count,num,count/num))

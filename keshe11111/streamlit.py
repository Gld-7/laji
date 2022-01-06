import torch
import torchvision
import streamlit as st
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import time
from subprocess import Popen, PIPE
import streamlit.components.v1 as components
net=torch.load('keshe11111/resnet18.pkl', map_location='cpu')
net.eval()
#table=['hazardous_waste_dry_battery','hazardous_waste_expired_drugs','hazardous_waste_ointment','厨余垃圾_bone','厨余垃圾_eggshell','厨余垃圾_fish_bone','厨余垃圾_fruit_peel','厨余垃圾_meal','厨余垃圾_pulp','厨余垃圾_tea','厨余垃圾_vegetable','其他垃圾_bamboo_chopsticks','其他垃圾_cigarette','其他垃圾_fast_food_box','其他垃圾_flowerpot','其他垃圾_soiled_plastic','其他垃圾_toothpick','可回收垃圾_anvil','可回收垃圾_bag','可回收垃圾_bottle','可回收垃圾_can','可回收垃圾_cardboard','可回收垃圾_cosmetic_bottles','可回收垃圾_drink_bottle','可回收垃圾_edible_oil_barrel','可回收垃圾_glass_cup','可回收垃圾_metal_food_cans','可回收垃圾_old_clothes','可回收垃圾_paper_bags','可回收垃圾_pillow','可回收垃圾_plastic_bowl','可回收垃圾_plastic_hanger','可回收垃圾_plug_wire','可回收垃圾_plush_toys','可回收垃圾_pot','可回收垃圾_powerbank','可回收垃圾_seasoning_bottle','可回收垃圾_shampoo_bottle','可回收垃圾_shoes','可回收垃圾_toys']
table=['有害垃圾_干电池','有害垃圾_过期药品','有害垃圾_药膏','厨余垃圾_骨头','厨余垃圾_蛋壳','厨余垃圾_鱼骨头','厨余垃圾_水果皮','厨余垃圾_剩饭','厨余垃圾_烂水果','厨余垃圾_茶叶','厨余垃圾_蔬菜','其他垃圾_竹筷','其他垃圾_烟头','其他垃圾_快餐盒','其他垃圾_碎瓷器','其他垃圾_塑料袋','其他垃圾_牙签','可回收垃圾_砧板','可回收垃圾_包','可回收垃圾_酒瓶','可回收垃圾_易拉罐','可回收垃圾_纸箱','可回收垃圾_化妆品瓶','可回收垃圾_饮料瓶','可回收垃圾_食用油桶','可回收垃圾_玻璃杯','可回收垃圾_金属罐','可回收垃圾_旧衣服','可回收垃圾_纸袋','可回收垃圾_枕头','可回收垃圾_塑料碗','可回收垃圾_塑料衣架','可回收垃圾_插头电线','可回收垃圾_毛绒玩具','可回收垃圾_锅','可回收垃圾_充电宝','可回收垃圾_调味品瓶','可回收垃圾_洗发水瓶','可回收垃圾_鞋子','可回收垃圾_玩具']

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])
])
def main():
    st.sidebar.markdown('<h3>机器学习垃圾分类课程设计</h3>', unsafe_allow_html=True)
    st.sidebar.markdown('<h5>组员：童亨  侯照坤  陈鹏飞  花雪龙  方翔</h5>', unsafe_allow_html=True)
    lable1 = st.sidebar.radio('', ('首页', '垃圾分类小知识', '模型性能测试', '识别垃圾'))
    if lable1 == '首页':
        st.markdown('<h1 style="color:DarkRed;text-align:center;font-family:KaiTi">机器学习垃圾分类课程设计</h1>', unsafe_allow_html=True)
        html_temp1 = """
                                <h2 style="background-color:rgb(90,130,100);padding:18px;color:Gainsboro;text-align:center">童亨 2019329621065</h2>
                                """
        st.markdown(html_temp1, unsafe_allow_html=True)

        html_temp2 = """
                                <h2 style="background-color:rgb(110,150,120);padding:16px;color:Gainsboro;text-align:center">侯照坤 2019329621054</h2>
                                """
        st.markdown(html_temp2, unsafe_allow_html=True)

        html_temp3 = """
                                <h2 style="background-color:rgb(130,170,140);padding:14px;color:Gainsboro;text-align:center">陈鹏飞 2019329621047</h2>
                                """
        st.markdown(html_temp3, unsafe_allow_html=True)

        html_temp4 = """
                                <h2 style="background-color:rgb(150,190,160);padding:12px;color:Gainsboro;text-align:center">花雪龙 2019329621055</h2>
                                """
        st.markdown(html_temp4, unsafe_allow_html=True)

        html_temp5 = """        
                                <h2 style="background-color:rgb(160,200,170);padding:10px;color:Gainsboro;text-align:center">方翔 2019329621051</h2>
                                """
        st.markdown(html_temp5, unsafe_allow_html=True)


    elif lable1 == '垃圾分类小知识':
        welcome()
        st.markdown('')
        if st.checkbox('有害垃圾'):
            st.markdown('有害垃圾含有对人体健康有害的重金属、有毒的物质或者对环境造成现实危害或者潜在危害的废弃物。包括电池、荧光灯管、灯泡、水银温度计、油漆桶、部分家电、过期药品及其容器、过期化妆品等。这些垃圾一般使用单独回收或填埋处理。')
            youhai = Image.open('keshe11111/有害垃圾.jpg')
            st.image(youhai, caption='有害垃圾',width=500)
        if st.checkbox('厨余垃圾'):
            st.markdown('厨余垃圾（上海称湿垃圾）包括剩菜剩饭、骨头、菜根菜叶、果皮等食品类废物。经生物技术就地处理堆肥，每吨可生产0.6~0.7吨有机肥料。')
            chuyu = Image.open('keshe11111/厨余垃圾.jpg')
            st.image(chuyu, caption='厨余垃圾', width=500)
        if st.checkbox('其他垃圾'):
            st.markdown('其他垃圾（上海称干垃圾）包括除上述几类垃圾之外的砖瓦陶瓷、渣土、卫生间废纸、纸巾等难以回收的废弃物及尘土、食品袋（盒）。采取卫生填埋可有效减少对地下水、地表水、土壤及空气的污染。')
            qita = Image.open('keshe11111/其他垃圾.jpg')
            st.image(qita, caption='其他垃圾', width=600)
        if st.checkbox('可回收垃圾'):
            st.markdown('可回收物主要包括废纸、塑料、玻璃、金属和布料五大类。废纸：主要包括报纸、期刊、图书、各种包装纸等。但是，要注意纸巾和厕所纸由于水溶性太强不可回收。塑料：各种塑料袋、塑料泡沫、塑料包装（快递包装纸是其他垃圾/干垃圾）、一次性塑料餐盒餐具、硬塑料、塑料牙刷、塑料杯子、矿泉水瓶等。玻璃：主要包括各种玻璃瓶、碎玻璃片、暖瓶等。（镜子是其他垃圾/干垃圾）金属物：主要包括易拉罐、罐头盒等。布料：主要包括废弃衣服、桌布、洗脸巾、书包、鞋等。这些垃圾通过综合处理回收利用，可以减少污染，节省资源。')
            kehuishou = Image.open('keshe11111/可回收垃圾.jpg')
            st.image(kehuishou, caption='可回收垃圾', width=500)
        if st.checkbox('宣传视频'):
            video_file = open('垃圾分类小知识.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

    elif lable1=='模型性能测试':
        st.markdown('<h3 style="color:DarkRed;font-family:KaiTi">请上传需要测试的数据集及其对应标签</h3>',
                    unsafe_allow_html=True)
        image = st.file_uploader('批量上传测试文件', type='jpg',accept_multiple_files=True)
        lable=st.file_uploader('上传标签文件',type='csv')
        t1=time.perf_counter()
        if st.button('开始测试'):
            if image and lable is not None:
                data=[]
                for i in image:
                    test = np.array(bytearray(i.read()), dtype=np.uint8)
                    test = cv2.imdecode(test, 1)
                    test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
                    #st.image(test, channels='RGB')
                    test = Image.fromarray(test)
                    test = transform(test)
                    test = test.unsqueeze(0)
                    data.append(table[net(test).argmax()])
                print(data)
                sub=[]
                lable=str(lable.read().decode('utf-8'))
                string = ''
                for i in lable:

                    if i != ',':
                        string+=i
                    else:
                        sub.append(string)
                        string=''
                acc=0
                for i in range(len(data)):
                    if data[i]==sub[i]:acc+=1
                acc=acc/len(data)
                t2=time.perf_counter()
                st.success('测试准确率为：{}%'.format(acc*100))
                st.markdown('<h6 style="color:grey">输出用时：{}s</h6>'.format(t2-t1), unsafe_allow_html=True)




    elif lable1 == '识别垃圾':
        st.markdown('<h2 style="color:DarkRed;font-family:KaiTi">请上传需要识别的垃圾照片</h2>',
                    unsafe_allow_html=True)
        image=st.file_uploader('',type='jpg')

        if image is not None:
            image=np.array(bytearray(image.read()),dtype=np.uint8)
            image=cv2.imdecode(image,1)

            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image,channels='RGB')
            image=Image.fromarray(image)
            image=transform(image)
            print(image.shape)
            image=image.unsqueeze(0)
            tic1=time.perf_counter()
            print(image.shape)
            a1=net(image)

            max1 = a1.argmax()
            a1[0][max1] = -100
            if table[max1]=='可回收垃圾_砧板':
                max1=a1.argmax()
                a1[0][max1] = -100
                max2 = a1.argmax()
                a1[0][max2] = -100
                max3 = a1.argmax()
            else:
                max2 = a1.argmax()
                a1[0][max2] = -100
                max3 = a1.argmax()
            tic2 = time.perf_counter()
            if st.button('识别垃圾'):
                st.success('识别此垃圾最可能为：{}'.format(table[max1]))
                st.markdown('<h6 style="color:grey">第二可能为：{}</h6>'.format(table[max2]), unsafe_allow_html=True)
                st.markdown('<h6 style="color:grey">第三可能为：{}</h6>'.format(table[max3]), unsafe_allow_html=True)
                st.markdown('<h6 style="color:grey">输出用时：{}s</h6>'.format(tic2-tic1), unsafe_allow_html=True)



def welcome():
    html_temp = """
                        <div style="background-color:rgb(107,194,53);padding:10px">
                        <h2 style="color:white;text-align:center;font-family:KaiTi">垃圾分类小知识</h2>
                        </div>
                        """
    st.markdown(html_temp, unsafe_allow_html=True)

if __name__ == '__main__':
    main()

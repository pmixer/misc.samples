# -*- coding: utf-8 -*- 

import webbrowser
import os
import sys
import importlib
importlib.reload(sys)
# sys.setdefaultencoding('utf-8')

estates = ['兰园', '西山华府', '菊园', '竹园', '百草园', '菊花盛苑', '绿苑',
        '中海枫涟山庄', '景和园', '六里屯二区', '六里屯三区', '六里屯四区',
        '六里屯五区', '春晖园', '夏霖园', '秋露园', '冬晴园', '紫成嘉园',
        '东馨园', '东旭园', '唐家岭新城', '图灵嘉园', '上地东里', '宜品上层',
        '怡美家园', '柳浪家园东里', '上地西里', '圆明园西路2号院', '六郎庄新村',
        '上地佳园', '毛纺厂西小区', '马连洼1号院', '金隅美和园', '锦顺佳园',
        '加气混凝土厂宿舍', '裕和嘉园', '621小区', '和韵家园', '辉煌国际',
        '领秀新硅谷1号院', '领秀新硅谷2号院', '领秀硅谷', '回龙观新村东区',
        '回龙观新村中区', '回龙观新村西区', '金域华府', '蓝天嘉园',
        '龙兴园中区', '龙兴园北区', '龙兴园西区', '北京人家', '动力厂宿舍',
        '清上园', '上林溪', '小营西路23号院', '燕尚园', '智学苑', '龙兴园南区',
        '铭科苑', '天巢园', 'iMOMA', '博雅德园', '当代城市家园', '金达园',
        '融泽嘉园', '万树园', '博雅西园', '二拨子新村东区', '二拔子新村西区']

zz_url = u'https://www.ziroom.com/z/z2/?qwd='

command = u'open --new -a "Google Chrome" --args '

import pdb

for estate in estates:
    # url = repr(zz_url + estate).encode('utf-8').decode('gbk')
    # webbrowser.open(zz_url + estate)
    command += ' "' + zz_url + estate + '"'
    # pdb.set_trace()
    # webbrowser.open(str(repr(url[1:-1]).encode('utf-8'))[3:-2])

os.system(command)

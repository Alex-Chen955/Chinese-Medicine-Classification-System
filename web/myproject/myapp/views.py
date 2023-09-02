from django.shortcuts import render
import os
import base64
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from django.http import JsonResponse
from django.shortcuts import render

# Load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 100)
path = r"myapp/resnet50_model.pth"
os_path = os.path.normpath(path)
model.load_state_dict(torch.load(os_path))
model.to(device)
model.eval()
herbs = {
    0: "安息香",
    1: "白扁豆",
    2: "白矾",
    3: "白莲",
    4: "白茅根",
    5: "白前",
    6: "白芍",
    7: "白芷",
    8: "柏子仁",
    9: "北沙参",
    10: "碧波",
    11: "荜澄茄",
    12: "鳖甲",
    13: "槟榔",
    14: "苍术",
    15: "草豆蔻",
    16: "沉香",
    17: "川楝子",
    18: "川木香",
    19: "川牛膝",
    20: "大腹皮",
    21: "淡豆豉",
    22: "稻芽",
    23: "地龙",
    24: "冬虫夏草",
    25: "防风",
    26: "蕃瀉叶",
    27: "蜂房",
    28: "甘草",
    29: "干姜",
    30: "甘松",
    31: "藁本",
    32: "赤石脂",
    33: "枸杞子",
    34: "桂枝",
    35: "谷精草",
    36: "谷芽",
    37: "海龙",
    38: "海螵蛸",
    39: "合欢皮",
    40: "黄柏",
    41: "黄芪",
    42: "黄芩",
    43: "湖北贝母",
    44: "僵蚕",
    45: "芥子",
    46: "鸡冠花",
    47: "锦灯笼",
    48: "鸡内金",
    49: "荆芥穗",
    50: "金果榄",
    51: "金钱白花蛇",
    52: "九香虫",
    53: "橘核",
    54: "苦地丁",
    55: "莱菔子",
    56: "莲房",
    57: "连栩",
    58: "莲子",
    59: "莲子心",
    60: "灵芝",
    61: "荔枝核",
    62: "龙眼肉",
    63: "卢根",
    64: "路路通",
    65: "麦冬",
    66: "母丁香",
    67: "羌活",
    68: "千年健",
    69: "秦皮",
    70: "全蝎",
    71: "忍冬藤",
    72: "人参",
    73: "肉豆蔻",
    74: "桑寄生",
    75: "桑螵蛸",
    76: "桑椹",
    77: "山茨菇",
    78: "山奈",
    79: "山茱萸",
    80: "沙苑子",
    81: "石榴皮",
    82: "丝瓜络",
    83: "酸枣仁",
    84: "苏木",
    85: "太子参",
    86: "天花粉",
    87: "天麻",
    88: "土荆皮",
    89: "瓦楞子",
    90: "五加皮",
    91: "细辛",
    92: "银柴胡",
    93: "薏苡仁",
    94: "郁金",
    95: "浙贝母",
    96: "枳壳",
    97: "竹茹",
    98: "猪牙皂",
    99: "自然铜"
}

def index(request):
    if request.method == "POST":
        image = request.FILES.get('image')
        if image:
            img = Image.open(image)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img = transform(img).unsqueeze(0)
            img = img.to(device)
            output = model(img)
            _, pred = torch.max(output, 1)
            result = int(pred.item())
            result = herbs[result]
            print("Prediction result:", result)
            print(result)
            return JsonResponse({"result": result})
        else:
            return JsonResponse({"error": "No image received."})

    return render(request, "myapp/index.html")



def predict(request):
    if request.method == "POST":
        data = request.POST.get("image")
        img_data = data.split(",")[1]
        img = Image.open(io.BytesIO(base64.b64decode(img_data)))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        output = model(img)
        _, pred = torch.max(output, 1)
        result = int(pred.item())
        print(result)
        return JsonResponse({"result": result})
    return JsonResponse({"error": "Invalid request"})

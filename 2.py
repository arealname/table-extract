
image_path = '/nfsshare/home/chengwenjie/pro/Extract/llm_related/table_extract/imgs/2.jpg'

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
table_recognition = pipeline(Tasks.table_recognition, model='./dg/iic/cv_dla34_table-structure-recognition_cycle-centernet')
result = table_recognition(image_path)


from paddleocr import PaddleOCR
ocr = PaddleOCR(use_gpu=True, lang='ch')

res = ocr.ocr(image_path, cls=True)
print(res)


from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
def draw_ocr_boxes(image_path, boxes, texts):
   
    img = Image.open(image_path)
    img = Image.new('RGB', img.size, (255, 255, 255))
    
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("./chinese_cht.ttf", size=15)  
    

    # 遍历每个文本框和对应的文本
    for box, text in zip(boxes, texts):
        draw.rectangle(box, outline='red', width=2)
        x, y = box[:2]
        draw.text((x,y), text, font=font, fill='black')
    
    img.save('image_with_boxes_and_text.jpg')

# 示例文本框坐标和对应的文字
boxes = [(*i[0][0],*i[0][2]) for i in res[0]]
texts = [i[1][0] for i in res[0]]
draw_ocr_boxes(image_path, boxes, texts)


def is_inside_text(cell, text):
    """检查文字是否完全在单元格内"""
    cx1, cy1, cx2, cy2 = cell
    tx1, ty1, tx2, ty2 = text['coords']
    return cx1 <= tx1 and cy1 <= ty1 and cx2 >= tx2 and cy2 >= ty2
def calculate_iou(cell, text):
    """
    计算两个矩形框的交并比（IoU）。
    
    :param cell: 单元格的坐标 (x1, y1, x2, y2)
    :param text: 文本框的坐标 (x1, y1, x2, y2)
    :return: 交并比（IoU）
    """
    # 计算交集的左上角和右下角坐标
    intersection_x1 = max(cell[0], text['coords'][0])
    intersection_y1 = max(cell[1], text['coords'][1])
    intersection_x2 = min(cell[2], text['coords'][2])
    intersection_y2 = min(cell[3], text['coords'][3])

    # 如果没有交集，返回 0
    if intersection_x1 >= intersection_x2 or intersection_y1 >= intersection_y2:
        return 0.0

    # 计算交集的面积
    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)

    # 计算并集的面积
    area_box1 = (cell[2] - cell[0]) * (cell[3] - cell[1])
    area_box2 = (text['coords'][2] - text['coords'][0]) * (text['coords'][3] - text['coords'][1])
    union_area = area_box1 + area_box2 - intersection_area

    # 计算 IoU
    iou = intersection_area / union_area

    return iou
def calculate_iot(cell, text):
    """
    计算两个矩形框的交集面积和文本框面积的比值（IoT）。
    
    :param cell: 单元格的坐标 (x1, y1, x2, y2)
    :param text: 文本框的坐标 (x1, y1, x2, y2)
    :return: IoT
    """
    # 计算交集的左上角和右下角坐标
    intersection_x1 = max(cell[0], text['coords'][0])
    intersection_y1 = max(cell[1], text['coords'][1])
    intersection_x2 = min(cell[2], text['coords'][2])
    intersection_y2 = min(cell[3], text['coords'][3])

    # 如果没有交集，返回 0
    if intersection_x1 >= intersection_x2 or intersection_y1 >= intersection_y2:
        return 0.0
    # 计算交集的面积
    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)

    text_area = (text['coords'][2] - text['coords'][0]) * (text['coords'][3] - text['coords'][1])
    # 计算 IoT
    iot = intersection_area / text_area
    return iot

def merge_text_into_cells(cell_coords, ocr_results):
    """将文字合并到单元格"""
    # 创建一个字典，键是单元格坐标，值是属于该单元格的文字列表
    cell_text_dict = {cell: [] for cell in cell_coords}
    noncell_text_dict = {}
    
    # 遍历 OCR 结果，将文字分配给正确的单元格
    for cell in cell_coords:
        for result in ocr_results:
            if calculate_iot(cell, result)>0.5:
                cell_text_dict[cell].append(result['text'])
    
    for result in ocr_results:
        if all(calculate_iot(cell, result)<0.1 for cell in cell_coords):
            noncell_text_dict[result['coords']] = result['text']

    merged_text = {}
    for cell, texts in cell_text_dict.items():
        merged_text[cell] = ''.join(texts).strip()
    for coords, text in noncell_text_dict.items():
        merged_text[coords] = ''.join(text).strip()
    
    return merged_text

cell_coords = [tuple([*i[:2],*i[4:6]]) for i in result['polygons']]
ocr_results = [
    {'text': i[1][0], 'coords': tuple([*i[0][0],*i[0][2]])} for i in res[0]]
merged_text = merge_text_into_cells(cell_coords, ocr_results)
print(merged_text)


from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
def draw_text_boxes(image_path, boxes, texts):
    # 加载图像
    img = Image.open(image_path)
    img = Image.new('RGB', img.size, (255, 255, 255))
    # 创建一个 ImageDraw 对象
    draw = ImageDraw.Draw(img)
    
    # 设置字体
    font = ImageFont.truetype("./chinese_cht.ttf", size=15)  # 选择合适的字体和大小
    

    # 遍历每个文本框和对应的文本
    for box, text in zip(boxes, texts):
        # 绘制文本框
        draw.rectangle(box, outline='red', width=2)
       
        
        text_len = draw.textbbox(xy=box[:2], text=text, font=font)
        
        if (text_len[2]-text_len[0]) > (box[2] - box[0]):
            # 如果文本长度大于文本框宽度,则将文本换行
            text = '\n'.join(textwrap.wrap(text, width=int(np.ceil((len(text) / np.ceil((text_len[2]-text_len[0]) / (box[2] - box[0])))))))
        else:
            # 否则直接绘制文本
            text = text
        x, y = box[:2]
        
        # 在文本框内居中文本
        draw.text((x,y), text, font=font, fill='black')
    
    # 保存带有文本框和文字的图像
    img.save("output1.png")

# 示例文本框坐标和对应的文字
boxes = list(merged_text.keys())
texts = list(merged_text.values())
draw_text_boxes(image_path, boxes, texts)




def adjust_coordinates(merged_text, image_path):
    
    image = Image.open(image_path)
    width, height = image.size
    threshold = height / 100
    groups = {}
    
    for coordinates, text in merged_text.items():
        # 查找与当前 y 坐标相差不超过 threshold 的分组
        found_group = False
        for group_y in groups.keys():
            if abs(coordinates[1] - group_y) <= threshold:
                groups[group_y].append((coordinates,text))
                found_group = True
                break

        # 如果没有找到合适的分组，则创建一个新的分组
        if not found_group:
            groups[coordinates[1]] = [(coordinates,text)]
    
    # 计算每个分组的 y 坐标的平均值，并更新坐标列表
    adjusted_coordinates = {}
    for group_y, group_coords in groups.items():
        avg_y = sum(coord[0][1] for coord in group_coords) / len(group_coords)
        for i in group_coords:
            adjusted_coordinates[(i[0][0], avg_y, i[0][2], i[0][3])] = i[1]
        

    return adjusted_coordinates

# 调用函数处理坐标
adjusted_merged_text = adjust_coordinates(merged_text, image_path)

# 打印结果
print("原始坐标:", merged_text)
print("调整后的坐标:", adjusted_merged_text)


from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
def draw_text_boxes(image_path, boxes, texts):
   
    img = Image.open(image_path)
    img = Image.new('RGB', img.size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("./chinese_cht.ttf", size=15)  # 选择合适的字体和大小
    for box, text in zip(boxes, texts):
        
        draw.rectangle(box, outline='red', width=2)
       
        
        text_len = draw.textbbox(xy=box[:2], text=text, font=font)
        
        if (text_len[2]-text_len[0]) > (box[2] - box[0]):
            # 如果文本长度大于文本框宽度,则将文本换行
            text = '\n'.join(textwrap.wrap(text, width=int(np.ceil(len(text) / np.ceil((text_len[2]-text_len[0]) / (box[2] - box[0]))))))
        else:
            # 否则直接绘制文本
            text = text
        x, y = box[:2]
        
        draw.text((x,y), text, font=font, fill='black')
    img.save('output2.png')

boxes = list(adjusted_merged_text.keys())
texts = list(adjusted_merged_text.values())
draw_text_boxes(image_path, boxes, texts)


#输出最终的文本
adjusted_merged_text_sorted = sorted(adjusted_merged_text.items(), key=lambda x: (x[0][1], x[0][0]))
adjusted_merged_text_sorted_group = {}
for coordinates, text in adjusted_merged_text_sorted:
    if coordinates[1] not in adjusted_merged_text_sorted_group:
        adjusted_merged_text_sorted_group[coordinates[1]] = [text]
    else:
        adjusted_merged_text_sorted_group[coordinates[1]].append(text)
for text_list in adjusted_merged_text_sorted_group.values():
    print(' | '.join(text_list))




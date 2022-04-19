import os
import cv2
import numpy as np
import pandas as pd
import json
import pickle as pkl


def main():
    global points
    path = 'D:/yanhui/yanhui/19/'
    path_list = os.listdir(path)
    for file_name in path_list:
        if os.path.splitext(file_name)[1] == '.json':
            json_name = file_name
            json_path = path + json_name
            # print('json_path: {}'.format(json_path))

            # 加载解析json文件
            with open(json_path, "r+") as f:
                json_data = json.load(f)
            # print('json_data: {}'.format(json_data))
            shapes = json_data['shapes']
            img_name, suffix = json_data['imagePath'].split('.')
            img_path = path + img_name + '.jpg'
            pkl_path = path + img_name + '.pkl'
            print('open {} 文件'.format(json_path))
            print("shapes: {}".format(shapes))
            for element in shapes:
                points = element['points']
            print("shape: {}".format(points))

            # 加载原图
            img = cv2.imread(img_path)

            # make four points
            point1 = points[0]
            point2 = [points[0][0], points[1][1]]
            point3 = points[1]
            point4 = [points[1][0], points[0][1]]
            points = np.array([point1, point2, point3, point4], dtype=np.int32)
            print('width: {} length: {}'.format(points[1][1] - points[0][1], points[2][0] - points[0][0]))
            mask_zeros = np.zeros(img.shape, np.uint8)
            mask = cv2.fillPoly(mask_zeros, [points], color=(255, 255, 255))
            img_rectangel = img.copy()
            img_rectangel = cv2.polylines(img_rectangel, [points], isClosed=True, color=(0, 0, 255))
            cv2.imshow('img_rectangle', img_rectangel)
            cv2.imshow('mask', mask)
            # 计算mask
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            img_mask = cv2.copyTo(img_gray, mask)
            cv2.imshow('img_mask', img_mask)
            cv2.waitKey()

            csv_savePath = 'data/' + img_name + '.csv'
            # 加载pkl文件并掩膜处理
            with open(pkl_path, "rb") as f:
                pkl_data = pkl.load(f)
            pkl_data = np.array(pkl_data, np.float32)

            # 掩膜处理
            pkl_mask = cv2.copyTo(pkl_data, mask)
            pkl_line_data = []
            for i in range(points[1][1] - points[0][1]):
                for j in range(points[2][0] - points[0][0]):
                    pkl_line_data.append([i, j, pkl_data[i + points[0][1]][j + points[0][0]]])
            pkl_DataFrame = pd.DataFrame(pkl_line_data)
            pkl_DataFrame.to_csv(csv_savePath, index=False)
            print('pkl to csv done! save to {}\n'.format(csv_savePath))


if __name__ == "__main__":
    main()

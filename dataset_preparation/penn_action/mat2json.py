import json
import scipy.io
import os
import tqdm
if __name__ == "__main__":
    CHECK_EMPTY = True
    dataset_path = ''
    label_path = os.path.join(dataset_path, 'labels')
    bbox_path = os.path.join(dataset_path, 'bbox')
    mocap_path = os.path.join(dataset_path, 'mocap')

    for action in tqdm.tqdm(range(1, 2327)):  # 2135)):#2327
        #action = 27
        mat = scipy.io.loadmat(os.path.join(
            label_path, '{0:04d}.mat'.format(action)))
        # print("mat",mat)
        dict_frank = {}
        action_path = os.path.join(dataset_path, 'bbox/{0:04d}'.format(action))

        if CHECK_EMPTY:
            bbox_len = len(os.listdir(os.path.join(
                mocap_path, "{0:04d}".format(action), "bbox")))
            # print("matframes",mat['nframes'][0][0])
            # print("bbox_len",bbox_len)
            if mat['nframes'][0][0] != bbox_len:
                print("action", action)
        else:
            if not os.path.exists(action_path):
                os.mkdir(action_path)
            for frame, bbox in enumerate(mat['bbox']):
                x = float(bbox[0])
                y = float(bbox[1])
                w = float(bbox[2]-bbox[0])
                h = float(bbox[3]-bbox[1])
                dict_frank = {"image_path": "{0}/frames/{1:04d}/{2:06d}.jpg".format(
                    dataset_path, action, frame+1), "body_bbox_list": [[x, y, w, h]]}
                # print("dict_frank",dict_frank)
                with open(os.path.join(dataset_path, 'bbox', '{0:04d}/{1:06d}.json'.format(action, frame+1)), 'w') as outfile:
                    json.dump(dict_frank, outfile)
            #{"image_path": "xxx.jpg", "hand_bbox_list":[{"left_hand":[x,y,w,h], "right_hand":[x,y,w,h]}], "body_bbox_list":[[x,y,w,h]]}

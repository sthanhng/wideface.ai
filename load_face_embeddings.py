import pickle
import numpy as np

# load the serialized face encodings + bounding box locations from
# disk, then extract the set of encodings to so we can cluster on
# them
print("[INFO] loading encodings...")
data = pickle.loads(open("extracted_dict.pickle", "rb").read())
# data = np.array(data)
# print(data.ndim)
# print(data['outputs/0001.png'])
# print("EBF1", data[0][0])
# print("EBF2", data[0][1])
# encodings = [d["encoding"] for d in data]
print(data, type(data))
# print(data["dataset/AlignedFaces/2018-10-26_16-56-13_aligned_000000.jpg"])
# print(data["dataset/AlignedFaces/2018-10-26_17-19-39_aligned_000000.jpg"])
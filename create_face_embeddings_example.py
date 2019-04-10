from face_utils import create_face_embeddings


model_path = "models/20170512-110547"
batch_size = 3
image_size = 160
data_dir = "dataset/AlignedFaces/"

if __name__ == "__main__":
    create_face_embeddings(model_path, data_dir, image_size, batch_size)

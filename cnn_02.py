from tensorflow.keras.applications import VGG16
  
# # load the VGG16 network *pre-trained* on the ImageNet dataset
# model = VGG16(weights="imagenet")
# model.summary()

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# def show_image(image_path):
#     # Membaca gambar dari file
#     image = mpimg.imread(image_path)
    
#     # Menampilkan bentuk (shape) dari gambar
#     print(image.shape)
    
#     # Menampilkan gambar menggunakan plt.imshow
#     plt.imshow(image)
    
#     # Menampilkan plot
#     plt.axis('off')  # Mematikan axis untuk tampilan yang lebih bersih
#     plt.show()

# # Menampilkan gambar dengan menggunakan fungsi show_image
# show_image("data/doggy_door_images/happy_dog.jpg")

# PREPROCESSING the IMAGE
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.vgg16 import preprocess_input

def load_and_process_image(image_path):
    # Print image's original shape, for reference
    print('Original image shape: ', mpimg.imread(image_path).shape)
    
    # Load in the image with a target size of 224, 224
    image = image_utils.load_img(image_path, target_size=(224, 224))
    # Convert the image from a PIL format to a numpy array
    image = image_utils.img_to_array(image)
    # Add a dimension for number of images, in our case 1
    image = image.reshape(1,224,224,3)
    # Preprocess image to align with original ImageNet dataset
    image = preprocess_input(image)
    # Print image's shape after processing
    print('Processed image shape: ', image.shape)
    return image

processed_image = load_and_process_image("data/anjing.jpg")

# make a predicition
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Fungsi untuk menampilkan gambar
def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)
    plt.axis('off')  # Mematikan axis untuk tampilan yang lebih bersih
    plt.show()

# Fungsi untuk memuat dan memproses gambar
def load_and_process_image(image_path, target_size=(224, 224)):
    # Memuat gambar dengan ukuran target
    image = load_img(image_path, target_size=target_size)
    # Mengonversi gambar menjadi array numpy
    image = img_to_array(image)
    # Menambahkan dimensi batch
    image = np.expand_dims(image, axis=0)
    # Melakukan preprocessing sesuai dengan model VGG16
    image = preprocess_input(image)
    return image

# Memuat model VGG16 yang sudah dilatih dengan bobot ImageNet
model = VGG16(weights='imagenet')

# Fungsi untuk menampilkan prediksi dalam bentuk yang dapat dibaca
def readable_prediction(image_path):
    # Menampilkan gambar
    show_image(image_path)
    # Memuat dan memproses gambar
    image = load_and_process_image(image_path)
    # Melakukan prediksi
    predictions = model.predict(image)
    # Menampilkan prediksi dalam bentuk yang dapat dibaca
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    print('Predicted:')
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score:.2f})")

# Memanggil fungsi readable_prediction
readable_prediction("data/anjing.jpg")
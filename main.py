import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
import matplotlib.pyplot as plt


def make_network():
    image_size = (111, 108)
    class_names = ['1-Volvo', '10-BMW', '11-Jeep', '12-Kia', '13-Citroen', '14-Land Rover', '15-Lexus', '16-Mazda', '17-Mercedes', '18-Mini', '19-Mitsubishi', '2-Volkswagen', '20-Nissan', '21-Opel', '22-Peugeot', '23-Renault', '24-Seat', '25-GMC', '26-Smart', '27-Subaru', '28-Suzuki', '29-Tesla', '3-Hyundai', '30-Toyota', '31-Alfa Romeo', '32-Acura', '4-Lancia', '5-Dacia', '6-Daewoo', '7-Ford', '8-Skoda', '9-Honda']

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory="D:\\UTM\\IA\\laboratoare\\lab3\\Train",
        labels="inferred",
        class_names= class_names,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory="D:\\UTM\\IA\\laboratoare\\lab3\\Train",
        labels="inferred",
        label_mode="categorical",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
    )

    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[int(labels[i])])
            plt.axis("off")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    make_network()


import matplotlib.pyplot as plt


def draw(dataset, small_mlp_accuracy, big_mlp_accuracy, svm_accuracy):
    test_size = [0.2, 0.4, 0.5, 0.6, 0.8]

    plt.figure(figsize=(10, 6))
    plt.plot(test_size, small_mlp_accuracy, marker='o',
             linestyle='-', label='Small MLP')
    plt.plot(test_size, big_mlp_accuracy, marker='o',
             linestyle='-', label='Big MLP')
    plt.plot(test_size, svm_accuracy, marker='o', linestyle='-', label='SVM')
    plt.xlabel('Test size ')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy of MLP and SVM on {dataset}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../results/{dataset}')


if __name__ == '__main__':
    # small_mlp_accuracy = [1.0000, 0.9833, 0.6933, 0.6778, 0.9500]
    # big_mlp_accuracy = [0.9667,0.9667,0.9600,0.9778,0.9833]
    # svm_accuracy = [1.0000,1.0000,0.9867,0.9889,0.9833]

    # small_mlp_accuracy = [0.5581, 0.6353, 0.6226, 0.4803, 0.4379]
    # big_mlp_accuracy = [0.6279,0.6471,0.6604,0.6535,0.5976]
    # svm_accuracy = [0.6235,0.6235,0.6038,0.6063,0.5680]

    small_mlp_accuracy = [0.5113, 0.4960, 0.5112, 0.4528, 0.4006]
    big_mlp_accuracy = [0.9286, 0.9277, 0.9259, 0.9229, 0.9166]
    svm_accuracy = [0.9172, 0.9134, 0.9127, 0.9079, 0.8937]
    draw('AwA2', small_mlp_accuracy, big_mlp_accuracy, svm_accuracy)

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src import result

pca = PCA(n_components=2)
voc_file, word_embedding = result.load("src/data/vocab", "src/data/nnlm_word_embeddings.npy")
we_2D = pca.fit_transform(word_embedding)
n = 1000
# words_index = [voc_file.index("he"), voc_file.index("she"), voc_file.index("I"), voc_file.index("we"),
#                voc_file.index("had"), voc_file.index("has"), voc_file.index("was"), voc_file.index("is"),
#                voc_file.index("student"), voc_file.index("school"), voc_file.index("book"), voc_file.index("teacher")]

words_index = [voc_file.index("country"), voc_file.index("capital"),
               voc_file.index("US"), voc_file.index("Washington"),
               voc_file.index("Britain"), voc_file.index("London"),
               voc_file.index("France"), voc_file.index("Paris")]
x_coords = we_2D[words_index, 0]
y_coords = we_2D[words_index, 1]

for i in range(0, len(x_coords), 2):
    a = np.array([x_coords[i], y_coords[i]])
    b = np.array([x_coords[i + 1], y_coords[i + 1]])
    print(np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2))

plt.scatter(x_coords, y_coords, s=4, c='b')
for label, x, y in zip(np.array(voc_file)[words_index], x_coords, y_coords):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=8)
# plt.show()
plt.savefig("embeddings.png", dpi=200)

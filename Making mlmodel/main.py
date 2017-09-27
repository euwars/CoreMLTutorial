# Required imports
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import coremltools


# Reading in and parsing data
raw_data = open('Archive.txt', 'r')
sms_data = []
for line in raw_data:
    split_line = line.split("\t")
    sms_data.append(split_line)

sms_data = np.array(sms_data)

# Separating content and labesl
X = sms_data[:, 1]
y = sms_data[:, 0]

# Vectorize the data to tf-idf using sklearn extraction
vectorizer = TfidfVectorizer()
vectorized = vectorizer.fit_transform(X)

# Saving words into text file
words = open('Words.txt', 'w')
for feature in vectorizer.get_feature_names():
    words.write(feature.encode('utf-8') + '\n')
words.close()

# New model and training
model = LinearSVC()
model.fit(vectorized, y)

# Converting the sklearn model to coreml
coreml_model = coremltools.converters.sklearn.convert(model, "message", 'label')
coreml_model.save('MessageClassifier.mlmodel')
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from models import IrisModel


#pick a manual seed for randomization
torch.manual_seed(41)

model = IrisModel()

url ='https://gist.github.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
my_df = pd.read_csv(url)
print(my_df.describe())
#Changed string to int
replace_map = {'Setosa': 0.0, 'Versicolor': 1.0, 'Virginica': 2.0}


my_df['variety'] = my_df['variety'].map(replace_map).astype(float)
#print(my_df.tail())

# Train Test Split,
X = my_df.drop('variety', axis=1)
y = my_df['variety']

print(X.shape, y.shape)

#Convert to numpy
X = X.values
y = y.values
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Convert X features to float tensors

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert Y labels to tensors long

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set the criterion of model to measure the error

criterion = nn.CrossEntropyLoss()

# Choose Adam optimizer, lr = learning rate ( if error doesn't go down after a bunch of iterations ( epochs ), will lower learning rate

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#print(model.parameters)

# Train our model
# Epochs?
epochs = 1000

losses = []

for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())

    if i % 10 == 0:
        #print(f'Epoch {i}, loss: {loss.detach().numpy()}')
        pass


    # Do some back propagation: take the error rate of forward propagation and feed it back
    # through the network to fine tune the weights

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

'''
plt.plot(range(epochs), losses)
plt.ylabel('loss/error')
plt.xlabel('epoch')
plt.show()
'''

# Evaluate the Model on Test Data set

with torch.no_grad(): # Turn off back propagation
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)

print(f'Loss is {loss.detach().numpy()}')


correct =0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'Accuracy is {100 * correct / len(X_test)}')
print(f'You have got {correct} correct!')

#replace_map = {'Setosa': 0.0, 'Versicolor': 1.0, 'Virginica': 2.0}
value_index = {0:'Setosa',1:'Versicolor',2:'Virginica'}

#5.8	2.6	4	1.2 Versicolor
# 7.4	2.8	6.1	1.9	Virginica
# 5.1	3.5	1.4	.3	Setosa
def predict(values):
    new_iris = torch.tensor(values)
    with torch.no_grad():
        print( value_index[model.forward(new_iris).argmax().item()])


import time
start = time.perf_counter()

predict([7.4,2.8,6.1,1.9]) #Virginica
end = time.perf_counter()
print(f"Execution time: {end - start:.4f} seconds")
#predict([5.1,3.5,1.4,.3]) #Setosa


#torch.save(model.state_dict(), "models/iris_model.pt")

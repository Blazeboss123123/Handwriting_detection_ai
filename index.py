from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader            #data loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")      #check if u have a gpu that you can use
print(DEVICE)

training_data = datasets.MNIST(
    root="data",
    train=True,                                      #all training and testing data converted to tensor
    download=True,                                   #training batch
    transform=ToTensor()
)

testing_data = datasets.MNIST(
    root="data",     
    train=False,                                     #testing batch
    download=True,
    transform=ToTensor()
)

loaders = {                                #loading data into the model
    "train": DataLoader(
        training_data,
        batch_size=50,
        shuffle=True,
        num_workers=0 ),
    "test": DataLoader(
    testing_data,
    batch_size=50,
    shuffle=True,
    num_workers=0 ),
}

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)                #input output channel
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()                     #dropout layer
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))  # Apply relu after the first fully connected layer
        x = F.dropout(x, training=self.training)  # Apply dropout after relu
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)         #applies Log-Softmax Function to x     (complex maths not really sure about right now)



model = CNN().to(DEVICE)                             
optimizer = optim.Adam(model.parameters(),lr=0.001)                

loss_fn = nn.CrossEntropyLoss()                                 #does the cross entropy

def train(EPOCH):
    model.train()                           #puts model into training mode
    for batch_idx, (data,target) in enumerate(loaders["train"]):          #extracts batch data and index
        data, target = data.to(DEVICE), target.to(DEVICE)                     #keeps data on cpu or gpu
        optimizer.zero_grad()
        output = model(data)                         #get the raw output from the mode
        target = target.squeeze()
        loss= loss_fn(output, target)                  #calculate loss using the difference between output and target
        loss.backward()                               #back propagration 
        optimizer.step()       #upate based on current gradient
        if batch_idx % 20 == 0:
            print(f"Train epoch {EPOCH} [{batch_idx * len(data)}/{len(loaders['train'].dataset)} "                   #formatting of data
      f"({100. * batch_idx / len(loaders['train']):.0f}%)]\t{loss.item():.6f}")
    
    
def test():
    model.eval()                        #model put in evaluation mode
    test_loss=0                   #amount of predicitions corrects aswell as loss
    correct=0

    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(DEVICE), target.squeeze().to(DEVICE)                       #squeeze tensor to make it right size    
            output = model(data)                             #output from the model
            target = target.squeeze()                        #squeezing the target values
            test_loss += loss_fn(output,target).item()                              
            prediction= output.argmax(dim=1,keepdim= True)                                 #checks predictions
            correct += prediction.eq(target.view_as(prediction)).sum().item()                #adds to correct when a prediciton is right

    test_loss/= len(loaders["test"].dataset)
    print(f"\n Test set : Average Loss: {test_loss:.4f}, accuracy: {correct}/{len(loaders['test'].dataset)} ({100. * correct / len(loaders['test'].dataset):.0f}%\n)")

try:
    if __name__ == "__main__":                                       #training loop in action!
        print(model)
        for EPOCH in range(1,5):                                   #Epoch is the cycles taken less strain on your cpu or gpu reduce the epochs
            train(EPOCH)
            test()

except Exception as e:
    print("-"*70)
    print(e)

model.eval()
data, target = testing_data[1]
data = data.unsqueeze(0).to(DEVICE)
output = model(data)
prediction = output.argmax(dim=1, keepdim=True).item()
print(f"Prediction {prediction}")
image=data.squeeze(0).squeeze(0).cpu().numpy()                     #squeezes the tensor twice then saves to cpu then converts into a numpy.
plt.imshow(image, cmap="gray")                                     #Lots found online from Nerual Nine YT
plt.show()

torch.save(model, "Handwriting_model.pt")        #saved model



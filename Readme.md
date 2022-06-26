**Assumption**: Deepfake videos created by GANs (generally) tend to possess distortions when compared to adjacent frames
<br/>

[Dataset From DFDC](https://ai.facebook.com/datasets/dfdc/)

**Real images**

![image](https://user-images.githubusercontent.com/58447982/175811325-ccce2c7f-af76-49ff-ab0c-8a83424c5d58.png)


**Fake images**: Notice there are abrupt changes/distortions between adjacent frames. 

![image](https://user-images.githubusercontent.com/58447982/175812221-08224f61-bf8e-4ea4-80b3-b679ff6c84d3.png)

<br/>

#### Method: Detect distortions within a consecutive frames
**Data Processing**
- mtcnn (Face Recognition), opencv (Convert video into frames), Exception Handling (Face detection failure, Multiple speakers ...etc)
<br/>

```c
vidcap = cv2.VideoCapture(invideofilename)
videoname = os.path.basename(invideofilename).split('.')[0]
count = 0
save_count=0

while True:
    success,image = vidcap.read()
    if not success:
        break
    mtcnn=MTCNN(image_size=img_size,margin=20,keep_all=True)
    faces= mtcnn(image)
```

There are frames with exceptions (eg. no face detection)
These frames create gaps between frames and since our assumption lies on the fact that the interval betweeen every frame is constant, 
these frames needed to be marked. Our dataloader searches frames linearly and when it detects a marked frame, it will return the set of frames up to that point
Two Pointer Algorithm was used.

<br/>

```c
class DeepfakeFaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.csv_fileline = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_count=0
        start_idx = self.csv_fileline.iloc[idx, 2]
        end_idx = self.csv_fileline.iloc[idx,3]
        image_stack=[]
        jump_list=[]
        for img_count in range(start_idx,end_idx+1):
            img_name = os.path.join(self.root_dir,
                                self.csv_fileline.iloc[idx, 0].split('.')[0]+'_{0:05d}.jpg'.format(img_count))
            image = io.imread(img_name) # exception 발생
            image_stack.append(image)
            image_stack_np = np.asarray(image_stack)

            label = self.csv_fileline.iloc[idx, 1]
            label = np.array([label])

            sample = {'image': image_stack_np, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
```

<br/>

![image](https://user-images.githubusercontent.com/58447982/175812389-76898be4-6cf8-4856-8856-ef12b9c5cc7c.png)

<br/>




**Embedding Model**
- Extract 128-d vector code from face images
Triplet loss is used to guide real images to produce similar vector codes and distinct codes for fake images
This process helps LSTM to better classify since vectors of two classes from clusters of their own 


```c
def Triplet_embedding_loss(input, target,label):
    margin = 1.0
    loss = torch.nn.L1Loss(reduction='mean')
    if label == 1:
        return loss(input,target)
    else:
        return max(torch.zeros(1, requires_grad=True),margin-loss(input,target))

Embedding_Model = InceptionResnetV1().double()
optimizer = optim.Adam(Embedding_Model.parameters(), lr=0.005)
```

**LSTM Classifier**
- Feed sequence of 128-d vectors into LSTM (Cross Entropy Loss)

```c

class classifier(nn.Module):
    
    #define all the layers used in model
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout):

        super().__init__()          
        print(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.act = nn.Sigmoid()
        
    def forward(self, embedding_vector):
        
        packed_output, (hidden, cell) = self.lstm(embedding_vector)
        
        print(packed_output[:, -1, :])
        dense_outputs=self.fc(packed_output[:, -1, :])

        outputs=self.act(dense_outputs)
        
        return outputs

model = classifier(embedding_dim, num_hidden_nodes, num_output_nodes, num_layers, bidirectional=False, dropout=dropout)

```

![image](https://user-images.githubusercontent.com/58447982/175812504-4d96ca78-f8e4-4403-baa2-0561edde1b24.png)


**Conclusion**

The model's accuracy is 0.81 with log loss 0.514.



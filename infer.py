import os
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
import torch


DATASET_FOLDER = './datasets'

# Load devices and models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load face features model
resnet = InceptionResnetV1(pretrained="vggface2", device=device).eval()
# Load face detector model
mtcnn = MTCNN(image_size=160, margin=40, keep_all=False, device=device)
print(resnet.device)
print(mtcnn.device)

def collate_fn(x):
  return x[0]

def init():
  
  print('Running on device: {}'.format(device))
  
  workers = 0 if os.name == 'nt' else 4
  
  dataset = datasets.ImageFolder(DATASET_FOLDER)
  dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
  loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
  
  # Loading before start
  print("Initializing data from datasets: ")
  aligned = []
  names = []
  for x, y in loader:
      x_aligned, prob = mtcnn(x, return_prob=True)
      if x_aligned is not None:
          print("Found labelled face name: {}".format(x))
          print('Face detected with probability: {:8f}'.format(prob))
          aligned.append(x_aligned)
          names.append(dataset.idx_to_class[y])
  
  aligned = torch.stack(aligned).to(device)
  embeddings = resnet(aligned).detach()
  
  return aligned, names, embeddings



def capture(aligned, names, embeddings):

  # Start open video capture
  cap = cv2.VideoCapture(0)
  # Frame to skip after the camera is opened
  ramp_frames = 30 
  # Skip frame for high quality video
  for i in range(ramp_frames):
    cap.read()
    
  while True:    
    _, frame = cap.read()

    scaled = cv2.resize(frame, (0, 0), fx=.25, fy=.25)
    current_face = mtcnn(scaled)
    if current_face is not None:
      current_face_emb = resnet(current_face.unsqueeze(0).cuda())
      for (idx, embedding) in enumerate(embeddings):
        distance = (current_face_emb - embedding).norm().item()
        
        print("Current distance: ",distance)
        # Linear search for a face
        if distance <= 0.76:
          print("Found {} face with look up through {}".format(names[idx], idx))
          
          cv2.putText(frame,  
                names[idx] + " ({})".format(distance),  
                (50, 50),  
                cv2.FONT_HERSHEY_SIMPLEX, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4
          ) 
          continue
        
        print("Invalid face ")  
        cv2.putText(frame,  
                "Invalid face "+ " ({:2f})".format(distance),  
                (50, 150),  
                cv2.FONT_HERSHEY_SIMPLEX, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 

    
    cv2.imshow("Inference", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
      break  

  # After the loop release the cap object 
  cap.release() 
  # Destroy all the windows 
  cv2.destroyAllWindows() 

  del (cap)
  

if __name__ == "__main__":
  
  # Initial the datasets
  aligned, names, embeddings = init()
  print(aligned, names, embeddings)
  
  capture(aligned, names, embeddings)
  
  

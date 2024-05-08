import os
import cv2
from facenet_pytorch import MTCNN
from torchvision.utils import save_image

DATASET_FOLDER = './datasets'

def init():
  name = input("Enter username: ")
  # Make a directory if failed
  if os.path.exists(DATASET_FOLDER) == False:
    os.mkdir(DATASET_FOLDER)

  # Make a label directory
  label_dir = os.path.join(DATASET_FOLDER, name)
  print("The data will write into {}".format(label_dir))
  if os.path.exists(label_dir) == False: 
    os.mkdir(label_dir)
    
    
    
  return label_dir



def capture(labelled_dir):
  
  # Load face detector models
  mtcnn = MTCNN(image_size=160, margin=40, keep_all=False)
  
  # Start open video capture
  cap = cv2.VideoCapture(0)
  current_size = 0
  # Frame to skip after the camera is opened
  ramp_frames = 30 
  # Skip frame for high quality video
  for i in range(ramp_frames):
    cap.read()
    
    
  while True:
    
    _, frame = cap.read()
    
    scaled = cv2.resize(frame, (0, 0), fx=.25, fy=.25)
    cv2.imshow('Preview ', frame)
    
    face_img, prob = mtcnn(scaled, return_prob=True)
    
    print(prob)
    
    
    if face_img is not None and prob >= 0.99:
      print("Capturing {}".format(current_size))
      # save_image(frame, os.path.join(labelled_dir, "{}.png".format(current_size)))
      cv2.imwrite(os.path.join(labelled_dir, "{}.png".format(current_size)), frame)
      current_size = current_size + 1
    
    if current_size >= 10:
      break
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
      break  

  # After the loop release the cap object 
  cap.release() 
  # Destroy all the windows 
  cv2.destroyAllWindows() 

  del (cap)
  

if __name__ == "__main__":
  
  # Initial the dir first
  labelled_dir = init()
  # Capture the face and store
  capture(labelled_dir)
  
  print("Successfully capture your face.")
  

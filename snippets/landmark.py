

import sys
sys.path.append("/export/home/oblum/projects/face-alignment/")
import face_alignment
from skimage import io



fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
                                      enable_cuda=True, 
                                      flip_input=False)

fname = 'data/faceScrub/imgs_top_aligned/Aaron_Eckhart/Aaron_Eckhart_1.jpg'
input = io.imread(fname)
preds = fa.get_landmarks(input)

print preds
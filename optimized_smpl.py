import numpy as np
import glob
import os


def getBetasAverage(file_path,frames):
    model_name = os.path.join(file_path,"{:06d}.npz".format(frames[0]))
    model_dict = np.load(model_name)
    betas = model_dict['betas']
    for i in frames[1:]:
        model_name = os.path.join(file_path,"{:06d}.npz".format(i))
        model_dict = np.load(model_name)
        betas = betas + model_dict['betas']
    betas = betas / len(frames)
    return betas

def optimize_betas(file_path,new_file_path,frames,betas):
    os.makedirs(new_file_path,exist_ok=True)
    model_name = os.path.join(file_path,"{:06d}.npz".format(frames[0]))
    model_dict = np.load(model_name)
    np.savez(os.path.join(new_file_path,"{:06d}.npz".format(frames[0])),trans=model_dict['trans'],
                                  root_orient= model_dict['root_orient'],pose_hand = model_dict['pose_hand'], pose_body = model_dict['pose_body'],
                     betas = betas)
    for i in frames[1:]:
        model_name = os.path.join(file_path,"{:06d}.npz".format(i))
        model_dict = np.load(model_name)
        np.savez(os.path.join(new_file_path,"{:06d}.npz".format(i)),trans=model_dict['trans'],
                                  root_orient= model_dict['root_orient'],pose_hand = model_dict['pose_hand'], pose_body = model_dict['pose_body'],
                     betas = betas)

file_path = "output/foreground/first/camerahmr"
new_file_path = "output/foreground/first/camerahmr_optimized"
# female-3-casual
#train_frames = range(0,446,4)
#val_frames = range(446,447,4)
#test_frames = range(446,648,4)
# female-4-casual
train_frames = range(0,336,4)
val_frames = range(335,336,4)
test_frames = range(335,524,4)
# male-3-casual 
#train_frames = range(0,456,4)
#val_frames = range(456,457,4)
#test_frames = range(456,676,4)
# male-4-casual
#train_frames = range(0,660,6)
#val_frames = range(660,661,6)
#test_frames = range(660,873,6)

betas = getBetasAverage(file_path,train_frames)
optimize_betas(file_path,new_file_path,train_frames,betas)
print(betas)
betas = getBetasAverage(file_path,val_frames)
optimize_betas(file_path,new_file_path,val_frames,betas)
print(betas)
betas = getBetasAverage(file_path,test_frames)
optimize_betas(file_path,new_file_path,test_frames,betas)
print(betas)


#model_files = glob.glob("npz/*.npz")
#model_dict = np.load(model_files[0])
#betas = model_dict['betas']

#for model_file in model_files[1:]: 
#	model_dict = np.load(model_file)
#	betas = betas + model_dict['betas']

#betas = betas/len(model_files)

#for model_file in model_files:
#	model_dict = np.load(model_file)
#	np.savez("2"+model_file,trans=model_dict['trans'], bone_transforms = model_dict['bone_transforms'],
#                                  root_orient= model_dict['root_orient'],pose_hand = model_dict['pose_hand'], pose_body = model_dict['pose_body'],minimal_shape=model_dict['minimal_shape'],
#                     betas = betas)

#model_dict = np.load(os.path.join(new_file_path,"000000.npz"))
#print(model_dict['betas'])

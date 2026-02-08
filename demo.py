import argparse
from mesh_estimator import HumanMeshEstimator


def make_parser():
    parser = argparse.ArgumentParser(description='CameraHMR Regressor')
    parser.add_argument("--image_folder", "--image_folder", type=str, 
        help="Path to input image folder.")
    parser.add_argument("--camera_type", "--camera_type", type=str,
        help="The Type of the Camera.")
    parser.add_argument("--camera_name", "--camera_name", type=str, default="",
        help="Name of the output camera")
    parser.add_argument("--camera_path", "--camera_path", type=str, default="",
        help="Path to the camera from which the intrinsics and distortion coefficients are used, if the camera_type = calibrated.")
    parser.add_argument("--output_folder", "--output_folder", type=str,
        help="Path to folder output folder.")
    parser.add_argument("--model_type", type=str, default='smpl', choices=['smpl', 'smplx'],
        help="Type of model to use.")
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    estimator = HumanMeshEstimator(model_type=args.model_type)
    estimator.run_on_images(args.image_folder, args.output_folder, args.camera_type, args.camera_path,args.camera_name)
    
if __name__=='__main__':
    main()

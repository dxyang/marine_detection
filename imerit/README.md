## Instructions for using AWS CLI

- Install AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

## Downloading data from AWS:
/usr/local/bin/aws configure sso
SSO start URL: <example from invite email: https://d-90679b6642.awsapps.com/start/>
SSO region: us-east-1

This will generate a profile name, save this for subsequent commands.

Then to list data:
/usr/local/bin/aws s3 ls s3://whoi-rsi-fish-detection/datasets/ --profile <profile name>

Or to download data:
/usr/local/bin/aws s3 sync s3://whoi-rsi-fish-detection/datasets/ <local destination folder> --profile <profile name>

## iMerit -> COCO (for fish detection labels)
1. Download iMerit labels from AWS
2. Use imerit2coco.ipynb

## COCO -> Labelbox (for species-level labelling)
1. Use coco2labelbox.ipynb

## Labelbox -> COCO (for local use)
1. Use labelbox2coco.ipynb

## COCO Format:
info: Dict

licenses: List[Dict]

images: List[Dict]
    "id": int,                      // unique for every image
    "width": int, 
    "height": int, 
    "file_name": str,               // dataset relative path to image
    "license": int,
    "date_captured": str,

annotations: List[Dict]
    "id": int,                      // unique for every annotation      
    "image_id": int,           
    "category_id": int,             // 1 = fish, not sure how to consider hierarchies yet but COCO has supercategories for depth 1 trees
    "bbox": List[int],              // [x, y, w, h]  
    "area": int,                    // w * h
    "iscrowd": int

categories: List[Dict]

video_sequences: List[Dict]
    "id": int,                      // unique for every video sequence
    "image_id_list": List[int],     // list of ids of images
    "file_name": str,               // dataset relative path to image directory corresponding to video

object_tracks: List[Dict]
    "id": int,                      // unique for every video sequence
    "bbox_id_list": List[int],      // list of ids of bboxes
    "image_id_list": List[int],     // list of ids of images, maybe redundant (bboxes are each associated with an image)
    "video_seq_id": int,            // maybe redundant (video_sequences has list of image ids)
    "category_id": int,             // maybe redundant (each bbox already has a category)
# marine_detection


## bags of interest
[2022-10-USVI Data Logs](https://docs.google.com/spreadsheets/d/1h3gqMo6qbuo2wdldPzKdIfek9tsK62DqrL3CMTWyFxk/edit#gid=616306361)
* /vortexfs1/share/warplab/shared/datasets/2022-10-USVI/2022-11-03/CUREE3/bags/warpauv_3_xavier4_2022-11-03-10-12-06.bag
* /vortexfs1/share/warplab/shared/datasets/2022-10-USVI/2022-11-03/CUREE3/bags/warpauv_3_xavier4_2022-11-03-10-59-34.bag

whoi_rs_701515_vmad2 upload this model to tator



## useful commands

### convert bag to mp4
```bash
python bagutils/bag2mp4.py \
    /data_nvme/dxy/biomap/warpauv_3_xavier4_2022-11-03-10-12-06/warpauv_3_xavier4_2022-11-03-10-12-06.bag \
    15 \
    /data_nvme/dxy/biomap/warpauv_3_xavier4_2022-11-03-10-12-06 \
    /warpauv_3/cameras/forward/rgb/image_stream/h264
```

### ffmpeg (mp4 <=> image directory)
```bash
ffmpeg -i forward.mp4 -start_number 0 'forward/vanilla/frame_%06d.png'
```

```bash
ffmpeg -framerate 15 -pattern_type glob -i '*.png' \
  -c:v libx264 -pix_fmt yuv420p whoi_rs_701515_vmad2.mp4
```

### running yolo
```bash
python detect.py \
    --weights ~/localdata/biomap/yolov5_models/whoi_rs_701515_vmad2/weights/best.pt \
    --source ~/localdata/biomap/warpauv_3_xavier4_2022-11-03-10-59-34/forward/vanilla \
    --conf-thres=0.01 \
    --save-txt --save-conf --save-crop \
    --project ~/localdata/biomap/warpauv_3_xavier4_2022-11-03-10-59-34/forward/ \
    --name whoi_rs_701515_vmad2

python detect.py \
    --weights ~/localdata/biomap/yolov5_models/whoi_rs_701515_vmad2/weights/best.pt \
    --source ~/localdata/biomap/warpauv_3_xavier4_2022-11-03-10-12-06/forward/vanilla \
    --conf-thres=0.01 \
    --save-txt --save-conf --save-crop \
    --projec ~/localdata/biomap/warpauv_3_xavier4_2022-11-03-10-12-06/forward/ \
    --name whoi_rs_701515_vmad2
```

### align poses with image frames
```bash
python -m scripts.extract_image_pose --bagfile ~/localdata/warp/biomap/warpauv_2_xavier3_2022-11-02-09-55-23-metashape-odom.bag
```

### integrate fish detection counts with pose data
```bash
python -m scripts.integrate_detections \
  --data_pkl ~/localdata/warp/biomap/forward_data.pkl \
  --detection_dir ~/localdata/warp/biomap/forward/whoi_rs701515_mouss_all_neumann_200/labels \
  --confidence_threshold 0.01
```

### generate map
```bash
python -m scripts.build_map \
  --data_pkl ~/localdata/warp/biomap/forward_data.pkl \
  --model_name whoi_rs_only_701515
```

### tator video upload
```bash
sbatch --array=1-$(wc -l < video_list.txt)%1 upload_video_list.sbatch listfiles/video_list.txt
```

```bash
# cd tator_yolo_scripts

python run_yolo_model_on_video_and_upload_rois.py \
  ~/localdata/biomap/warpauv_3_xavier4_2022-11-03-10-12-06/forward/vanilla \
  ~/localdata/biomap/yolov5_models/whoi_rs_701515_vmad2/weights/best.pt \
  --cache_offline \
  --local_project ~/localdata/biomap/warpauv_3_xavier4_2022-11-03-10-12-06/forward/whoi_rs_701515_vmad2 \
  --media_id 437 \
  --token $(cat tator_token.txt) \
  --media_type 2 \
  --classlist class_list.txt \
  --conf-thres 0.01 \
  --device 0

python run_yolo_model_on_video_and_upload_rois.py \
  ~/localdata/biomap/warpauv_3_xavier4_2022-11-03-10-59-34/forward/vanilla \
  ~/localdata/biomap/yolov5_models/whoi_rs_701515_vmad2/weights/best.pt \
  --cache_offline \
  --local_project ~/localdata/biomap/warpauv_3_xavier4_2022-11-03-10-59-34/forward/whoi_rs_701515_vmad2 \
  --media_id 438 \
  --token $(cat tator_token.txt) \
  --media_type 2 \
  --classlist class_list.txt \
  --conf-thres 0.01 \
  --device 0

```



### nice strings
```
whoi_rs701515_mouss_all_neumann_200
whoi_rs_only_701515
```
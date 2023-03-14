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
ffmpeg -i forward.mp4 'forward/vanilla/%06d.png'
```

```bash
ffmpeg -framerate 15 -pattern_type glob -i '*.png' \
  -c:v libx264 -pix_fmt yuv420p out.mp4
```

### running yolo
```bash
python detect.py \
    --weights ~/localdata/warp/yolov5_models/whoi_rs701515_mouss_all_neumann_200/weights/best.pt \
    --source ~/localdata/warp/biomap/forward/vanilla/ \
    --conf-thres=0.01 \
    --save-txt --save-conf --save-crop \
    --project ~/localdata/warp/biomap/forward/ \
    --name whoi_rs701515_mouss_all_neumann_200

python detect.py \
    --weights ~/localdata/warp/yolov5_models/whoi_rs_only_701515/weights/best.pt \
    --source ~/localdata/warp/biomap/forward/vanilla/ \
    --conf-thres=0.01 \
    --save-txt --save-conf --save-crop \
    --project ~/localdata/warp/biomap/forward/ \
    --name whoi_rs_only_701515

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

### nice strings
```
whoi_rs701515_mouss_all_neumann_200
whoi_rs_only_701515
```
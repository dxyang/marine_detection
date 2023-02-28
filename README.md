# marine_detection

## useful commands

### convert bag to mp4
```bash
python bagutils/bag2mp4.py \
    --bag_file ~/localdata/warp/biomap/warpauv_2_xavier3_2022-11-02-09-55-23-metashape-odom.bag \
    --framerate 15 \
    --output_dir ~/localdata \
    --image_topic /warpauv_2/cameras/forward/rgb/image_stream/h264

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
FRAMERATE=$1
IMAGETYPE=$2
INPUTDIR=$3

ffmpeg -framerate $FRAMERATE -pattern_type glob -i "$INPUTDIR/*.$IMAGETYPE" -c:v libx264 -pix_fmt yuv420p $INPUTDIR/clip.mp4

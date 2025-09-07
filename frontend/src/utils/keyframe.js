const parseKeyframe = (keyframe) => {
  const parts = String(keyframe).split(".")[0].split("_");
  const batch_id = parts[0];
  const video_id = parts[1];
  const index = Number(parts[2]);
  return { batch_id, video_id, index };
};

const buildImageSrcFromKeyframe = (keyframe) => {
  const { batch_id, video_id } = parseKeyframe(keyframe);
  return `keyframes/Videos_${batch_id}/${batch_id}_${video_id}/${keyframe}`;
};

const buildVideoSrcFromKeyframe = (keyframe) => {
  const { batch_id, video_id } = parseKeyframe(keyframe);
  return `videos/${batch_id}_${video_id}.mp4`;
};

export { parseKeyframe, buildImageSrcFromKeyframe, buildVideoSrcFromKeyframe };

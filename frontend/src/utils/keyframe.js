import getVideoFps from "../services/videoApi";

const parseKeyframe = (keyframe) => {
  const parts = String(keyframe).split(".")[0].split("_");
  const batch_id = parts[0];
  const video_id = parts[1];
  const index = Number(parts[2]);
  return { batch_id, video_id, index };
};

const buildImageSrcFromKeyframe = (keyframe) => {
  const { batch_id, video_id } = parseKeyframe(keyframe);
  return `/media/keyframes/Videos_${batch_id}/${batch_id}_${video_id}/${keyframe}`;
};

const buildVideoSrcFromKeyframe = (keyframe) => {
  const { batch_id, video_id } = parseKeyframe(keyframe);
  return `/media/videos/${batch_id}/${batch_id}_${video_id}.mp4`;
};

const keyframeToTimestamp = async (keyframe) => {
  const { batch_id, video_id, index } = parseKeyframe(keyframe);
  const fps = await getVideoFps(`${batch_id}_${video_id}`);
  return index / Number(fps);
};

export {
  parseKeyframe,
  buildImageSrcFromKeyframe,
  buildVideoSrcFromKeyframe,
  keyframeToTimestamp,
};

const getVideoFps = (video) => {
  return fetch(`/api/video/get-video-fps?video=${video}`).then((res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  });
};

export default getVideoFps;

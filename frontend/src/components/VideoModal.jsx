import { useEffect, useRef, useMemo, useState } from "react";
import useClickOutside from "../hooks/useClickOutside";
import {
  buildVideoSrcFromKeyframe,
  keyframeToTimestamp,
} from "../utils/keyframe";

function VideoModal({ keyframe, onClose }) {
  const wrapperRef = useRef(null);
  const videoRef = useRef(null);
  useClickOutside(wrapperRef, onClose);

  const [startTime, setStartTime] = useState(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const startTime = await keyframeToTimestamp(keyframe);
      if (cancelled) return;
      setStartTime(startTime);
    })();
    return () => {
      cancelled = true;
    };
  }, [keyframe]);

  const videoSrc = useMemo(
    () => buildVideoSrcFromKeyframe(keyframe),
    [keyframe]
  );

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const handleLoaded = () => {
    const el = videoRef.current;
    if (!el) return;
    el.currentTime = startTime;
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
      <div
        ref={wrapperRef}
        className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl overflow-hidden"
      >
        <div className="flex justify-end px-4 py-3 border-b">
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 rounded-lg bg-gray-100 hover:bg-gray-200"
          >
            Close
          </button>
        </div>

        <div className="p-4">
          <video
            ref={videoRef}
            src={videoSrc}
            className="w-full max-h-[70vh] rounded-lg"
            controls
            playsInline
            preload="metadata"
            onLoadedMetadata={handleLoaded}
          />
        </div>
      </div>
    </div>
  );
}

export default VideoModal;

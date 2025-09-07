import { useEffect, useRef, useMemo } from "react";
import useClickOutside from "../hooks/useClickOutside";
import { parseKeyframe, buildVideoSrcFromKeyframe } from "../utils/keyframe";

function VideoModel({ keyframe, onClose }) {
  const wrapperRef = useRef(null);
  const videoRef = useRef(null);
  useClickOutside(wrapperRef, onClose);

  const { video_id, index } = useMemo(
    () => parseKeyframe(keyframe),
    [keyframe]
  );

  const fps = 30;
  const startTime = index / fps;
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
        <div className="flex items-center justify-between px-4 py-3 border-b">
          <div className="text-sm font-medium text-gray-700 truncate">
            {video_id} • frame {index} (~{startTime.toFixed(2)}s @ {fps}fps)
          </div>
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
            src={videoSrc}
            ref={videoRef}
            onLoadedData={handleLoaded}
            onError={(e) => {
              e.target.src =
                "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik02MCAxMDBDODAgODAgMTIwIDgwIDE0MCAxMDBDMTYwIDEyMCAxNDAgMTQwIDEyMCAxNDBDMTAwIDE0MCA4MCAxMjAgNjAgMTAwWiIgZmlsbD0iI0QxRDVETyIvPgo8L3N2Zz4K";
            }}
            className="w-full max-h-[70vh] rounded-lg"
            controls
            playsInline
            preload="metadata"
          />
          <div className="mt-2 text-xs text-gray-500 break-all">
            Source: <code>{videoSrc}</code>
          </div>
        </div>
      </div>
    </div>
  );
}

export default VideoModel;

import { useState } from "react";
import ResultList from "./ResultList";
import ExportButton from "./ExportButton";
import VideoModal from "./VideoModal";

function ResultContainer({ keyframes }) {
  const [selectedKeyframe, setSelectedKeyframe] = useState(null);

  return (
    <div className="w-full">
      {keyframes.length > 0 && (
        <div className="flex justify-end items-center px-10 pt-6">
          <ExportButton keyframes={keyframes} />
        </div>
      )}

      {keyframes.length > 0 && (
        <ResultList
          keyframes={keyframes}
          onItemClick={(keyframe) => setSelectedKeyframe(keyframe)}
        />
      )}

      {selectedKeyframe && (
        <VideoModal
          keyframe={selectedKeyframe}
          onClose={() => setSelectedKeyframe(null)}
        />
      )}
    </div>
  );
}

export default ResultContainer;

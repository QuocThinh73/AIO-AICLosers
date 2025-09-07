import ResultItem from "./ResultItem";
import { buildImageSrcFromKeyframe } from "../utils/keyframe";

function ResultList({ keyframes, onItemClick }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6 m-10">
      {keyframes.map((keyframe, index) => (
        <ResultItem
          key={index}
          src={buildImageSrcFromKeyframe(keyframe)}
          name={keyframe}
          rank={index + 1}
          onClick={() => onItemClick(keyframe)}
        />
      ))}
    </div>
  );
}

export default ResultList;

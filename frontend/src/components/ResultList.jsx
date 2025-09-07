import ResultItem from "./ResultItem";

function ResultList({ keyframes }) {
  const buildKeyframeSource = (name) => {
    const parts = name.split("_");
    return (
      "keyframes/Videos_" +
      parts[0] +
      "/" +
      parts[0] +
      "_" +
      parts[1] +
      "/" +
      name
    );
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6 m-10">
      {keyframes.map((keyframe, index) => (
        <ResultItem
          key={index}
          src={buildKeyframeSource(keyframe)}
          name={keyframe}
          rank={index + 1}
        />
      ))}
    </div>
  );
}

export default ResultList;

import ResultList from "./ResultList";
import ExportButton from "./ExportButton";

function ResultContainer({ keyframes }) {
  return (
    <div className="w-full">
      {keyframes.length > 0 && (
        <div className="flex justify-end items-center px-10 pt-6">
          <ExportButton keyframes={keyframes} />
        </div>
      )}
      {keyframes.length > 0 && <ResultList keyframes={keyframes} />}
    </div>
  );
}

export default ResultContainer;

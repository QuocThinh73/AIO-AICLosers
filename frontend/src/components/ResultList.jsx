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
        <ul style={{ listStyle: "none", padding: 0, display: "flex", flexWrap: "wrap", gap: 12 }}>
            {keyframes.map((keyframe, index) => (
                <ResultItem key={index} src={buildKeyframeSource(keyframe)} name={keyframe} />
            ))}
        </ul>
    );
}

export default ResultList;
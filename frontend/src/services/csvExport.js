const toCsv = (rows) => {
  const body = rows.map((cols) => cols.join(", ")).join("\n") + "\n";
  return `${body}`;
};

const downloadCsv = (content, filename) => {
  const blob = new Blob(["\uFEFF" + content], {
    type: "text/csv;charset=utf-8;",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename + ".csv";
  a.click();
  URL.revokeObjectURL(url);
};

const makeKISRows = (keyframes) =>
  keyframes.map((keyframe) => {
    const parts = String(keyframe).split("_");
    const video = `${parts[0]}_${parts[1]}`;
    const index = parts[2];
    return [video, index];
  });

const makeVQARows = (keyframes, answer) =>
  keyframes.map((keyframe) => {
    const parts = String(keyframe).split("_");
    const video = `${parts[0]}_${parts[1]}`;
    const index = parts[2];
    return [video, index, answer];
  });

const exportCSV = ({ type, keyframes, filename, answer }) => {
  const rows =
    type === "KIS" ? makeKISRows(keyframes) : makeVQARows(keyframes, answer);
  const csv = toCsv(rows);
  downloadCsv(csv, filename);
};

export default exportCSV;

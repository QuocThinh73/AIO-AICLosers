const toCsv = (rows) => rows.map((cols) => cols.join(", ")).join("\n") + "\n";

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

const parseKeyframe = (keyframe) => {
  const parts = String(keyframe).split(".")[0].split("_");
  const video = `${parts[0]}_${parts[1]}`;
  const index = Number(parts[2]);
  return { video, index };
};

const makeRows = (type, payload) => {
  if (type === "KIS") {
    return payload.keyframes.map((keyframe) => {
      const { video, index } = parseKeyframe(keyframe);
      return [video, index];
    });
  }
  else if (type === "QA") {
    return payload.keyframes.map((keyframe) => {
      const { video, index } = parseKeyframe(keyframe);
      return [video, index, payload.answer];
    })
  };
};

const exportCSV = ( filename , type, payload ) => {
  const rows = makeRows(type, payload);
  const csv = toCsv(rows);
  downloadCsv(csv, filename);
};

export default exportCSV;

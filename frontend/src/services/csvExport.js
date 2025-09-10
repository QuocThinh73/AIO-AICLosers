import { parseKeyframe } from "../utils/keyframe";

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

const makeRows = (type, payload) => {
  if (type === "KIS") {
    return payload.keyframes.map((keyframe) => {
      const { batch_id, video_id, index } = parseKeyframe(keyframe);
      const video_name = `${batch_id}_${video_id}`;
      return [video_name, index];
    });
  } else if (type === "QA") {
    return payload.keyframes.map((keyframe) => {
      const { batch_id, video_id, index } = parseKeyframe(keyframe);
      const video_name = `${batch_id}_${video_id}`;
      return [video_name, index, payload.answer];
    });
  }
};

const exportCSV = (filename, type, payload) => {
  const rows = makeRows(type, payload);
  const csv = toCsv(rows);
  downloadCsv(csv, filename);
};

export default exportCSV;

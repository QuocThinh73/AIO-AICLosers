import { useState } from "react";
import exportCSV from "../services/csvExport";
import TextInput from "./TextInput";
import ExportTypeSelector from "./ExportTypeSelector";

function ExportForm({ keyframes, onClose }) {
  const [type, setType] = useState("KIS");
  const [filename, setFilename] = useState(null);
  const [answer, setAnswer] = useState(null);

  const onExport = () => {
    const payload = { keyframes, answer };
    exportCSV(filename, type, payload);
    onClose();
  };

  return (
    <>
      <ExportTypeSelector type={type} onChange={setType} />
      <TextInput label="Filename" value={filename} onChange={setFilename} />
      {type == "QA" && (
        <TextInput label="Answer" value={answer} onChange={setAnswer} />
      )}
      <div className="flex justify-end gap-2 pt-1">
        <button
          type="button"
          onClick={onClose}
          className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={onExport}
          className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:opacity-50"
        >
          Confirm
        </button>
      </div>
    </>
  );
}

export default ExportForm;

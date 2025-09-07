import { useState, useRef, useEffect } from "react";
import exportCSV from "../services/csvExport";

function ExportButton({ keyframes }) {
  const [open, setOpen] = useState(false);
  const [type, setType] = useState(null);
  const [filename, setFilename] = useState(null);
  const [answer, setAnswer] = useState(null);
  const menuRef = useRef(null);

  useEffect(() => {
    const onClick = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target))
        setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  const onExport = () => {
    const payload = { keyframes, answer };
    exportCSV(filename, type, payload);
    setOpen(false);
  };

  const onCancel = () => {
    setOpen(false);
  };

  return (
    <div className="relative inline-block text-left" ref={menuRef}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="px-4 py-2 bg-emerald-600 text-white rounded-lg shadow hover:bg-emerald-700 transition"
        title="Export CSV"
      >
        Export CSV
      </button>

      {open && (
        <div className="absolute right-0 mt-2 w-80 origin-top-right rounded-xl bg-white shadow-lg ring-1 ring-black/5 z-20 p-4">
          <div className="mb-3">
            <span className="block text-sm font-medium text-gray-700 mb-1">
              Loại export
            </span>
            <div className="flex items-center gap-4">
              <label className="inline-flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="exportType"
                  value="KIS"
                  checked={type === "KIS"}
                  onChange={() => setType("KIS")}
                />
                <span className="text-sm">KIS</span>
              </label>
              <label className="inline-flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="exportType"
                  value="QA"
                  checked={type === "QA"}
                  onChange={() => setType("QA")}
                />
                <span className="text-sm">QA</span>
              </label>
            </div>
          </div>

          <div className="mb-3">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Filename
            </label>
            <input
              type="text"
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
              className="w-full px-3 py-2 border rounded-lg border-gray-300 focus:outline-none focus:ring-2 focus:ring-emerald-500"
            />
          </div>

          {type === "QA" && (
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Answer (QA)
              </label>
              <input
                type="text"
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg border-gray-300 focus:outline-none focus:ring-2 focus:ring-emerald-500"
              />
            </div>
          )}

          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onCancel}
              className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200"
            >
              Huỷ
            </button>
            <button
              type="button"
              onClick={onExport}
              className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700"
            >
              Export
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default ExportButton;

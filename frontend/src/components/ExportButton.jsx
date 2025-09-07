import { useState } from "react";
import ExportDialog from "./ExportDialog";

function ExportButton({ keyframes }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative inline-block text-left">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="px-4 py-2 bg-emerald-600 text-white rounded-lg shadow hover:bg-emerald-700 transition"
        title="Export"
      >
        Export
      </button>

      {open && (
        <ExportDialog keyframes={keyframes} onClose={() => setOpen(false)} />
      )}
    </div>
  );
}

export default ExportButton;
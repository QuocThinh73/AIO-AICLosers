import { useRef } from "react";
import useClickOutside from "../hooks/useClickOutside";
import ExportForm from "./exportForm";

function ExportDialog({ keyframes, onClose }) {
  const ref = useRef(null);
  useClickOutside(ref, onClose);

  return (
    <div
      ref={ref}
      className="absolute right-0 mt-2 w-80 origin-top-right rounded-xl bg-white shadow-lg ring-1 ring-black/5 z-20 p-4"
    >
      <ExportForm keyframes={keyframes} onClose={onClose} />
    </div>
  );
}

export default ExportDialog;

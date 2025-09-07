function ExportTypeSelector({ type, onChange }) {
  return (
    <div className="mb-3">
      <span className="block text-sm font-medium text-gray-700 mb-1">
        Export type
      </span>
      <div className="flex items-center gap-4">
        <label className="inline-flex items-center gap-2 cursor-pointer">
          <input
            type="radio"
            name="exportType"
            value="KIS"
            checked={type === "KIS"}
            onChange={() => onChange("KIS")}
          />
          <span className="text-sm">KIS</span>
        </label>
        <label className="inline-flex items-center gap-2 cursor-pointer">
          <input
            type="radio"
            name="exportType"
            value="QA"
            checked={type === "QA"}
            onChange={() => onChange("QA")}
          />
          <span className="text-sm">QA</span>
        </label>
      </div>
    </div>
  );
}

export default ExportTypeSelector;

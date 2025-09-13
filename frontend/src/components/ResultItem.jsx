function ResultItem({ src, name, rank, onClick }) {
  return (
    <div className="group relative bg-white shadow-md">
      <div className="absolute top-2 left-2 z-10 bg-blue-600 text-white text-xs font-bold px-2 py-1 rounded-full shadow-md">
        #{rank}
      </div>

      <div className="relative">
        <img
          src={src}
          alt={name}
          onClick={onClick}
          className="w-full h-72 object-cover cursor-pointer"
          onError={(e) => {
            e.target.src =
              "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik02MCAxMDBDODAgODAgMTIwIDgwIDE0MCAxMDBDMTYwIDEyMCAxNDAgMTQwIDEyMCAxNDBDMTAwIDE0MCA4MCAxMjAgNjAgMTAwWiIgZmlsbD0iI0QxRDVETyIvPgo8L3N2Zz4K";
          }}
        />
      </div>

      <div className="p-3">
        <h3 className="text-sm font-medium text-gray-800">
          {name.split(".")[0]}
        </h3>
      </div>
    </div>
  );
}

export default ResultItem;

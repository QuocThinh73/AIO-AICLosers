import { useState } from "react";


function SearchBar({ onSearch }) {
    const [query, setQuery] = useState("");
    const [ocr, setOcr] = useState("");
    const [image, setImage] = useState(null);
    const [translate, setTranslate] = useState(true);
    const [includeBatchIds, setIncludeBatchIds] = useState("");
    const [excludeBatchIds, setExcludeBatchIds] = useState("");
    const [topK, setTopK] = useState(100);

    function handleSubmit(e) {
        e.preventDefault();
        onSearch({ 
            query, 
            ocr,
            image, 
            translate, 
            includeBatchIds, 
            excludeBatchIds, 
            topK 
        });
    }

    const handleImageUpload = (e) => {
        const file = e.target.files?.[0];
        if (file) {
            setImage(file);
        }
    };

    return (
        <div className="w-full max-w-7xl mx-auto p-6 bg-white shadow-lg border border-gray-200">
            <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="space-y-2">
                        <label htmlFor="query" className="block text-sm font-medium text-gray-700">
                            Query Search
                        </label>
                        <input
                            id="query"
                            type="text"
                            value={query}
                            onChange={e => setQuery(e.target.value)}
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                        />
                    </div>

                    <div className="space-y-2">
                        <label htmlFor="ocr" className="block text-sm font-medium text-gray-700">
                            OCR Search
                        </label>
                        <input
                            id="ocr"
                            type="text"
                            value={ocr}
                            onChange={e => setOcr(e.target.value)}
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                        />
                    </div>

                    <div className="space-y-2">
                        <label htmlFor="image" className="block text-sm font-medium text-gray-700">
                            Image Search
                        </label>
                        <div className="relative">
                            <input
                                id="image"
                                type="file"
                                accept="image/*"
                                onChange={handleImageUpload}
                                className="hidden"
                            />
                            <label
                                htmlFor="image"
                                className="flex items-center justify-center w-full px-4 py-3 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors duration-200"
                            >
                                <div className="text-center">
                                    <svg className="mx-auto h-8 w-8 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                                        <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    </svg>
                                </div>
                            </label>
                        </div>
                        {image && (
                            <p className="text-xs text-green-600 mt-1">
                                ✓ Selected: {image.name}
                            </p>
                        )}
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                    <div className="space-y-2">
                        <label className="block text-sm font-medium text-gray-700">
                            Translation
                        </label>
                        <div className="flex items-center h-12">
                            <label className="flex items-center cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={translate}
                                    onChange={e => setTranslate(e.target.checked)}
                                    className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300"
                                />
                                <span className="ml-2 text-sm text-gray-700">
                                    Translate to English
                                </span>
                            </label>
                        </div>
                    </div>

                    <div className="space-y-2">
                        <label htmlFor="includeBatchIds" className="block text-sm font-medium text-gray-700">
                            Include Batch IDs
                        </label>
                        <input
                            id="includeBatchIds"
                            type="text"
                            value={includeBatchIds}
                            onChange={e => setIncludeBatchIds(e.target.value)}
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                        />
                    </div>

                    <div className="space-y-2">
                        <label htmlFor="excludeBatchIds" className="block text-sm font-medium text-gray-700">
                            Exclude Batch IDs
                        </label>
                        <input
                            id="excludeBatchIds"
                            type="text"
                            value={excludeBatchIds}
                            onChange={e => setExcludeBatchIds(e.target.value)}
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                        />
                    </div>

                    <div className="space-y-2">
                        <label htmlFor="topK" className="block text-sm font-medium text-gray-700">
                            TopK
                        </label>
                        <input
                            id="topK"
                            type="number"
                            value={topK}
                            onChange={e => setTopK(parseInt(e.target.value || "1", 10))}
                            min={1}
                            max={1000}
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                        />
                    </div>
                </div>

                <div className="flex justify-end">
                    <button
                        type="submit"
                        className="px-8 py-3 bg-blue-600 text-white font-medium rounded-lg"
                    >
                        Search
                    </button>
                </div>
            </form>
        </div>
    );
}

export default SearchBar;
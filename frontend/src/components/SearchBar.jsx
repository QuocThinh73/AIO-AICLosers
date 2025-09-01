import { useState } from "react";


function SearchBar({ onSearch }) {
    const [query, setQuery] = useState("");
    const [image, setImage] = useState(null);
    const [translate, setTranslate] = useState(true);
    const [includeBatchIds, setIncludeBatchIds] = useState("");
    const [excludeBatchIds, setExcludeBatchIds] = useState("");
    const [topK, setTopK] = useState(100);

    function handleSubmit(e) {
        e.preventDefault();
        onSearch({ query, image, translate, includeBatchIds, excludeBatchIds, topK });
    }

    return (
        <form onSubmit={handleSubmit}>
        <div>
            <label htmlFor="query">Query: </label>
            <input id="query" value={query} onChange={e => setQuery(e.target.value)} />
        </div>

        <div>
            <label htmlFor="image">Image: </label>
            <input
            id="image"
            type="file"
            accept="image/*"
            onChange={e => setImage(e.target.files?.[0] ?? null)}
            />
        </div>

        <div>
            <label htmlFor="translate">Translate to English: </label>
            <input
            id="translate"
            type="checkbox"
            checked={translate}
            onChange={e => setTranslate(e.target.checked)}
            />
        </div>

        <div>
            <label htmlFor="topk">Number of results: </label>
            <input
            id="topk"
            type="number"
            value={topK}
            onChange={e => setTopK(parseInt(e.target.value || "0", 10))}
            min={1}
            max={100}
            />
        </div>

        <div>
            <label htmlFor="includeBatchIds">Include batch IDs: </label>
            <input
            id="includeBatchIds"
            value={includeBatchIds}
            onChange={e => setIncludeBatchIds(e.target.value)}
            placeholder="L21,L22"
            />
        </div>

        <div>
            <label htmlFor="excludeBatchIds">Exclude batch IDs: </label>
            <input
            id="excludeBatchIds"
            value={excludeBatchIds}
            onChange={e => setExcludeBatchIds(e.target.value)}
            />
        </div>

        <button type="submit">
            Search
        </button>
        </form>
    );
}

export default SearchBar;
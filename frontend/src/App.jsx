import { useState } from "react"

function App() {
  const [query, setQuery] = useState("")
  const [image, setImage] = useState(null)
  const [translate, setTranslate] = useState(true)
  const [includeIds, setIncludeIds] = useState("")
  const [excludeIds, setExcludeIds] = useState("")
  const [topK, setTopK] = useState(100)

  const [keyframes, setKeyframes] = useState([])
  const [error, setError] = useState("")

  async function handleSubmit(e) {
    e.preventDefault()
    setError("")
    setKeyframes([])

    try {
      const url = "/api/search/base_search"
      
      const fd = new FormData()

      fd.append("use_translation", translate ? "true" : "false")
      if (query.trim()) fd.append("embedding_text", query.trim())
      if (image) fd.append("embedding_image", image)
      fd.append("top_k", String(topK))

      if (includeIds) fd.append("include_batch_ids", includeIds)
      if (excludeIds) fd.append("exclude_batch_ids", excludeIds)

      const res = await fetch(url, {method: "POST", body: fd })
      if (!res.ok) throw new Error("HTTP ${res.status}: ${await res.text()}")
      
      const data = await res.json()
      setKeyframes(Array.from(data))
    } catch (error) {
      setError(error.message)
    }
  }

  const buildKeyframePath = (name) => {
    const parts = name.split("_")
    return "keyframes/Videos_" + parts[0] + "/" + parts[0] + "_" + parts[1] + "/" + name
  }

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="query">Query: </label>
          <input id="query" value={query} onChange={(e) => setQuery(e.target.value)} />
        </div>

        <div>
          <label htmlFor="image">Image: </label>
          <input id="image" type="file" accept="image/*" onChange={(e) => setImage(e.target.files[0])} />
        </div>
        
        <div>
          <label htmlFor="translate">Translate to English: </label>
          <input id="translate" type="checkbox" checked={translate} onChange={(e) => setTranslate(e.target.checked)} />
        </div>

        <div>
          <label htmlFor="topk">Number of results: </label>
          <input id="topk" type="number" value={topK} onChange={(e) => setTopK(e.target.value)} />
        </div>

        <div>
          <label htmlFor="includeIds">Include batch IDs: </label>
          <input id="includeIds" value={includeIds} onChange={(e) => setIncludeIds(e.target.value)} />
        </div>

        <div>
          <label htmlFor="excludeIds">Exclude batch IDs: </label>
          <input id="excludeIds" value={excludeIds} onChange={(e) => setExcludeIds(e.target.value)} />
        </div>

        <button type="submit">Search</button>
      </form>

      {error && <p>{error}</p>}

      {keyframes.length > 0 && (
        <ul style={{ listStyle: "none", padding: 0 }}>
          {keyframes.map((name) => {
            const src = buildKeyframePath(name);
            return (
              <li key={name} style={{ display: "flex", alignItems: "center", gap: 12, margin: "8px 0" }}>
                <img
                  src={src}
                  alt={name}
                  width={700}
                  height={450}
                  style={{ objectFit: "cover", borderRadius: 6, border: "1px solid #ddd" }}
                  onError={(e) => { e.currentTarget.style.opacity = 0.3; }}
                />
                <span>{name}</span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  )
}

export default App
 